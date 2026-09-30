"""
LT3 tender-pricing / execution engine for Rotman Interactive Trader.
Version: 1.0

Designed for the Liability Trading 3 case:
- Reads active tenders.
- Prices the full unwind against the live order book.
- Checks portfolio net/gross limits.
- Accepts only if the current book implies enough edge after costs/buffer.
- ONLY trades after a tender has been accepted.
- Flattens accepted-tender inventory with market orders, respecting LT3 max order sizes.
- Logs every decision to lt3_log.csv.

IMPORTANT:
Start with LIVE = False. Run several demo heats and inspect the printed decisions.
Only switch LIVE = True after confirming your API key, tender semantics, fees, and execution behavior.
"""

import csv
import os
import time
from pathlib import Path
import requests

# --------------------------- CONFIG ---------------------------------

PORT = 9999
API_KEY = "YLSSSM7U"
BASE = f"http://localhost:{PORT}/v1"

# False = observe/score tenders only; no accepts/declines/orders are sent.
# True  = automatically accept/decline and flatten accepted tender inventory.
LIVE = False

POLL_SECONDS = 0.20

# LT3 case parameters from the case brief.
NET_LIMIT = 100_000
GROSS_LIMIT = 250_000
MAX_ORDER_SIZE = {
    "CRZY": 25_000,
    "TAME": 10_000,
}

# The case brief states a $0.02/share transaction fee.
# This engine conservatively charges it to the open-market unwind.
# If your demo P&L shows the tender itself is also charged, set TENDER_FEE_PER_SHARE = 0.02.
MARKET_FEE_PER_SHARE = 0.02
TENDER_FEE_PER_SHARE = 0.00

# Extra cushion for book movement between snapshot and execution.
SAFETY_BUFFER_PER_SHARE = 0.03

# Require BOTH minimum per-share edge and minimum total expected profit.
MIN_NET_EDGE_PER_SHARE = 0.03
MIN_EXPECTED_PROFIT = 750.0

# Get flat before the 300-tick period ends.
HARD_FLATTEN_TICK = 285

LOG_FILE = Path("lt3_log.csv")

# -------------------------------------------------------------------


class RITError(RuntimeError):
    pass


def session():
    s = requests.Session()
    s.headers.update({"X-API-Key": API_KEY})
    return s


def request(s, method, path, **kwargs):
    """HTTP request wrapper with basic RIT 429 handling."""
    url = BASE + path
    while True:
        r = s.request(method, url, timeout=2, **kwargs)

        if r.status_code == 429:
            wait = 0.25
            try:
                body = r.json()
                wait = float(body.get("wait", wait))
            except Exception:
                try:
                    wait = float(r.headers.get("Retry-After", wait))
                except Exception:
                    pass
            time.sleep(max(wait, 0.05))
            continue

        if r.status_code == 401:
            raise RITError(
                "401 Unauthorized. Click the API icon in RIT and copy the exact API key "
                "into API_KEY. Also confirm the local API port."
            )

        r.raise_for_status()
        return r


def get_case(s):
    return request(s, "GET", "/case").json()


def get_tenders(s):
    return request(s, "GET", "/tenders").json()


def get_securities(s):
    return request(s, "GET", "/securities").json()


def get_book(s, ticker):
    return request(
        s, "GET", "/securities/book", params={"ticker": ticker}
    ).json()


def available_quantity(order):
    return max(0, int(order.get("quantity", 0) - order.get("quantity_filled", 0)))


def sweep_vwap(book_side, quantity):
    """
    Price an immediate market sweep through bids or asks.
    Returns (vwap, worst_price, visible_quantity_used, fully_covered).
    """
    remaining = int(quantity)
    notional = 0.0
    used = 0
    worst = None

    for level in book_side:
        avail = available_quantity(level)
        if avail <= 0:
            continue

        take = min(remaining, avail)
        price = float(level["price"])

        notional += take * price
        used += take
        remaining -= take
        worst = price

        if remaining <= 0:
            break

    if used == 0:
        return None, None, 0, False

    return notional / used, worst, used, remaining == 0


def positions(s):
    result = {}
    for sec in get_securities(s):
        ticker = sec.get("ticker")
        if ticker in MAX_ORDER_SIZE:
            result[ticker] = int(sec.get("position", 0))
    for ticker in MAX_ORDER_SIZE:
        result.setdefault(ticker, 0)
    return result


def risk_after_tender(current_positions, ticker, action, quantity):
    """
    Tender action is from OUR perspective:
      BUY  -> accepting makes us long quantity
      SELL -> accepting makes us short quantity
    """
    p = dict(current_positions)
    signed = quantity if action == "BUY" else -quantity
    p[ticker] = p.get(ticker, 0) + signed

    net = sum(p.values())
    gross = sum(abs(x) for x in p.values())
    return p, net, gross


def analyze_tender(s, tender):
    ticker = tender["ticker"]
    action = tender["action"].upper()
    qty = int(tender["quantity"])
    tender_price = float(tender["price"])

    current_pos = positions(s)
    projected, projected_net, projected_gross = risk_after_tender(
        current_pos, ticker, action, qty
    )

    book = get_book(s, ticker)

    # If we BUY the tender, we become long and must SELL into bids.
    # If we SELL the tender, we become short and must BUY from asks.
    if action == "BUY":
        unwind_side = "SELL"
        book_side = book["bids"]
        unwind_vwap, worst, visible, covered = sweep_vwap(book_side, qty)
        raw_edge = None if unwind_vwap is None else unwind_vwap - tender_price
    elif action == "SELL":
        unwind_side = "BUY"
        book_side = book["asks"]
        unwind_vwap, worst, visible, covered = sweep_vwap(book_side, qty)
        raw_edge = None if unwind_vwap is None else tender_price - unwind_vwap
    else:
        raise RITError(f"Unknown tender action: {action}")

    costs = MARKET_FEE_PER_SHARE + TENDER_FEE_PER_SHARE
    if raw_edge is None:
        net_edge = None
        expected_profit = None
    else:
        net_edge = raw_edge - costs - SAFETY_BUFFER_PER_SHARE
        expected_profit = net_edge * qty

    limit_ok = (
        abs(projected_net) <= NET_LIMIT
        and projected_gross <= GROSS_LIMIT
    )

    edge_ok = (
        net_edge is not None
        and net_edge >= MIN_NET_EDGE_PER_SHARE
        and expected_profit >= MIN_EXPECTED_PROFIT
    )

    accept = bool(covered and limit_ok and edge_ok)

    reasons = []
    if not covered:
        reasons.append(f"visible book covers only {visible:,}/{qty:,}")
    if not limit_ok:
        reasons.append(
            f"projected limits net={projected_net:,}, gross={projected_gross:,}"
        )
    if net_edge is None:
        reasons.append("no executable book liquidity")
    elif not edge_ok:
        reasons.append(
            f"net edge ${net_edge:.4f}/sh, expected ${expected_profit:,.0f}"
        )

    return {
        "tender_id": int(tender["tender_id"]),
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "tender_price": tender_price,
        "unwind_side": unwind_side,
        "unwind_vwap": unwind_vwap,
        "worst_price": worst,
        "visible_qty": visible,
        "covered": covered,
        "raw_edge": raw_edge,
        "net_edge": net_edge,
        "expected_profit": expected_profit,
        "current_positions": current_pos,
        "projected_positions": projected,
        "projected_net": projected_net,
        "projected_gross": projected_gross,
        "accept": accept,
        "reason": "; ".join(reasons) if reasons else "passes filters",
    }


def print_analysis(a, tick, expires):
    decision = "ACCEPT" if a["accept"] else "DECLINE"
    print("\n" + "=" * 78)
    print(
        f"TICK {tick:>3} | Tender {a['tender_id']} | "
        f"{a['action']} {a['qty']:,} {a['ticker']} @ {a['tender_price']:.2f}"
    )
    print(f"Expires: {expires}")
    print(
        f"Immediate unwind: {a['unwind_side']} {a['qty']:,} | "
        f"VWAP={a['unwind_vwap'] if a['unwind_vwap'] is not None else 'N/A'} | "
        f"Worst={a['worst_price'] if a['worst_price'] is not None else 'N/A'}"
    )
    if a["raw_edge"] is not None:
        print(
            f"Raw edge: ${a['raw_edge']:.4f}/sh | "
            f"After fee + buffer: ${a['net_edge']:.4f}/sh | "
            f"Expected: ${a['expected_profit']:,.0f}"
        )
    print(
        f"Projected risk: net {a['projected_net']:,}/{NET_LIMIT:,} | "
        f"gross {a['projected_gross']:,}/{GROSS_LIMIT:,}"
    )
    print(f">>> {decision}: {a['reason']}")
    print("=" * 78)


def log_decision(tick, expires, a):
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "tick", "expires", "tender_id", "ticker", "action", "quantity",
                "tender_price", "unwind_side", "unwind_vwap", "worst_price",
                "raw_edge_per_share", "net_edge_per_share", "expected_profit",
                "projected_net", "projected_gross", "decision", "reason"
            ])
        w.writerow([
            tick, expires, a["tender_id"], a["ticker"], a["action"], a["qty"],
            a["tender_price"], a["unwind_side"], a["unwind_vwap"],
            a["worst_price"], a["raw_edge"], a["net_edge"],
            a["expected_profit"], a["projected_net"], a["projected_gross"],
            "ACCEPT" if a["accept"] else "DECLINE", a["reason"]
        ])


def accept_tender(s, tender):
    # RIT REST API v1.0.4 expects price even for fixed-price tenders.
    r = request(
        s,
        "POST",
        f"/tenders/{int(tender['tender_id'])}",
        params={"price": float(tender["price"])},
    )
    return r.json()


def decline_tender(s, tender_id):
    return request(s, "DELETE", f"/tenders/{int(tender_id)}").json()


def send_market_order(s, ticker, action, qty):
    return request(
        s,
        "POST",
        "/orders",
        params={
            "ticker": ticker,
            "type": "MARKET",
            "quantity": int(qty),
            "action": action,
        },
    ).json()


def flatten_ticker(s, ticker):
    """
    Target zero inventory after an accepted tender.
    Re-queries the position after every market-order slice.
    """
    max_size = MAX_ORDER_SIZE[ticker]

    for _ in range(30):
        pos = positions(s)[ticker]
        if pos == 0:
            print(f"{ticker}: flat.")
            return

        action = "SELL" if pos > 0 else "BUY"
        qty = min(abs(pos), max_size)

        order = send_market_order(s, ticker, action, qty)
        print(
            f"  {action} {qty:,} {ticker} | "
            f"order_id={order.get('order_id')} | "
            f"vwap={order.get('vwap')}"
        )
        time.sleep(0.05)

    raise RITError(f"Could not flatten {ticker} after repeated attempts.")


def flatten_all(s):
    p = positions(s)
    for ticker, pos in p.items():
        if pos:
            print(f"HARD FLATTEN: {ticker} position {pos:,}")
            flatten_ticker(s, ticker)


def main():
    if API_KEY == "PASTE_YOUR_RIT_API_KEY_HERE":
        raise SystemExit(
            "Open RIT -> click the API icon -> copy the API key into API_KEY first."
        )

    seen = set()

    with session() as s:
        print("Connected to LT3 engine.")
        print(f"LIVE={LIVE}  BASE={BASE}")
        print(
            f"Decision rule: edge >= ${MIN_NET_EDGE_PER_SHARE:.2f}/sh "
            f"and expected >= ${MIN_EXPECTED_PROFIT:,.0f}, "
            f"after ${MARKET_FEE_PER_SHARE + TENDER_FEE_PER_SHARE:.2f}/sh costs "
            f"+ ${SAFETY_BUFFER_PER_SHARE:.2f}/sh safety buffer."
        )

        while True:
            case = get_case(s)
            tick = int(case.get("tick", 0))
            status = str(case.get("status", ""))

            if status != "ACTIVE":
                seen.clear()
                time.sleep(0.5)
                continue

            if tick >= HARD_FLATTEN_TICK:
                if LIVE:
                    flatten_all(s)
                else:
                    p = positions(s)
                    if any(p.values()):
                        print(f"[PAPER] HARD FLATTEN would run now: {p}")
                time.sleep(0.25)
                continue

            for tender in get_tenders(s):
                tid = int(tender["tender_id"])
                if tid in seen:
                    continue

                seen.add(tid)
                a = analyze_tender(s, tender)
                expires = tender.get("expires")
                print_analysis(a, tick, expires)
                log_decision(tick, expires, a)

                if not LIVE:
                    print("[PAPER MODE] No tender/order action sent.")
                    continue

                if a["accept"]:
                    # Legal sequencing: ACCEPT FIRST, then trade to unwind.
                    result = accept_tender(s, tender)
                    if not result.get("success", False):
                        print(f"Tender acceptance failed: {result}")
                        continue

                    print("Tender accepted. Now flattening resulting inventory...")
                    flatten_ticker(s, a["ticker"])
                else:
                    result = decline_tender(s, tid)
                    print(f"Tender declined: {result}")

            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")

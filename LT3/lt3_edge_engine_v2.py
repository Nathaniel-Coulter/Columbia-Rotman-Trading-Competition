"""
LT3 tender-pricing / research / execution engine for Rotman Interactive Trader.
Version: 2.0

What changed vs v1:
- Logs always go beside this script in ./logs, regardless of where PowerShell was opened.
- Appends across unlimited script restarts and unlimited back-to-back LT3 heats.
- Detects each new 5-minute heat and assigns run_id + heat_id.
- Stores a compact tender master CSV for easy inspection.
- Stores a persistent SQLite research database containing:
    * heats
    * tenders
    * follow-up snapshots
    * order-book levels
- After every tender, tracks the market for 30 ticks (default: once per tick).
- Records full book depth (capped by BOOK_LEVELS_TO_STORE) during those follow-ups.
- Records full-tender and 25/50/75% unwind VWAPs, spread, midpoint, imbalance,
  profitable visible liquidity, and counterfactual "liquidate now" P&L.
- Keeps LIVE=False by default. Research collection requires no order submission.

The decision rule itself is intentionally still simple:
full-book immediate unwind VWAP -> fees -> safety buffer -> thresholds -> risk limits.

Do not turn LIVE=True until paper logging has been validated and the decision/execution
logic has been tuned from the collected data.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests


# ============================================================================
# CONFIG
# ============================================================================

PORT = 9999

# Safer than hard-coding a key into a file you may later share.
# Either:
#   1) paste your key below, OR
#   2) set PowerShell env var RIT_API_KEY before running.
API_KEY = os.getenv("RIT_API_KEY", "PASTE_YOUR_RIT_API_KEY_HERE")

BASE = f"http://localhost:{PORT}/v1"

# False = data collection / paper scoring only.
# True  = auto accept/decline + immediate market-order flattening.
LIVE = False

# Main polling rate. RIT 429 responses are handled automatically.
POLL_SECONDS = 0.20

# LT3 case parameters from the case brief.
NET_LIMIT = 100_000
GROSS_LIMIT = 250_000
MAX_ORDER_SIZE = {
    "CRZY": 25_000,
    "TAME": 10_000,
}

# Fee assumptions. Keep components separate so later analysis can be changed
# without losing the raw book/tender data.
MARKET_FEE_PER_SHARE = 0.02
TENDER_FEE_PER_SHARE = 0.00

# Current conservative scoring rule.
SAFETY_BUFFER_PER_SHARE = 0.03
MIN_NET_EDGE_PER_SHARE = 0.03
MIN_EXPECTED_PROFIT = 750.0

# Live-mode emergency flatten.
HARD_FLATTEN_TICK = 285

# Research logging.
FOLLOWUP_TICKS = 30
FOLLOWUP_EVERY_TICKS = 1
BOOK_LEVELS_TO_STORE = 60

# Script-relative paths. If the script is:
#   ...\columbia\lt3_edge_engine.py
# logs will always be:
#   ...\columbia\logs\
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = LOG_DIR / "lt3_training.sqlite"
TENDER_CSV = LOG_DIR / "lt3_tenders.csv"

# Unique process ID. Heat IDs below this make separate script restarts distinguishable.
RUN_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + uuid.uuid4().hex[:8]
)

shutdown = False


# ============================================================================
# BASIC UTILITIES
# ============================================================================

class RITError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def signal_handler(signum, frame):
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True


def rit_session():
    s = requests.Session()
    s.headers.update({"X-API-Key": API_KEY})
    return s


def request(s, method, path, **kwargs):
    """HTTP request wrapper with basic RIT 429 handling."""
    url = BASE + path

    while not shutdown:
        try:
            r = s.request(method, url, timeout=2, **kwargs)
        except requests.RequestException:
            time.sleep(0.25)
            continue

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
                "401 Unauthorized. Click the Client REST API icon in RIT and copy "
                "the exact API key into API_KEY (or RIT_API_KEY). Also confirm PORT."
            )

        r.raise_for_status()
        return r

    raise KeyboardInterrupt


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


def available_quantity(order) -> int:
    return max(
        0,
        int(order.get("quantity", 0))
        - int(order.get("quantity_filled", 0)),
    )


def positions_from_securities(securities):
    result = {}
    for sec in securities:
        ticker = sec.get("ticker")
        if ticker in MAX_ORDER_SIZE:
            result[ticker] = int(sec.get("position", 0))
    for ticker in MAX_ORDER_SIZE:
        result.setdefault(ticker, 0)
    return result


def get_positions(s):
    return positions_from_securities(get_securities(s))


# ============================================================================
# DATABASE / CSV
# ============================================================================

def open_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS heats (
            heat_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            heat_seq INTEGER NOT NULL,
            started_at_utc TEXT NOT NULL,
            ended_at_utc TEXT,
            start_tick INTEGER,
            end_tick INTEGER,
            start_period INTEGER,
            end_period INTEGER,
            end_reason TEXT,
            tender_count INTEGER DEFAULT 0,
            paper_accept_count INTEGER DEFAULT 0,
            paper_decline_count INTEGER DEFAULT 0,
            scored_profit_sum REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS tenders (
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            tick INTEGER NOT NULL,
            expires INTEGER,
            period INTEGER,
            tender_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            tender_price REAL NOT NULL,

            top_bid REAL,
            top_ask REAL,
            midpoint REAL,
            spread REAL,

            unwind_side TEXT NOT NULL,
            unwind_vwap REAL,
            worst_price REAL,
            visible_qty INTEGER,
            fully_covered INTEGER,

            raw_edge_per_share REAL,
            fee_adjusted_edge_per_share REAL,
            decision_edge_per_share REAL,

            immediate_profit_after_fees REAL,
            scored_profit REAL,

            vwap_25 REAL,
            vwap_50 REAL,
            vwap_75 REAL,
            vwap_100 REAL,

            imbalance_top5 REAL,
            imbalance_top10 REAL,
            imbalance_full REAL,

            breakeven_visible_qty INTEGER,
            buffered_visible_qty INTEGER,
            threshold_visible_qty INTEGER,

            actual_positions_json TEXT,
            projected_positions_json TEXT,
            projected_net INTEGER,
            projected_gross INTEGER,

            decision TEXT NOT NULL,
            reason TEXT,
            live_mode INTEGER NOT NULL,

            PRIMARY KEY (run_id, heat_id, tender_id)
        );

        CREATE TABLE IF NOT EXISTS followups (
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            observed_at_utc TEXT NOT NULL,

            arrival_tick INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            elapsed_ticks INTEGER NOT NULL,

            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            tender_price REAL NOT NULL,

            top_bid REAL,
            top_ask REAL,
            midpoint REAL,
            spread REAL,

            unwind_vwap REAL,
            worst_price REAL,
            visible_qty INTEGER,
            fully_covered INTEGER,

            raw_edge_per_share REAL,
            fee_adjusted_edge_per_share REAL,
            counterfactual_profit_after_fees REAL,

            vwap_25 REAL,
            vwap_50 REAL,
            vwap_75 REAL,
            vwap_100 REAL,

            imbalance_top5 REAL,
            imbalance_top10 REAL,
            imbalance_full REAL,

            breakeven_visible_qty INTEGER,
            buffered_visible_qty INTEGER,
            threshold_visible_qty INTEGER,

            PRIMARY KEY (run_id, heat_id, tender_id, elapsed_ticks)
        );

        CREATE TABLE IF NOT EXISTS book_levels (
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            observed_at_utc TEXT NOT NULL,
            tick INTEGER NOT NULL,
            elapsed_ticks INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            level_rank INTEGER NOT NULL,
            price REAL NOT NULL,
            available_qty INTEGER NOT NULL,
            cumulative_qty INTEGER NOT NULL,
            cumulative_vwap REAL NOT NULL,

            PRIMARY KEY (
                run_id, heat_id, tender_id, elapsed_ticks, side, level_rank
            )
        );

        CREATE INDEX IF NOT EXISTS idx_tenders_heat
            ON tenders(heat_id, tick);

        CREATE INDEX IF NOT EXISTS idx_followups_tender
            ON followups(heat_id, tender_id, elapsed_ticks);

        CREATE INDEX IF NOT EXISTS idx_book_tender
            ON book_levels(heat_id, tender_id, elapsed_ticks, side, level_rank);
        """
    )
    conn.commit()
    return conn


TENDER_CSV_FIELDS = [
    "run_id", "heat_id", "observed_at_utc", "tick", "expires", "period",
    "tender_id", "ticker", "action", "quantity", "tender_price",
    "top_bid", "top_ask", "midpoint", "spread",
    "unwind_side", "unwind_vwap", "worst_price", "visible_qty",
    "fully_covered",
    "raw_edge_per_share", "fee_adjusted_edge_per_share",
    "decision_edge_per_share",
    "immediate_profit_after_fees", "scored_profit",
    "vwap_25", "vwap_50", "vwap_75", "vwap_100",
    "imbalance_top5", "imbalance_top10", "imbalance_full",
    "breakeven_visible_qty", "buffered_visible_qty",
    "threshold_visible_qty",
    "projected_net", "projected_gross",
    "decision", "reason", "live_mode",
]


def append_tender_csv(row):
    new_file = not TENDER_CSV.exists()
    with TENDER_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TENDER_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in TENDER_CSV_FIELDS})


def insert_heat(conn, heat):
    conn.execute(
        """
        INSERT OR REPLACE INTO heats (
            heat_id, run_id, heat_seq, started_at_utc,
            start_tick, start_period
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            heat["heat_id"], RUN_ID, heat["heat_seq"],
            heat["started_at_utc"], heat["start_tick"], heat["start_period"],
        ),
    )
    conn.commit()


def finalize_heat(conn, heat, end_tick, end_period, reason):
    if not heat:
        return
    conn.execute(
        """
        UPDATE heats
        SET ended_at_utc = ?,
            end_tick = ?,
            end_period = ?,
            end_reason = ?,
            tender_count = ?,
            paper_accept_count = ?,
            paper_decline_count = ?,
            scored_profit_sum = ?
        WHERE heat_id = ?
        """,
        (
            utc_now(),
            end_tick,
            end_period,
            reason,
            heat["tender_count"],
            heat["accept_count"],
            heat["decline_count"],
            heat["scored_profit_sum"],
            heat["heat_id"],
        ),
    )
    conn.commit()


def insert_tender(conn, row):
    conn.execute(
        """
        INSERT OR REPLACE INTO tenders (
            run_id, heat_id, observed_at_utc, tick, expires, period,
            tender_id, ticker, action, quantity, tender_price,
            top_bid, top_ask, midpoint, spread,
            unwind_side, unwind_vwap, worst_price, visible_qty, fully_covered,
            raw_edge_per_share, fee_adjusted_edge_per_share,
            decision_edge_per_share,
            immediate_profit_after_fees, scored_profit,
            vwap_25, vwap_50, vwap_75, vwap_100,
            imbalance_top5, imbalance_top10, imbalance_full,
            breakeven_visible_qty, buffered_visible_qty,
            threshold_visible_qty,
            actual_positions_json, projected_positions_json,
            projected_net, projected_gross,
            decision, reason, live_mode
        ) VALUES (
            ?,?,?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,
            ?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,
            ?,?,
            ?,?,?
        )
        """,
        (
            row["run_id"], row["heat_id"], row["observed_at_utc"],
            row["tick"], row["expires"], row["period"],
            row["tender_id"], row["ticker"], row["action"],
            row["quantity"], row["tender_price"],
            row["top_bid"], row["top_ask"], row["midpoint"], row["spread"],
            row["unwind_side"], row["unwind_vwap"], row["worst_price"],
            row["visible_qty"], int(row["fully_covered"]),
            row["raw_edge_per_share"],
            row["fee_adjusted_edge_per_share"],
            row["decision_edge_per_share"],
            row["immediate_profit_after_fees"], row["scored_profit"],
            row["vwap_25"], row["vwap_50"], row["vwap_75"], row["vwap_100"],
            row["imbalance_top5"], row["imbalance_top10"],
            row["imbalance_full"],
            row["breakeven_visible_qty"],
            row["buffered_visible_qty"],
            row["threshold_visible_qty"],
            json.dumps(row["actual_positions"], sort_keys=True),
            json.dumps(row["projected_positions"], sort_keys=True),
            row["projected_net"], row["projected_gross"],
            row["decision"], row["reason"], int(row["live_mode"]),
        ),
    )
    conn.commit()


def insert_followup(conn, row):
    conn.execute(
        """
        INSERT OR REPLACE INTO followups (
            run_id, heat_id, tender_id, observed_at_utc,
            arrival_tick, tick, elapsed_ticks,
            ticker, action, quantity, tender_price,
            top_bid, top_ask, midpoint, spread,
            unwind_vwap, worst_price, visible_qty, fully_covered,
            raw_edge_per_share, fee_adjusted_edge_per_share,
            counterfactual_profit_after_fees,
            vwap_25, vwap_50, vwap_75, vwap_100,
            imbalance_top5, imbalance_top10, imbalance_full,
            breakeven_visible_qty, buffered_visible_qty,
            threshold_visible_qty
        ) VALUES (
            ?,?,?,?,
            ?,?,?,
            ?,?,?,?,
            ?,?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?
        )
        """,
        (
            row["run_id"], row["heat_id"], row["tender_id"],
            row["observed_at_utc"],
            row["arrival_tick"], row["tick"], row["elapsed_ticks"],
            row["ticker"], row["action"], row["quantity"], row["tender_price"],
            row["top_bid"], row["top_ask"], row["midpoint"], row["spread"],
            row["unwind_vwap"], row["worst_price"], row["visible_qty"],
            int(row["fully_covered"]),
            row["raw_edge_per_share"], row["fee_adjusted_edge_per_share"],
            row["counterfactual_profit_after_fees"],
            row["vwap_25"], row["vwap_50"], row["vwap_75"], row["vwap_100"],
            row["imbalance_top5"], row["imbalance_top10"],
            row["imbalance_full"],
            row["breakeven_visible_qty"], row["buffered_visible_qty"],
            row["threshold_visible_qty"],
        ),
    )
    conn.commit()


def insert_book_levels(
    conn, heat_id, tender_id, observed_at, tick, elapsed_ticks, ticker, book
):
    rows = []

    for side_name, side_key in (("BID", "bids"), ("ASK", "asks")):
        cumulative_qty = 0
        cumulative_notional = 0.0

        for rank, level in enumerate(book.get(side_key, [])[:BOOK_LEVELS_TO_STORE], 1):
            qty = available_quantity(level)
            if qty <= 0:
                continue

            price = float(level["price"])
            cumulative_qty += qty
            cumulative_notional += qty * price
            cumulative_vwap = cumulative_notional / cumulative_qty

            rows.append(
                (
                    RUN_ID, heat_id, tender_id, observed_at,
                    tick, elapsed_ticks, ticker, side_name, rank,
                    price, qty, cumulative_qty, cumulative_vwap,
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO book_levels (
            run_id, heat_id, tender_id, observed_at_utc,
            tick, elapsed_ticks, ticker, side, level_rank,
            price, available_qty, cumulative_qty, cumulative_vwap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


# ============================================================================
# MARKET / BOOK FEATURES
# ============================================================================

def sweep_vwap(book_side, quantity):
    """
    Price an immediate sweep through a side of the book.

    Returns:
        vwap, worst_price, visible_quantity_used, fully_covered
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


def side_total(levels, n=None):
    selected = levels if n is None else levels[:n]
    return sum(available_quantity(x) for x in selected)


def imbalance(book, n=None):
    b = side_total(book.get("bids", []), n)
    a = side_total(book.get("asks", []), n)
    denom = b + a
    return None if denom == 0 else (b - a) / denom


def first_price(levels):
    for level in levels:
        if available_quantity(level) > 0:
            return float(level["price"])
    return None


def profitable_visible_qty(
    book_side,
    action,
    tender_price,
    required_edge_per_share,
):
    """
    Visible quantity currently executable at or better than a specified edge.

    action == BUY: we would own the tender and SELL into bids.
    action == SELL: we would be short the tender and BUY from asks.
    """
    qty = 0

    if action == "BUY":
        threshold = tender_price + required_edge_per_share
        for level in book_side:
            if float(level["price"]) + 1e-12 >= threshold:
                qty += available_quantity(level)

    elif action == "SELL":
        threshold = tender_price - required_edge_per_share
        for level in book_side:
            if float(level["price"]) - 1e-12 <= threshold:
                qty += available_quantity(level)

    return qty


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


def book_features(tender, book):
    ticker = tender["ticker"]
    action = tender["action"].upper()
    qty = int(tender["quantity"])
    tender_price = float(tender["price"])

    bids = book.get("bids", [])
    asks = book.get("asks", [])

    top_bid = first_price(bids)
    top_ask = first_price(asks)

    if top_bid is not None and top_ask is not None:
        midpoint = (top_bid + top_ask) / 2.0
        spread = top_ask - top_bid
    else:
        midpoint = None
        spread = None

    if action == "BUY":
        unwind_side = "SELL"
        unwind_book = bids
    elif action == "SELL":
        unwind_side = "BUY"
        unwind_book = asks
    else:
        raise RITError(f"Unknown tender action: {action}")

    full_vwap, worst, visible, covered = sweep_vwap(unwind_book, qty)

    fraction_vwaps = {}
    for frac, name in ((0.25, "vwap_25"), (0.50, "vwap_50"),
                       (0.75, "vwap_75"), (1.00, "vwap_100")):
        q = max(1, math.ceil(qty * frac))
        vwap, _, _, full = sweep_vwap(unwind_book, q)
        fraction_vwaps[name] = vwap if full else None

    if full_vwap is None:
        raw_edge = None
    elif action == "BUY":
        raw_edge = full_vwap - tender_price
    else:
        raw_edge = tender_price - full_vwap

    fees = MARKET_FEE_PER_SHARE + TENDER_FEE_PER_SHARE
    fee_edge = None if raw_edge is None else raw_edge - fees
    decision_edge = (
        None if fee_edge is None else fee_edge - SAFETY_BUFFER_PER_SHARE
    )

    breakeven_qty = profitable_visible_qty(
        unwind_book,
        action,
        tender_price,
        required_edge_per_share=fees,
    )

    buffered_qty = profitable_visible_qty(
        unwind_book,
        action,
        tender_price,
        required_edge_per_share=fees + SAFETY_BUFFER_PER_SHARE,
    )

    threshold_qty = profitable_visible_qty(
        unwind_book,
        action,
        tender_price,
        required_edge_per_share=(
            fees + SAFETY_BUFFER_PER_SHARE + MIN_NET_EDGE_PER_SHARE
        ),
    )

    return {
        "ticker": ticker,
        "action": action,
        "quantity": qty,
        "tender_price": tender_price,
        "unwind_side": unwind_side,

        "top_bid": top_bid,
        "top_ask": top_ask,
        "midpoint": midpoint,
        "spread": spread,

        "unwind_vwap": full_vwap,
        "worst_price": worst,
        "visible_qty": visible,
        "fully_covered": covered,

        "raw_edge_per_share": raw_edge,
        "fee_adjusted_edge_per_share": fee_edge,
        "decision_edge_per_share": decision_edge,

        "immediate_profit_after_fees": (
            None if fee_edge is None else fee_edge * qty
        ),
        "scored_profit": (
            None if decision_edge is None else decision_edge * qty
        ),

        **fraction_vwaps,

        "imbalance_top5": imbalance(book, 5),
        "imbalance_top10": imbalance(book, 10),
        "imbalance_full": imbalance(book, None),

        "breakeven_visible_qty": breakeven_qty,
        "buffered_visible_qty": buffered_qty,
        "threshold_visible_qty": threshold_qty,
    }


def analyze_tender(s, tender, heat_id, tick, period):
    observed_at = utc_now()
    securities = get_securities(s)
    current_pos = positions_from_securities(securities)

    ticker = tender["ticker"]
    action = tender["action"].upper()
    qty = int(tender["quantity"])

    projected, projected_net, projected_gross = risk_after_tender(
        current_pos, ticker, action, qty
    )

    book = get_book(s, ticker)
    feat = book_features(tender, book)

    limit_ok = (
        abs(projected_net) <= NET_LIMIT
        and projected_gross <= GROSS_LIMIT
    )

    edge_ok = (
        feat["decision_edge_per_share"] is not None
        and feat["decision_edge_per_share"] >= MIN_NET_EDGE_PER_SHARE
        and feat["scored_profit"] is not None
        and feat["scored_profit"] >= MIN_EXPECTED_PROFIT
    )

    accept = bool(feat["fully_covered"] and limit_ok and edge_ok)

    reasons = []
    if not feat["fully_covered"]:
        reasons.append(
            f"visible book covers only {feat['visible_qty']:,}/{qty:,}"
        )
    if not limit_ok:
        reasons.append(
            f"projected limits net={projected_net:,}, gross={projected_gross:,}"
        )
    if feat["decision_edge_per_share"] is None:
        reasons.append("no executable book liquidity")
    elif not edge_ok:
        reasons.append(
            f"decision edge ${feat['decision_edge_per_share']:.4f}/sh, "
            f"scored ${feat['scored_profit']:,.0f}"
        )

    row = {
        "run_id": RUN_ID,
        "heat_id": heat_id,
        "observed_at_utc": observed_at,
        "tick": tick,
        "expires": tender.get("expires"),
        "period": period,
        "tender_id": int(tender["tender_id"]),
        **feat,

        "actual_positions": current_pos,
        "projected_positions": projected,
        "projected_net": projected_net,
        "projected_gross": projected_gross,

        "decision": "ACCEPT" if accept else "DECLINE",
        "reason": "; ".join(reasons) if reasons else "passes filters",
        "live_mode": LIVE,
    }

    return row, book


def followup_row(tender_row, book, tick):
    feat = book_features(
        {
            "ticker": tender_row["ticker"],
            "action": tender_row["action"],
            "quantity": tender_row["quantity"],
            "price": tender_row["tender_price"],
        },
        book,
    )

    return {
        "run_id": RUN_ID,
        "heat_id": tender_row["heat_id"],
        "tender_id": tender_row["tender_id"],
        "observed_at_utc": utc_now(),
        "arrival_tick": tender_row["tick"],
        "tick": tick,
        "elapsed_ticks": max(0, tick - tender_row["tick"]),
        "ticker": tender_row["ticker"],
        "action": tender_row["action"],
        "quantity": tender_row["quantity"],
        "tender_price": tender_row["tender_price"],

        "top_bid": feat["top_bid"],
        "top_ask": feat["top_ask"],
        "midpoint": feat["midpoint"],
        "spread": feat["spread"],

        "unwind_vwap": feat["unwind_vwap"],
        "worst_price": feat["worst_price"],
        "visible_qty": feat["visible_qty"],
        "fully_covered": feat["fully_covered"],

        "raw_edge_per_share": feat["raw_edge_per_share"],
        "fee_adjusted_edge_per_share": feat["fee_adjusted_edge_per_share"],
        "counterfactual_profit_after_fees": (
            feat["immediate_profit_after_fees"]
        ),

        "vwap_25": feat["vwap_25"],
        "vwap_50": feat["vwap_50"],
        "vwap_75": feat["vwap_75"],
        "vwap_100": feat["vwap_100"],

        "imbalance_top5": feat["imbalance_top5"],
        "imbalance_top10": feat["imbalance_top10"],
        "imbalance_full": feat["imbalance_full"],

        "breakeven_visible_qty": feat["breakeven_visible_qty"],
        "buffered_visible_qty": feat["buffered_visible_qty"],
        "threshold_visible_qty": feat["threshold_visible_qty"],
    }


# ============================================================================
# PRINTING
# ============================================================================

def fmt_money(x):
    return "N/A" if x is None else f"${x:,.0f}"


def fmt_price(x, digits=4):
    return "N/A" if x is None else f"{x:.{digits}f}"


def print_analysis(a):
    print("\n" + "=" * 86)
    print(
        f"HEAT {a['heat_id']} | TICK {a['tick']:>3} | Tender {a['tender_id']} | "
        f"{a['action']} {a['quantity']:,} {a['ticker']} @ {a['tender_price']:.2f}"
    )
    print(f"Expires: {a['expires']}")
    print(
        f"Book: bid={fmt_price(a['top_bid'], 2)} "
        f"ask={fmt_price(a['top_ask'], 2)} "
        f"spread={fmt_price(a['spread'], 4)}"
    )
    print(
        f"Immediate unwind: {a['unwind_side']} {a['quantity']:,} | "
        f"VWAP={fmt_price(a['unwind_vwap'], 6)} | "
        f"Worst={fmt_price(a['worst_price'], 2)}"
    )

    if a["raw_edge_per_share"] is not None:
        print(
            f"Raw edge={a['raw_edge_per_share']:.4f}/sh | "
            f"After fees={a['fee_adjusted_edge_per_share']:.4f}/sh "
            f"({fmt_money(a['immediate_profit_after_fees'])})"
        )
        print(
            f"Decision edge after safety buffer="
            f"{a['decision_edge_per_share']:.4f}/sh | "
            f"Scored={fmt_money(a['scored_profit'])}"
        )

    print(
        f"VWAP curve 25/50/75/100%: "
        f"{fmt_price(a['vwap_25'], 4)} / "
        f"{fmt_price(a['vwap_50'], 4)} / "
        f"{fmt_price(a['vwap_75'], 4)} / "
        f"{fmt_price(a['vwap_100'], 4)}"
    )
    print(
        f"Imbalance top5/top10/full: "
        f"{fmt_price(a['imbalance_top5'], 3)} / "
        f"{fmt_price(a['imbalance_top10'], 3)} / "
        f"{fmt_price(a['imbalance_full'], 3)}"
    )
    print(
        f"Projected risk: net {a['projected_net']:,}/{NET_LIMIT:,} | "
        f"gross {a['projected_gross']:,}/{GROSS_LIMIT:,}"
    )
    print(f">>> {a['decision']}: {a['reason']}")
    print("=" * 86)


# ============================================================================
# LIVE EXECUTION FUNCTIONS
# ============================================================================

def accept_tender(s, tender):
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

    for _ in range(40):
        pos = get_positions(s)[ticker]
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
    p = get_positions(s)
    for ticker, pos in p.items():
        if pos:
            print(f"HARD FLATTEN: {ticker} position {pos:,}")
            flatten_ticker(s, ticker)


# ============================================================================
# HEAT / FOLLOW-UP TRACKING
# ============================================================================

def new_heat(conn, heat_seq, tick, period):
    heat = {
        "heat_seq": heat_seq,
        "heat_id": f"{RUN_ID}-H{heat_seq:04d}",
        "started_at_utc": utc_now(),
        "start_tick": tick,
        "start_period": period,
        "tender_count": 0,
        "accept_count": 0,
        "decline_count": 0,
        "scored_profit_sum": 0.0,
    }
    insert_heat(conn, heat)
    print(
        f"\n### NEW HEAT {heat['heat_id']} | "
        f"tick={tick} period={period} ###"
    )
    return heat


def register_tracker(trackers, tender_row):
    tid = tender_row["tender_id"]
    trackers[tid] = {
        "tender_row": tender_row,
        "last_sampled_elapsed": 0,
        "finished": False,
    }


def sample_trackers(s, conn, trackers, current_tick, current_heat_id):
    finished = []

    for tid, tr in list(trackers.items()):
        tender_row = tr["tender_row"]

        if tender_row["heat_id"] != current_heat_id:
            finished.append(tid)
            continue

        elapsed = current_tick - tender_row["tick"]

        if elapsed < 0:
            finished.append(tid)
            continue

        if elapsed > FOLLOWUP_TICKS:
            finished.append(tid)
            continue

        if elapsed == 0:
            # Arrival sample is already logged separately.
            continue

        if elapsed % FOLLOWUP_EVERY_TICKS != 0:
            continue

        if elapsed <= tr["last_sampled_elapsed"]:
            continue

        book = get_book(s, tender_row["ticker"])
        row = followup_row(tender_row, book, current_tick)
        insert_followup(conn, row)
        insert_book_levels(
            conn,
            tender_row["heat_id"],
            tender_row["tender_id"],
            row["observed_at_utc"],
            current_tick,
            elapsed,
            tender_row["ticker"],
            book,
        )

        tr["last_sampled_elapsed"] = elapsed

    for tid in finished:
        trackers.pop(tid, None)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not API_KEY or API_KEY == "PASTE_YOUR_RIT_API_KEY_HERE":
        raise SystemExit(
            "Set API_KEY in this file or set PowerShell env var RIT_API_KEY first."
        )

    conn = open_db()

    print("Connected to LT3 research engine.")
    print(f"RUN_ID={RUN_ID}")
    print(f"LIVE={LIVE}  BASE={BASE}")
    print(f"Database: {DB_FILE}")
    print(f"Tender CSV: {TENDER_CSV}")
    print(
        f"Follow-up logging: every {FOLLOWUP_EVERY_TICKS} tick(s) "
        f"for {FOLLOWUP_TICKS} ticks after each tender; "
        f"up to {BOOK_LEVELS_TO_STORE} levels/side."
    )
    print(
        f"Decision rule: decision edge >= ${MIN_NET_EDGE_PER_SHARE:.2f}/sh "
        f"and scored profit >= ${MIN_EXPECTED_PROFIT:,.0f}; "
        f"fees=${MARKET_FEE_PER_SHARE + TENDER_FEE_PER_SHARE:.2f}/sh, "
        f"safety=${SAFETY_BUFFER_PER_SHARE:.2f}/sh."
    )

    seen = set()
    trackers = {}

    heat = None
    heat_seq = 0
    last_tick = None
    last_status = None
    last_period = None

    try:
        with rit_session() as s:
            while not shutdown:
                case = get_case(s)
                tick = int(case.get("tick", 0))
                status = str(case.get("status", ""))
                period = int(case.get("period", 0) or 0)

                # Detect a new heat either after inactive->active or a tick reset.
                tick_reset = (
                    last_tick is not None
                    and status == "ACTIVE"
                    and last_status == "ACTIVE"
                    and tick < last_tick
                )

                became_active = (
                    status == "ACTIVE"
                    and last_status != "ACTIVE"
                )

                if status == "ACTIVE" and (
                    heat is None or tick_reset or became_active
                ):
                    if heat is not None:
                        finalize_heat(
                            conn,
                            heat,
                            last_tick if last_tick is not None else tick,
                            last_period if last_period is not None else period,
                            "tick_reset" if tick_reset else "restart",
                        )

                    heat_seq += 1
                    heat = new_heat(conn, heat_seq, tick, period)
                    seen.clear()
                    trackers.clear()

                if status != "ACTIVE":
                    if heat is not None and last_status == "ACTIVE":
                        finalize_heat(
                            conn,
                            heat,
                            last_tick if last_tick is not None else tick,
                            last_period if last_period is not None else period,
                            "inactive",
                        )
                        print(
                            f"### HEAT COMPLETE {heat['heat_id']} | "
                            f"tenders={heat['tender_count']} "
                            f"paper_accepts={heat['accept_count']} "
                            f"paper_declines={heat['decline_count']} "
                            f"scored=${heat['scored_profit_sum']:,.0f} ###"
                        )
                        heat = None
                        trackers.clear()
                        seen.clear()

                    last_tick = tick
                    last_status = status
                    last_period = period
                    time.sleep(0.25)
                    continue

                # We now know there is an active heat.
                if heat is None:
                    heat_seq += 1
                    heat = new_heat(conn, heat_seq, tick, period)

                # Sample post-tender market path before processing new tenders.
                sample_trackers(
                    s, conn, trackers, tick, heat["heat_id"]
                )

                if tick >= HARD_FLATTEN_TICK:
                    if LIVE:
                        flatten_all(s)
                    time.sleep(0.20)
                    last_tick = tick
                    last_status = status
                    last_period = period
                    continue

                for tender in get_tenders(s):
                    tid = int(tender["tender_id"])
                    key = (heat["heat_id"], tid)

                    if key in seen:
                        continue

                    seen.add(key)

                    tender_row, arrival_book = analyze_tender(
                        s, tender, heat["heat_id"], tick, period
                    )

                    print_analysis(tender_row)

                    insert_tender(conn, tender_row)
                    append_tender_csv(tender_row)

                    # Arrival follow-up sample (elapsed 0).
                    arrival_followup = followup_row(
                        tender_row, arrival_book, tick
                    )
                    insert_followup(conn, arrival_followup)
                    insert_book_levels(
                        conn,
                        heat["heat_id"],
                        tid,
                        arrival_followup["observed_at_utc"],
                        tick,
                        0,
                        tender_row["ticker"],
                        arrival_book,
                    )

                    register_tracker(trackers, tender_row)

                    heat["tender_count"] += 1
                    if tender_row["decision"] == "ACCEPT":
                        heat["accept_count"] += 1
                    else:
                        heat["decline_count"] += 1

                    if tender_row["scored_profit"] is not None:
                        heat["scored_profit_sum"] += max(
                            0.0, tender_row["scored_profit"]
                        ) if tender_row["decision"] == "ACCEPT" else 0.0

                    if not LIVE:
                        print(
                            "[PAPER MODE] Tender/order action NOT sent. "
                            "Tracking its book for the next 30 ticks."
                        )
                        continue

                    if tender_row["decision"] == "ACCEPT":
                        # Legal sequencing: ACCEPT FIRST, then trade.
                        result = accept_tender(s, tender)
                        if not result.get("success", False):
                            print(f"Tender acceptance failed: {result}")
                            continue

                        print(
                            "Tender accepted. Now flattening resulting inventory..."
                        )
                        flatten_ticker(s, tender_row["ticker"])
                    else:
                        result = decline_tender(s, tid)
                        print(f"Tender declined: {result}")

                last_tick = tick
                last_status = status
                last_period = period
                time.sleep(POLL_SECONDS)

    finally:
        if heat is not None:
            finalize_heat(
                conn,
                heat,
                last_tick if last_tick is not None else 0,
                last_period if last_period is not None else 0,
                "script_stop",
            )
        conn.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped cleanly.")

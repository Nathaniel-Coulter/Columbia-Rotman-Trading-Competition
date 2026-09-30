"""
LT3 tender-pricing / research / live-execution calibration engine
Rotman Interactive Trader
Version: 4.0

PURPOSE
-------
V4 preserves the full V2/V3 research history but moves execution onto a
latency-prioritized hot path. Tender selection remains conservative; the main
experiment is faster, safer post-acceptance hedging with deferred persistence
and full execution telemetry. It is designed to answer:

1) How much latency can we remove after tender acceptance?
2) What execution VWAP do we actually receive versus the untouched arrival book?
3) What order-submit rate does the active case advertise, and do we hit limits?
4) Does a pipelined max-size hedge remain reliably flat and fully filled?
5) Does security-level realized P&L confirm the $0.02 market-hedge-only fee model?
6) Do CRZY/TAME and BUY/SELL tenders exhibit different execution error?
7) Which counterfactual cent/buffer thresholds would have accepted each tender?
8) Does deep-book imbalance predict short-horizon liquidity stability?

SAFETY / CASE RULES
-------------------
- LIVE defaults to False.
- Even when LIVE=True, the conservative V2 decision rule is unchanged.
- NO hedge order is submitted before the tender acceptance response succeeds.
- The live execution experiment uses only MARKET orders to flatten tender-created
  inventory, sliced at the LT3 maximum order sizes.
- No directional speculation or market making is performed.
- HARD_FLATTEN_TICK remains enabled as a safety backstop.

LOGS
----
All logs are stored beside this script in ./logs:

    lt3_training.sqlite       main research database (V2 + V3 + V4)
    lt3_tenders_v4.csv        compact V4 tender master table
    lt3_executions_v4.csv     one row per accepted V4 live tender
    lt3_orders_v4.csv         one row per V4 hedge/repair slice

The SQLite database is the authoritative dataset. CSVs are conveniences.

The schema is migration-safe: V4 can point at the existing V2/V3 database;
it adds version-specific execution tables while preserving historical data.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# ============================================================================
# CONFIG
# ============================================================================

VERSION = "4.0"
PORT = 9999
API_KEY = os.getenv("RIT_API_KEY", "PASTE_YOUR_RIT_API_KEY_HERE")
BASE = f"http://localhost:{PORT}/v1"

# ------------------------- LIVE SWITCH --------------------------------------
# Safe default. Use the explicit command-line switch:
#     python lt3_edge_engine_v4.py --live-demo
#
# Paper mode remains the default when no flag is supplied.
LIVE = False

# ------------------------- LOOP / CASE --------------------------------------
POLL_SECONDS = 0.05
NET_LIMIT = 100_000
GROSS_LIMIT = 250_000

MAX_ORDER_SIZE = {
    "CRZY": 25_000,
    "TAME": 10_000,
}

HARD_FLATTEN_TICK = 285

# ------------------------- CURRENT LIVE POLICY ------------------------------
# Keep the V2 conservative policy unchanged for the first execution calibration.
MARKET_FEE_PER_SHARE = 0.02
TENDER_FEE_PER_SHARE = 0.00
SAFETY_BUFFER_PER_SHARE = 0.03
MIN_NET_EDGE_PER_SHARE = 0.03
MIN_EXPECTED_PROFIT = 750.0

# ------------------------- COUNTERFACTUAL RESEARCH --------------------------
# These DO NOT change live decisions. V4 logs whether each tender would pass
# alternative after-fee edge thresholds so later analysis can compare policies.
AFTER_FEE_EDGE_THRESHOLDS = [
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.10, 0.12, 0.15,
]

# Alternative safety buffers to score later. Again: observational only.
SAFETY_BUFFER_GRID = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

# ------------------------- RESEARCH LOGGING ---------------------------------
FOLLOWUP_TICKS = 30
FOLLOWUP_EVERY_TICKS = 1
BOOK_LEVELS_TO_STORE = 60

# ------------------------- LIVE EXECUTION CALIBRATION -----------------------
EXECUTION_MODE = "PIPELINED_MARKET_V4"

# V4 does NOT wait for the portfolio screen/API to display the accepted tender
# before hedging. A successful POST /tenders/{id} response is the authorization
# event. This preserves the case rule: no hedge order is sent before acceptance.

# Order rate handling:
# - Prefer the security's advertised api_orders_per_second field when present.
# - Operate at a small safety fraction below that rate.
# - Fall back conservatively if the field is absent.
ORDER_RATE_FALLBACK_PER_SECOND = 9.0
ORDER_RATE_SAFETY_FRACTION = 0.90
ABSOLUTE_MIN_SUBMIT_INTERVAL_SECONDS = 0.001

# After the whole hedge burst is submitted, V4 verifies fills/position. These
# waits happen AFTER the exposure-reducing orders are already on the server.
POSITION_SETTLE_TIMEOUT_SECONDS = 2.50
STALE_POSITION_GRACE_SECONDS = 1.00
VERIFY_POLL_SECONDS = 0.02

# A repair burst is only sent when there is objective evidence of residual
# inventory/underfill. If all submitted orders report fully filled but the
# portfolio position display is stale, V4 waits/logs rather than over-hedging.
MAX_REPAIR_ROUNDS = 3

# Maximum number of individual orders in any hedge/repair sequence.
MAX_HEDGE_SLICES = 50

# Do not add passive execution in V4. Deep-book imbalance remains research-only
# until the fast-immediate baseline is calibrated.
USE_IMBALANCE_FOR_EXECUTION = False

# The fee model is now empirically supported by the V3 calibration:
# no tender commission, $0.02/share on the market hedge.
FEE_MODEL_LABEL = "MARKET_HEDGE_ONLY_2C"

# Global runtime telemetry.
RATE_LIMIT_HITS = 0
execution_in_progress = False

# ------------------------- PATHS --------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = LOG_DIR / "lt3_training.sqlite"
TENDER_CSV = LOG_DIR / "lt3_tenders_v4.csv"
EXECUTION_CSV = LOG_DIR / "lt3_executions_v4.csv"
ORDER_CSV = LOG_DIR / "lt3_orders_v4.csv"

RUN_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + uuid.uuid4().hex[:8]
)

shutdown = False


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

class RITError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def signal_handler(signum, frame):
    global shutdown
    # First Ctrl+C requests a graceful stop. If an execution burst is active,
    # V4 finishes its post-accept hedge/verification path before main exits.
    # A second Ctrl+C still falls back to the default handler.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True
    if execution_in_progress:
        print("\n[STOP REQUESTED] Finishing active hedge before shutdown...")


def rit_session():
    s = requests.Session()
    # The Client REST endpoint is localhost. Ignoring environment proxy settings
    # removes an unnecessary source of local request weirdness/latency.
    s.trust_env = False
    s.headers.update({"X-API-Key": API_KEY})
    return s


def request(s, method, path, allow_404=False, **kwargs):
    global RATE_LIMIT_HITS
    """
    HTTP wrapper:
    - automatically respects common RIT 429 wait responses;
    - retries transient requests exceptions;
    - provides a clear 401 error;
    - optionally returns None for 404 (used by order-detail fallback logic).
    """
    url = BASE + path

    while (not shutdown) or execution_in_progress:
        try:
            r = s.request(method, url, timeout=2, **kwargs)
        except requests.RequestException as exc:
            # GETs are idempotent and safe to retry. Blindly retrying a POST
            # after a lost response can duplicate an accepted tender or order,
            # so unsafe methods are escalated to reconciliation logic instead.
            if str(method).upper() == "GET":
                time.sleep(0.05)
                continue

            raise RITError(
                f"{method} {path} transport error; unsafe request was NOT "
                f"auto-retried: {exc}"
            ) from exc

        if r.status_code == 429:
            RATE_LIMIT_HITS += 1
            wait = 0.10
            try:
                body = r.json()
                wait = float(body.get("wait", wait))
            except Exception:
                try:
                    wait = float(r.headers.get("Retry-After", wait))
                except Exception:
                    pass
            time.sleep(max(wait, 0.02))
            continue

        if r.status_code == 401:
            raise RITError(
                "401 Unauthorized. Click the Client REST API icon in RIT and "
                "copy the exact API key into API_KEY (or RIT_API_KEY)."
            )

        if allow_404 and r.status_code == 404:
            return None

        r.raise_for_status()
        return r

    raise KeyboardInterrupt


# ============================================================================
# RIT GETTERS
# ============================================================================

def get_case(s):
    return request(s, "GET", "/case").json()


def get_trader(s):
    r = request(s, "GET", "/trader", allow_404=True)
    if r is None:
        return {}
    try:
        data = r.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_tenders(s):
    return request(s, "GET", "/tenders").json()


def get_securities(s):
    return request(s, "GET", "/securities").json()


def get_book(s, ticker):
    return request(
        s, "GET", "/securities/book", params={"ticker": ticker}
    ).json()


def get_orders(s):
    """
    RIT versions differ slightly in order-query behavior. The collection GET is
    useful as a fallback if /orders/{id} is unavailable.
    """
    r = request(s, "GET", "/orders", allow_404=True)
    if r is None:
        return []
    try:
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_order_detail(s, order_id):
    """
    Try GET /orders/{id}; if unsupported, fall back to GET /orders and search.
    Returns a dict or {}.
    """
    if order_id is None:
        return {}

    r = request(s, "GET", f"/orders/{order_id}", allow_404=True)
    if r is not None:
        try:
            data = r.json()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    for order in get_orders(s):
        if safe_int(order.get("order_id")) == safe_int(order_id):
            return order

    return {}


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


def security_for_ticker(securities, ticker):
    for sec in securities:
        if sec.get("ticker") == ticker:
            return sec
    return {}


def security_realized(securities, ticker):
    sec = security_for_ticker(securities, ticker)
    return first_numeric(
        sec,
        ["realized", "realized_pl", "realized_pnl", "realized_profit"],
    )


def security_unrealized(securities, ticker):
    sec = security_for_ticker(securities, ticker)
    return first_numeric(
        sec,
        ["unrealized", "unrealized_pl", "unrealized_pnl"],
    )


def advertised_order_rate(securities, ticker):
    """
    RIT case files can advertise a security-level API order rate. Names have
    varied across builds, so probe several plausible spellings and nested maps.
    Returns None when the case does not expose one.
    """
    sec = security_for_ticker(securities, ticker)
    keys = (
        "api_orders_per_second",
        "api_order_per_second",
        "orders_per_second",
        "max_orders_per_second",
        "api_orders_sec",
    )

    for key in keys:
        value = safe_float(sec.get(key))
        if value is not None and value > 0:
            return value

    for container_name in ("limits", "api", "rate_limits"):
        container = sec.get(container_name)
        if isinstance(container, dict):
            for key in keys:
                value = safe_float(container.get(key))
                if value is not None and value > 0:
                    return value

    return None


def effective_order_rate(advertised):
    raw = advertised if advertised and advertised > 0 else ORDER_RATE_FALLBACK_PER_SECOND
    return max(0.1, raw * ORDER_RATE_SAFETY_FRACTION)


def min_order_interval_seconds(advertised):
    return max(
        ABSOLUTE_MIN_SUBMIT_INTERVAL_SECONDS,
        1.0 / effective_order_rate(advertised),
    )


# ============================================================================
# PORTFOLIO / P&L SNAPSHOTS
# ============================================================================

def first_numeric(d: dict, names: Iterable[str]):
    for name in names:
        if name in d:
            v = safe_float(d.get(name))
            if v is not None:
                return v
    return None


def trader_metrics(trader):
    """
    Extract common account-level fields from GET /trader while preserving raw
    JSON. Field names can vary by RIT build, so missing values remain None.
    """
    if not isinstance(trader, dict):
        trader = {}

    pnl = first_numeric(
        trader,
        [
            "pnl", "P&L", "pl", "profit", "profit_loss",
            "total_pnl", "total_pl",
        ],
    )
    nlv = first_numeric(
        trader,
        ["nlv", "NLV", "net_liquidation_value"],
    )
    fines = first_numeric(
        trader,
        ["fines", "fine", "penalties", "penalty"],
    )

    return {
        "pnl": pnl,
        "nlv": nlv,
        "fines": fines,
        "raw": trader,
    }


def portfolio_metrics_from_securities(securities):
    """
    Extract security-level positions and P&L fields. Raw JSON is always kept,
    so a field-name mismatch never destroys information.
    """
    positions = positions_from_securities(securities)

    realized_values = []
    unrealized_values = []
    nlv_values = []
    by_ticker = {}

    for sec in securities:
        ticker = sec.get("ticker")
        if ticker not in MAX_ORDER_SIZE:
            continue

        realized = first_numeric(
            sec, ["realized", "realized_pl", "realized_pnl", "realized_profit"]
        )
        unrealized = first_numeric(
            sec, ["unrealized", "unrealized_pl", "unrealized_pnl"]
        )
        nlv = first_numeric(
            sec, ["nlv", "NLV", "net_liquidation_value"]
        )

        if realized is not None:
            realized_values.append(realized)
        if unrealized is not None:
            unrealized_values.append(unrealized)
        if nlv is not None:
            nlv_values.append(nlv)

        by_ticker[ticker] = {
            "position": positions.get(ticker, 0),
            "realized": realized,
            "unrealized": unrealized,
            "nlv": nlv,
            "raw": sec,
        }

    return {
        "positions": positions,
        "net_position": sum(positions.values()),
        "gross_position": sum(abs(x) for x in positions.values()),
        "realized_total": (
            sum(realized_values) if realized_values else None
        ),
        "unrealized_total": (
            sum(unrealized_values) if unrealized_values else None
        ),
        "nlv_total": sum(nlv_values) if nlv_values else None,
        "by_ticker": by_ticker,
        "raw": securities,
    }


def portfolio_snapshot(s):
    sec_metrics = portfolio_metrics_from_securities(get_securities(s))
    tr_metrics = trader_metrics(get_trader(s))

    sec_metrics["trader_pnl"] = tr_metrics["pnl"]
    sec_metrics["trader_nlv"] = tr_metrics["nlv"]
    sec_metrics["trader_fines"] = tr_metrics["fines"]
    sec_metrics["trader_raw"] = tr_metrics["raw"]
    return sec_metrics


def best_available_pnl(snapshot):
    """
    Prefer the account-level /trader P&L when exposed. Otherwise fall back to
    security-level NLV, then realized+unrealized, then realized.
    """
    if snapshot.get("trader_pnl") is not None:
        return snapshot["trader_pnl"]

    if snapshot.get("trader_nlv") is not None:
        return snapshot["trader_nlv"]

    if snapshot.get("nlv_total") is not None:
        return snapshot["nlv_total"]

    r = snapshot.get("realized_total")
    u = snapshot.get("unrealized_total")

    if r is not None and u is not None:
        return r + u
    if r is not None:
        return r
    return None


# ============================================================================
# DATABASE / MIGRATIONS
# ============================================================================

def table_columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_column(conn, table, column, declaration):
    if column not in table_columns(conn, table):
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
        )


def open_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # V2 base schema.
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
        """
    )

    # V3 columns added to V2 tables.
    tender_new_columns = {
        "ticker_action": "TEXT",
        "favorable_imbalance_top5": "REAL",
        "favorable_imbalance_top10": "REAL",
        "favorable_imbalance_full": "REAL",
        "unwind_depth_ratio": "REAL",
        "execution_id": "TEXT",
        "actual_accept_success": "INTEGER",
    }
    for col, decl in tender_new_columns.items():
        ensure_column(conn, "tenders", col, decl)

    followup_new_columns = {
        "favorable_imbalance_top5": "REAL",
        "favorable_imbalance_top10": "REAL",
        "favorable_imbalance_full": "REAL",
        "unwind_depth_ratio": "REAL",
        "live_intervention": "INTEGER DEFAULT 0",
    }
    for col, decl in followup_new_columns.items():
        ensure_column(conn, "followups", col, decl)

    # V3 research and execution tables.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            live_mode INTEGER NOT NULL,
            execution_mode TEXT,
            config_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS policy_grid (
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            after_fee_edge_threshold REAL NOT NULL,
            safety_buffer REAL NOT NULL,
            min_expected_profit REAL NOT NULL,
            would_accept INTEGER NOT NULL,
            modeled_profit_after_fee REAL,
            modeled_profit_after_buffer REAL,
            fully_covered INTEGER NOT NULL,
            limit_ok INTEGER NOT NULL,
            PRIMARY KEY (
                run_id, heat_id, tender_id,
                after_fee_edge_threshold, safety_buffer
            )
        );

        CREATE TABLE IF NOT EXISTS pnl_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER,
            execution_id TEXT,
            observed_at_utc TEXT NOT NULL,
            tick INTEGER,
            phase TEXT NOT NULL,
            net_position INTEGER,
            gross_position INTEGER,
            realized_total REAL,
            unrealized_total REAL,
            nlv_total REAL,
            trader_pnl REAL,
            trader_nlv REAL,
            trader_fines REAL,
            best_pnl REAL,
            positions_json TEXT NOT NULL,
            securities_json TEXT NOT NULL,
            trader_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS execution_sessions (
            execution_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            tender_action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            tender_price REAL NOT NULL,

            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,

            predicted_arrival_unwind_vwap REAL,
            predicted_arrival_after_fee_edge REAL,
            predicted_arrival_profit_after_fee REAL,

            favorable_imbalance_full REAL,

            accept_request_at_utc TEXT,
            accept_response_at_utc TEXT,
            accept_latency_ms REAL,
            accept_success INTEGER,
            accept_response_json TEXT,

            position_before_accept INTEGER,
            position_after_accept INTEGER,
            tender_position_latency_ms REAL,

            hedge_action TEXT,
            hedge_requested_qty INTEGER DEFAULT 0,
            hedge_observed_filled_qty INTEGER DEFAULT 0,
            hedge_slice_count INTEGER DEFAULT 0,

            realized_hedge_vwap REAL,
            execution_error_per_share REAL,
            execution_error_dollars REAL,

            gross_block_profit REAL,
            profit_less_unwind_fee REAL,
            profit_less_two_2c_fees REAL,

            pnl_before REAL,
            pnl_after_accept REAL,
            pnl_after_flat REAL,
            rit_pnl_delta REAL,

            inferred_fee_error_one_side REAL,
            inferred_fee_error_two_sides REAL,

            final_position INTEGER,
            fully_flat INTEGER,
            failure_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS order_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            execution_id TEXT NOT NULL,

            slice_no INTEGER NOT NULL,
            observed_at_utc TEXT NOT NULL,

            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            requested_qty INTEGER NOT NULL,

            position_before INTEGER,
            position_after INTEGER,
            observed_filled_qty INTEGER,

            predicted_slice_vwap REAL,
            predicted_slice_worst REAL,
            predicted_slice_covered INTEGER,

            order_id INTEGER,
            submit_latency_ms REAL,
            fill_observation_latency_ms REAL,

            response_json TEXT,
            detail_json TEXT,

            actual_vwap REAL,
            actual_price REAL,

            signed_price_improvement REAL,
            adverse_slippage_per_share REAL,

            status TEXT,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tenders_heat
            ON tenders(heat_id, tick);

        CREATE INDEX IF NOT EXISTS idx_followups_tender
            ON followups(heat_id, tender_id, elapsed_ticks);

        CREATE INDEX IF NOT EXISTS idx_book_tender
            ON book_levels(heat_id, tender_id, elapsed_ticks, side, level_rank);

        CREATE INDEX IF NOT EXISTS idx_execution_tender
            ON execution_sessions(heat_id, tender_id);

        CREATE INDEX IF NOT EXISTS idx_orders_execution
            ON order_events(execution_id, slice_no);

        CREATE INDEX IF NOT EXISTS idx_pnl_execution
            ON pnl_snapshots(execution_id, phase);
        """
    )

    # If an early V3 database exists, make P&L snapshot migration resilient.
    # (No-op for a fresh database.)
    if "pnl_snapshots" in {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }:
        for col, decl in {
            "trader_pnl": "REAL",
            "trader_nlv": "REAL",
            "trader_fines": "REAL",
            "trader_json": "TEXT",
        }.items():
            ensure_column(conn, "pnl_snapshots", col, decl)


    # V4 execution tables live alongside the V2/V3 tables. Keeping them
    # version-specific prevents schema ambiguity while preserving the full
    # historical research database.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS execution_sessions_v4 (
            execution_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            tender_action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            tender_price REAL NOT NULL,

            detected_at_utc TEXT,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,

            execution_mode TEXT NOT NULL,
            fee_model TEXT NOT NULL,

            advertised_order_rate REAL,
            effective_order_rate REAL,
            min_submit_interval_ms REAL,

            predicted_arrival_unwind_vwap REAL,
            predicted_arrival_after_fee_edge REAL,
            predicted_arrival_profit_after_fee REAL,
            favorable_imbalance_full REAL,

            accept_success INTEGER,
            accept_latency_ms REAL,
            decision_to_accept_start_ms REAL,
            accept_to_first_order_ms REAL,
            accept_to_last_order_ms REAL,
            burst_submission_span_ms REAL,
            accept_to_target_position_ms REAL,
            total_execution_ms REAL,

            hedge_action TEXT,
            hedge_slice_count INTEGER,
            hedge_submitted_qty INTEGER,
            hedge_reported_filled_qty INTEGER,

            realized_hedge_vwap REAL,
            execution_error_per_share REAL,
            execution_error_dollars REAL,

            gross_block_profit REAL,
            profit_less_unwind_fee REAL,

            security_realized_before REAL,
            security_realized_after REAL,
            security_realized_delta REAL,
            fee_model_error REAL,

            target_position INTEGER,
            final_position INTEGER,
            fully_flat INTEGER,
            all_orders_filled INTEGER,

            repair_qty INTEGER DEFAULT 0,
            repair_order_count INTEGER DEFAULT 0,

            rate_limit_hits_during_execution INTEGER DEFAULT 0,
            trader_fines_after REAL,

            accept_response_json TEXT,
            pre_securities_json TEXT,
            post_securities_json TEXT,
            trader_after_json TEXT,
            failure_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS order_events_v4 (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            execution_id TEXT NOT NULL,

            slice_no INTEGER NOT NULL,
            order_role TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,

            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            requested_qty INTEGER NOT NULL,

            predicted_slice_vwap REAL,
            predicted_slice_worst REAL,
            predicted_slice_covered INTEGER,

            order_id INTEGER,
            submit_start_relative_ms REAL,
            submit_response_relative_ms REAL,
            submit_latency_ms REAL,
            pacing_sleep_ms REAL,

            quantity_filled INTEGER,
            actual_vwap REAL,
            actual_price REAL,
            signed_price_improvement REAL,
            adverse_slippage_per_share REAL,
            status TEXT,

            response_json TEXT,
            detail_json TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS latency_events_v4 (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            execution_id TEXT NOT NULL,
            event_seq INTEGER NOT NULL,
            event_name TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            relative_ms REAL NOT NULL,
            payload_json TEXT
        );

        CREATE TABLE IF NOT EXISTS position_observations_v4 (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            heat_id TEXT NOT NULL,
            tender_id INTEGER NOT NULL,
            execution_id TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            relative_ms REAL NOT NULL,
            ticker TEXT NOT NULL,
            position INTEGER NOT NULL,
            target_position INTEGER NOT NULL,
            phase TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_exec_v4_heat
            ON execution_sessions_v4(heat_id, tender_id);

        CREATE INDEX IF NOT EXISTS idx_orders_v4_exec
            ON order_events_v4(execution_id, slice_no);

        CREATE INDEX IF NOT EXISTS idx_latency_v4_exec
            ON latency_events_v4(execution_id, event_seq);

        CREATE INDEX IF NOT EXISTS idx_position_v4_exec
            ON position_observations_v4(execution_id, relative_ms);
        """
    )

    config = {
        "version": VERSION,
        "live": LIVE,
        "execution_mode": EXECUTION_MODE,
        "poll_seconds": POLL_SECONDS,
        "net_limit": NET_LIMIT,
        "gross_limit": GROSS_LIMIT,
        "max_order_size": MAX_ORDER_SIZE,
        "market_fee_per_share": MARKET_FEE_PER_SHARE,
        "tender_fee_per_share": TENDER_FEE_PER_SHARE,
        "safety_buffer_per_share": SAFETY_BUFFER_PER_SHARE,
        "min_net_edge_per_share": MIN_NET_EDGE_PER_SHARE,
        "min_expected_profit": MIN_EXPECTED_PROFIT,
        "after_fee_edge_thresholds": AFTER_FEE_EDGE_THRESHOLDS,
        "safety_buffer_grid": SAFETY_BUFFER_GRID,
        "followup_ticks": FOLLOWUP_TICKS,
        "followup_every_ticks": FOLLOWUP_EVERY_TICKS,
        "book_levels_to_store": BOOK_LEVELS_TO_STORE,
        "hard_flatten_tick": HARD_FLATTEN_TICK,
        "order_rate_fallback_per_second": ORDER_RATE_FALLBACK_PER_SECOND,
        "order_rate_safety_fraction": ORDER_RATE_SAFETY_FRACTION,
        "position_settle_timeout_seconds": POSITION_SETTLE_TIMEOUT_SECONDS,
        "stale_position_grace_seconds": STALE_POSITION_GRACE_SECONDS,
        "verify_poll_seconds": VERIFY_POLL_SECONDS,
        "max_repair_rounds": MAX_REPAIR_ROUNDS,
        "fee_model_label": FEE_MODEL_LABEL,
    }

    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            run_id, version, started_at_utc, live_mode,
            execution_mode, config_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID, VERSION, utc_now(), int(LIVE),
            EXECUTION_MODE, json_dumps(config),
        ),
    )

    conn.commit()
    return conn


# ============================================================================
# CSV HELPERS
# ============================================================================

TENDER_CSV_FIELDS = [
    "run_id", "heat_id", "observed_at_utc", "tick", "expires", "period",
    "tender_id", "ticker", "action", "ticker_action",
    "api_orders_per_second",
    "quantity", "tender_price",
    "top_bid", "top_ask", "midpoint", "spread",
    "unwind_side", "unwind_vwap", "worst_price", "visible_qty",
    "fully_covered", "unwind_depth_ratio",
    "raw_edge_per_share", "fee_adjusted_edge_per_share",
    "decision_edge_per_share",
    "immediate_profit_after_fees", "scored_profit",
    "vwap_25", "vwap_50", "vwap_75", "vwap_100",
    "imbalance_top5", "imbalance_top10", "imbalance_full",
    "favorable_imbalance_top5", "favorable_imbalance_top10",
    "favorable_imbalance_full",
    "breakeven_visible_qty", "buffered_visible_qty",
    "threshold_visible_qty",
    "projected_net", "projected_gross",
    "decision", "reason", "live_mode",
]

EXECUTION_CSV_FIELDS = [
    "execution_id", "run_id", "heat_id", "tender_id",
    "ticker", "tender_action", "quantity", "tender_price",
    "execution_mode", "fee_model",
    "advertised_order_rate", "effective_order_rate",
    "min_submit_interval_ms",
    "predicted_arrival_unwind_vwap",
    "predicted_arrival_after_fee_edge",
    "predicted_arrival_profit_after_fee",
    "favorable_imbalance_full",
    "accept_success", "accept_latency_ms",
    "decision_to_accept_start_ms",
    "accept_to_first_order_ms", "accept_to_last_order_ms",
    "burst_submission_span_ms",
    "accept_to_target_position_ms", "total_execution_ms",
    "hedge_action", "hedge_slice_count",
    "hedge_submitted_qty", "hedge_reported_filled_qty",
    "realized_hedge_vwap",
    "execution_error_per_share", "execution_error_dollars",
    "gross_block_profit", "profit_less_unwind_fee",
    "security_realized_before", "security_realized_after",
    "security_realized_delta", "fee_model_error",
    "target_position", "final_position", "fully_flat",
    "all_orders_filled", "repair_qty", "repair_order_count",
    "rate_limit_hits_during_execution",
    "trader_fines_after", "failure_reason",
]

ORDER_CSV_FIELDS = [
    "run_id", "heat_id", "tender_id", "execution_id",
    "slice_no", "order_role", "observed_at_utc",
    "ticker", "action", "requested_qty",
    "predicted_slice_vwap", "predicted_slice_worst",
    "predicted_slice_covered",
    "order_id", "submit_start_relative_ms",
    "submit_response_relative_ms", "submit_latency_ms",
    "pacing_sleep_ms",
    "quantity_filled", "actual_vwap", "actual_price",
    "signed_price_improvement", "adverse_slippage_per_share",
    "status", "response_json", "detail_json", "error",
]




def append_csv(path, fieldnames, row):
    """
    Append only when the existing header matches this version's schema.
    If a stale/mismatched file somehow exists, rotate it instead of corrupting
    the CSV with rows from a different schema.
    """
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                existing = next(csv.reader(f), [])
            if list(existing) != list(fieldnames):
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                rotated = path.with_name(
                    f"{path.stem}_schema_mismatch_{stamp}{path.suffix}"
                )
                path.replace(rotated)
        except Exception:
            pass

    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


# ============================================================================
# DATABASE INSERTS / UPDATES
# ============================================================================

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
            decision, reason, live_mode,
            ticker_action,
            favorable_imbalance_top5,
            favorable_imbalance_top10,
            favorable_imbalance_full,
            unwind_depth_ratio,
            execution_id,
            actual_accept_success
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
            ?,?,?,
            ?,?,?,?,?,
            ?,?
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
            json_dumps(row["actual_positions"]),
            json_dumps(row["projected_positions"]),
            row["projected_net"], row["projected_gross"],
            row["decision"], row["reason"], int(row["live_mode"]),
            row["ticker_action"],
            row["favorable_imbalance_top5"],
            row["favorable_imbalance_top10"],
            row["favorable_imbalance_full"],
            row["unwind_depth_ratio"],
            row.get("execution_id"),
            row.get("actual_accept_success"),
        ),
    )
    conn.commit()


def update_tender_execution_link(
    conn, row, execution_id=None, actual_accept_success=None
):
    conn.execute(
        """
        UPDATE tenders
        SET execution_id = ?, actual_accept_success = ?
        WHERE run_id = ? AND heat_id = ? AND tender_id = ?
        """,
        (
            execution_id,
            actual_accept_success,
            row["run_id"], row["heat_id"], row["tender_id"],
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
            threshold_visible_qty,
            favorable_imbalance_top5,
            favorable_imbalance_top10,
            favorable_imbalance_full,
            unwind_depth_ratio,
            live_intervention
        ) VALUES (
            ?,?,?,?,
            ?,?,?,
            ?,?,?,?,
            ?,?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,?,?,?
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
            row["favorable_imbalance_top5"],
            row["favorable_imbalance_top10"],
            row["favorable_imbalance_full"],
            row["unwind_depth_ratio"],
            int(row["live_intervention"]),
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

        for rank, level in enumerate(
            book.get(side_key, [])[:BOOK_LEVELS_TO_STORE], 1
        ):
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


def insert_policy_grid(conn, tender_row, limit_ok):
    rows = []
    fee_edge = tender_row["fee_adjusted_edge_per_share"]
    qty = tender_row["quantity"]
    full = bool(tender_row["fully_covered"])

    for threshold in AFTER_FEE_EDGE_THRESHOLDS:
        for buffer in SAFETY_BUFFER_GRID:
            after_buffer = None if fee_edge is None else fee_edge - buffer
            modeled_profit_buffered = (
                None if after_buffer is None else after_buffer * qty
            )

            # This grid's "threshold" is an after-fee threshold. Buffer is
            # independently varied so we can study both choices.
            would_accept = (
                full
                and limit_ok
                and after_buffer is not None
                and after_buffer >= threshold
                and modeled_profit_buffered >= MIN_EXPECTED_PROFIT
            )

            rows.append(
                (
                    RUN_ID, tender_row["heat_id"], tender_row["tender_id"],
                    tender_row["ticker"], tender_row["action"],
                    threshold, buffer, MIN_EXPECTED_PROFIT,
                    int(would_accept),
                    tender_row["immediate_profit_after_fees"],
                    modeled_profit_buffered,
                    int(full), int(limit_ok),
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO policy_grid (
            run_id, heat_id, tender_id, ticker, action,
            after_fee_edge_threshold, safety_buffer, min_expected_profit,
            would_accept, modeled_profit_after_fee,
            modeled_profit_after_buffer, fully_covered, limit_ok
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def insert_pnl_snapshot(
    conn, s, heat_id, tender_id, execution_id, tick, phase
):
    snap = portfolio_snapshot(s)

    conn.execute(
        """
        INSERT INTO pnl_snapshots (
            run_id, heat_id, tender_id, execution_id,
            observed_at_utc, tick, phase,
            net_position, gross_position,
            realized_total, unrealized_total, nlv_total,
            trader_pnl, trader_nlv, trader_fines,
            best_pnl,
            positions_json, securities_json, trader_json
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?,
            ?, ?, ?
        )
        """,
        (
            RUN_ID, heat_id, tender_id, execution_id,
            utc_now(), tick, phase,
            snap["net_position"], snap["gross_position"],
            snap["realized_total"], snap["unrealized_total"],
            snap["nlv_total"],
            snap.get("trader_pnl"),
            snap.get("trader_nlv"),
            snap.get("trader_fines"),
            best_available_pnl(snap),
            json_dumps(snap["positions"]),
            json_dumps(snap["raw"]),
            json_dumps(snap.get("trader_raw", {})),
        ),
    )
    conn.commit()
    return snap


# ============================================================================
# MARKET / BOOK FEATURES
# ============================================================================

def sweep_vwap(book_side, quantity):
    """
    Price an immediate sweep through one side of the visible book.

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


def favorable_imbalance(action, generic_imbalance):
    """
    Positive means depth is favorable to our required unwind:
      tender BUY  -> we must SELL -> bids are favorable -> generic + is good
      tender SELL -> we must BUY  -> asks are favorable -> generic - is good
    """
    if generic_imbalance is None:
        return None
    return generic_imbalance if action == "BUY" else -generic_imbalance


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
    for frac, name in (
        (0.25, "vwap_25"),
        (0.50, "vwap_50"),
        (0.75, "vwap_75"),
        (1.00, "vwap_100"),
    ):
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
        None if fee_edge is None
        else fee_edge - SAFETY_BUFFER_PER_SHARE
    )

    im5 = imbalance(book, 5)
    im10 = imbalance(book, 10)
    imfull = imbalance(book, None)

    total_unwind_depth = side_total(unwind_book, None)
    depth_ratio = total_unwind_depth / qty if qty > 0 else None

    breakeven_qty = profitable_visible_qty(
        unwind_book, action, tender_price, required_edge_per_share=fees
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
        "ticker_action": f"{ticker}_{action}",
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
        "unwind_depth_ratio": depth_ratio,

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

        "imbalance_top5": im5,
        "imbalance_top10": im10,
        "imbalance_full": imfull,

        "favorable_imbalance_top5": favorable_imbalance(action, im5),
        "favorable_imbalance_top10": favorable_imbalance(action, im10),
        "favorable_imbalance_full": favorable_imbalance(action, imfull),

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
    api_rate = advertised_order_rate(securities, ticker)
    realized_before = security_realized(securities, ticker)

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

        "execution_id": None,
        "actual_accept_success": None,
        "api_orders_per_second": api_rate,
        "security_realized_before": realized_before,
        "_pre_securities": securities,
        "_limit_ok": limit_ok,
    }

    return row, book


def followup_row(tender_row, book, tick, live_intervention=False):
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

        "favorable_imbalance_top5": feat["favorable_imbalance_top5"],
        "favorable_imbalance_top10": feat["favorable_imbalance_top10"],
        "favorable_imbalance_full": feat["favorable_imbalance_full"],
        "unwind_depth_ratio": feat["unwind_depth_ratio"],

        "breakeven_visible_qty": feat["breakeven_visible_qty"],
        "buffered_visible_qty": feat["buffered_visible_qty"],
        "threshold_visible_qty": feat["threshold_visible_qty"],

        "live_intervention": bool(live_intervention),
    }


# ============================================================================
# CONSOLE
# ============================================================================

def fmt_money(x):
    return "N/A" if x is None else f"${x:,.0f}"


def fmt_price(x, digits=4):
    return "N/A" if x is None else f"{x:.{digits}f}"


def print_analysis(a):
    print("\n" + "=" * 92)
    print(
        f"HEAT {a['heat_id']} | TICK {a['tick']:>3} | Tender {a['tender_id']} | "
        f"{a['action']} {a['quantity']:,} {a['ticker']} @ "
        f"{a['tender_price']:.2f}"
    )
    print(f"Expires: {a['expires']}")
    print(
        f"Book: bid={fmt_price(a['top_bid'], 2)} "
        f"ask={fmt_price(a['top_ask'], 2)} "
        f"spread={fmt_price(a['spread'], 4)} "
        f"depth/x={fmt_price(a['unwind_depth_ratio'], 2)}"
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
            f"Decision edge after safety="
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
        f"Fav imbalance top5/top10/full: "
        f"{fmt_price(a['favorable_imbalance_top5'], 3)} / "
        f"{fmt_price(a['favorable_imbalance_top10'], 3)} / "
        f"{fmt_price(a['favorable_imbalance_full'], 3)}"
    )
    print(
        f"Projected risk: net {a['projected_net']:,}/{NET_LIMIT:,} | "
        f"gross {a['projected_gross']:,}/{GROSS_LIMIT:,}"
    )
    print(f">>> {a['decision']}: {a['reason']}")
    print("=" * 92)


# ============================================================================
# TENDER / ORDER ACTIONS
# ============================================================================

def accept_tender_fast(s, tender):
    t0 = monotonic_ms()
    request_at = utc_now()

    r = request(
        s,
        "POST",
        f"/tenders/{int(tender['tender_id'])}",
        params={"price": float(tender["price"])},
    )

    latency = monotonic_ms() - t0
    response_at = utc_now()

    try:
        body = r.json()
    except Exception:
        body = {}

    return {
        "success": bool(body.get("success", r.ok)),
        "request_at": request_at,
        "response_at": response_at,
        "latency_ms": latency,
        "body": body,
    }


def decline_tender(s, tender_id):
    r = request(s, "DELETE", f"/tenders/{int(tender_id)}")
    try:
        return r.json()
    except Exception:
        return {"success": r.ok}


def resolve_ambiguous_tender_accept(
    s,
    tender,
    position_before,
    timeout_seconds=2.50,
):
    """
    Resolve the rare case where a POST /tenders response is lost.

    We never blindly re-POST the tender. Instead:
      1) if the tender is still active, treat the accept as not completed;
      2) if it disappeared, wait for the tender-created position to appear;
      3) only a confirmed expected position authorizes the hedge.

    Returns (resolved_success, observed_position, reason), where
    resolved_success is True / False / None (unresolved).
    """
    tid = int(tender["tender_id"])
    ticker = tender["ticker"]
    action = tender["action"].upper()
    qty = int(tender["quantity"])

    expected = (
        position_before + qty
        if action == "BUY"
        else position_before - qty
    )

    try:
        active = get_tenders(s)
        active_ids = {
            int(x["tender_id"])
            for x in active
        }

        if tid in active_ids:
            return False, position_before, "tender_still_active"
    except Exception:
        pass

    deadline = time.perf_counter() + timeout_seconds
    latest = position_before

    while (
        time.perf_counter() < deadline
        and ((not shutdown) or execution_in_progress)
    ):
        try:
            latest = get_positions(s)[ticker]
        except Exception:
            time.sleep(VERIFY_POLL_SECONDS)
            continue

        if latest == expected:
            return True, latest, "expected_position_confirmed"

        time.sleep(VERIFY_POLL_SECONDS)

    return None, latest, "acceptance_unresolved"


def split_quantity(quantity, max_size):
    quantity = int(quantity)
    max_size = int(max_size)

    if quantity <= 0 or max_size <= 0:
        return []

    out = []
    remaining = quantity

    while remaining > 0:
        q = min(remaining, max_size)
        out.append(q)
        remaining -= q

    return out


def plan_sliced_sweep(book_side, slice_sizes):
    """
    Consume ONE immutable arrival-book snapshot across sequential slice sizes.

    Unlike V3, V4 does not issue a new book GET before every hedge slice.
    That read/parse/DB path was part of the latency we are removing. The
    per-slice predictions below are therefore a clean "arrival book expected
    path" against which actual fills can be compared.
    """
    levels = [
        {
            "price": float(level["price"]),
            "remaining": available_quantity(level),
        }
        for level in book_side
        if available_quantity(level) > 0
    ]

    level_index = 0
    predictions = []

    for slice_no, requested in enumerate(slice_sizes, 1):
        remaining = int(requested)
        notional = 0.0
        filled = 0
        worst = None

        while remaining > 0 and level_index < len(levels):
            level = levels[level_index]

            if level["remaining"] <= 0:
                level_index += 1
                continue

            take = min(remaining, level["remaining"])
            notional += take * level["price"]
            filled += take
            remaining -= take
            level["remaining"] -= take
            worst = level["price"]

            if level["remaining"] <= 0:
                level_index += 1

        predictions.append(
            {
                "slice_no": slice_no,
                "requested_qty": int(requested),
                "predicted_slice_vwap": (
                    notional / filled if filled > 0 else None
                ),
                "predicted_slice_worst": worst,
                "predicted_slice_covered": remaining == 0,
                "predicted_filled_qty": filled,
            }
        )

    return predictions


def extract_order_actuals(response, detail):
    merged = {}

    if isinstance(response, dict):
        merged.update(response)
    if isinstance(detail, dict):
        merged.update(detail)

    order_id = safe_int(merged.get("order_id"))

    actual_vwap = first_numeric(
        merged,
        ["vwap", "VWAP", "fill_vwap", "avg_price", "average_price"],
    )

    actual_price = first_numeric(
        merged,
        ["price", "fill_price", "last_fill_price"],
    )

    quantity_filled = None
    for name in (
        "quantity_filled",
        "filled",
        "filled_qty",
        "filled_quantity",
        "quantity_transacted",
    ):
        q = safe_int(merged.get(name))
        if q is not None:
            quantity_filled = max(0, q)
            break

    status = (
        merged.get("status")
        or merged.get("order_status")
        or merged.get("state")
    )

    return (
        order_id,
        quantity_filled,
        actual_vwap,
        actual_price,
        status,
    )


def signed_execution_metrics(action, predicted_vwap, actual_vwap):
    """
    Positive price improvement = actual fill BETTER than the arrival-book
    prediction. Positive adverse slippage = actual fill WORSE.
    """
    if predicted_vwap is None or actual_vwap is None:
        return None, None

    if action == "SELL":
        improvement = actual_vwap - predicted_vwap
    else:
        improvement = predicted_vwap - actual_vwap

    return improvement, -improvement


def pace_submission(last_submit_start, min_interval_seconds):
    if last_submit_start is None:
        return 0.0

    now = time.perf_counter()
    wait = min_interval_seconds - (now - last_submit_start)

    if wait > 0:
        time.sleep(wait)
        return wait * 1000.0

    return 0.0


def send_market_order_fast(s, ticker, action, qty):
    t0 = monotonic_ms()

    response = request(
        s,
        "POST",
        "/orders",
        params={
            "ticker": ticker,
            "type": "MARKET",
            "quantity": int(qty),
            "action": action,
        },
    )

    latency = monotonic_ms() - t0

    try:
        body = response.json()
    except Exception:
        body = {}

    return body, latency


def details_by_order_id(s):
    """
    Prefer one GET /orders call for the entire burst rather than N individual
    order-detail calls. V4 falls back to GET /orders/{id} only when needed.
    """
    result = {}

    for order in get_orders(s):
        oid = safe_int(order.get("order_id"))
        if oid is not None:
            result[oid] = order

    return result


def update_order_rows_from_api(s, order_rows):
    collection = details_by_order_id(s)

    for row in order_rows:
        oid = safe_int(row.get("order_id"))
        detail = {}

        if oid is not None:
            detail = collection.get(oid, {})

            if not detail:
                detail = get_order_detail(s, oid)

        (
            oid2,
            quantity_filled,
            actual_vwap,
            actual_price,
            status,
        ) = extract_order_actuals(row.get("_response", {}), detail)

        if row.get("order_id") is None:
            row["order_id"] = oid2

        if quantity_filled is not None:
            row["quantity_filled"] = quantity_filled

        if actual_vwap is not None:
            row["actual_vwap"] = actual_vwap

        if actual_price is not None:
            row["actual_price"] = actual_price

        if status is not None:
            row["status"] = status

        row["detail_json"] = json_dumps(detail)

        improvement, adverse = signed_execution_metrics(
            row["action"],
            row.get("predicted_slice_vwap"),
            row.get("actual_vwap"),
        )
        row["signed_price_improvement"] = improvement
        row["adverse_slippage_per_share"] = adverse


def orders_fully_filled(order_rows):
    if not order_rows:
        return False

    for row in order_rows:
        if row.get("error"):
            return False

        filled = safe_int(row.get("quantity_filled"), 0)
        requested = safe_int(row.get("requested_qty"), 0)

        if filled < requested:
            return False

    return True


def weighted_realized_vwap(order_rows):
    notional = 0.0
    quantity = 0

    for row in order_rows:
        q = safe_int(row.get("quantity_filled"), 0) or 0
        vwap = safe_float(row.get("actual_vwap"))

        if q > 0 and vwap is not None:
            notional += q * vwap
            quantity += q

    if quantity <= 0:
        return None, 0

    return notional / quantity, quantity


def execution_profit_metrics(
    tender_action,
    tender_price,
    quantity,
    realized_hedge_vwap,
):
    if (
        realized_hedge_vwap is None
        or quantity <= 0
    ):
        return None, None

    if tender_action == "BUY":
        gross = (
            realized_hedge_vwap - tender_price
        ) * quantity
    else:
        gross = (
            tender_price - realized_hedge_vwap
        ) * quantity

    less_unwind_fee = (
        gross - MARKET_FEE_PER_SHARE * quantity
    )

    return gross, less_unwind_fee


def timeline_mark(timeline, origin_mono, name, payload=None):
    timeline.append(
        {
            "event_seq": len(timeline) + 1,
            "event_name": name,
            "observed_at_utc": utc_now(),
            "relative_ms": (
                time.perf_counter() - origin_mono
            ) * 1000.0,
            "payload_json": json_dumps(payload or {}),
        }
    )


def observe_position(
    s,
    ticker,
    target_position,
    origin_mono,
    phase,
    observations,
):
    securities = get_securities(s)
    position = positions_from_securities(securities)[ticker]

    observations.append(
        {
            "observed_at_utc": utc_now(),
            "relative_ms": (
                time.perf_counter() - origin_mono
            ) * 1000.0,
            "ticker": ticker,
            "position": position,
            "target_position": target_position,
            "phase": phase,
        }
    )

    return securities, position


def wait_for_burst_settlement(
    s,
    ticker,
    target_position,
    order_rows,
    origin_mono,
    observations,
):
    """
    Verification occurs only AFTER the complete initial hedge burst has been
    submitted. Position polling can therefore no longer delay the hedge itself.
    """
    deadline = (
        time.perf_counter()
        + POSITION_SETTLE_TIMEOUT_SECONDS
    )

    latest_securities = []
    latest_position = None
    all_filled = False

    while (
        time.perf_counter() < deadline
        and ((not shutdown) or execution_in_progress)
    ):
        latest_securities, latest_position = observe_position(
            s,
            ticker,
            target_position,
            origin_mono,
            "verify_initial_burst",
            observations,
        )

        update_order_rows_from_api(s, order_rows)
        all_filled = orders_fully_filled(order_rows)

        if (
            latest_position == target_position
            and all_filled
        ):
            return (
                latest_securities,
                latest_position,
                all_filled,
                False,
            )

        time.sleep(VERIFY_POLL_SECONDS)

    # If every order record says fully filled but the portfolio display is
    # still stale, give it extra time. DO NOT send repair orders merely because
    # a lagging position field disagrees with filled order records.
    stale_position = (
        all_filled
        and latest_position != target_position
    )

    if stale_position:
        grace_deadline = (
            time.perf_counter()
            + STALE_POSITION_GRACE_SECONDS
        )

        while (
            time.perf_counter() < grace_deadline
            and ((not shutdown) or execution_in_progress)
        ):
            latest_securities, latest_position = observe_position(
                s,
                ticker,
                target_position,
                origin_mono,
                "position_stale_grace",
                observations,
            )

            if latest_position == target_position:
                return (
                    latest_securities,
                    latest_position,
                    True,
                    True,
                )

            time.sleep(VERIFY_POLL_SECONDS)

    return (
        latest_securities,
        latest_position,
        all_filled,
        stale_position,
    )


def build_order_row(
    tender_row,
    execution_id,
    prediction,
    role,
    action,
    observed_at_utc,
    submit_start_relative_ms,
    submit_response_relative_ms,
    submit_latency_ms,
    pacing_sleep_ms,
    response,
    error=None,
):
    (
        order_id,
        quantity_filled,
        actual_vwap,
        actual_price,
        status,
    ) = extract_order_actuals(response, {})

    improvement, adverse = signed_execution_metrics(
        action,
        prediction.get("predicted_slice_vwap"),
        actual_vwap,
    )

    return {
        "run_id": RUN_ID,
        "heat_id": tender_row["heat_id"],
        "tender_id": tender_row["tender_id"],
        "execution_id": execution_id,

        "slice_no": prediction["slice_no"],
        "order_role": role,
        "observed_at_utc": observed_at_utc,

        "ticker": tender_row["ticker"],
        "action": action,
        "requested_qty": prediction["requested_qty"],

        "predicted_slice_vwap":
            prediction.get("predicted_slice_vwap"),
        "predicted_slice_worst":
            prediction.get("predicted_slice_worst"),
        "predicted_slice_covered":
            bool(prediction.get("predicted_slice_covered")),

        "order_id": order_id,
        "submit_start_relative_ms":
            submit_start_relative_ms,
        "submit_response_relative_ms":
            submit_response_relative_ms,
        "submit_latency_ms": submit_latency_ms,
        "pacing_sleep_ms": pacing_sleep_ms,

        "quantity_filled": quantity_filled,
        "actual_vwap": actual_vwap,
        "actual_price": actual_price,

        "signed_price_improvement": improvement,
        "adverse_slippage_per_share": adverse,

        "status": status,
        "response_json": json_dumps(response),
        "detail_json": json_dumps({}),
        "error": error,

        "_response": response,
    }


def submit_pipeline(
    s,
    tender_row,
    arrival_book,
    execution_id,
    origin_mono,
    timeline,
    order_rows,
    quantity,
    action,
    role,
    starting_slice_no,
    advertised_rate,
):
    """
    Submit the whole hedge sequence without waiting for position/fill display
    updates between slices.

    The request/response round trip itself is still synchronous. That is
    deliberate: it gives us the server's order ID/response and avoids the
    failure ambiguity of blind multi-threaded POSTs. Since V3 measured ~30-50ms
    POST RTT and the demo case advertised a much higher API order rate, this
    removes the dominant ~170ms-per-slice position-observation wait while
    keeping error handling straightforward.
    """
    ticker = tender_row["ticker"]
    sizes = split_quantity(
        quantity,
        MAX_ORDER_SIZE[ticker],
    )

    if len(sizes) > MAX_HEDGE_SLICES:
        raise RITError(
            f"Hedge requires {len(sizes)} slices, above "
            f"MAX_HEDGE_SLICES={MAX_HEDGE_SLICES}."
        )

    if action == "SELL":
        side = arrival_book.get("bids", [])
    else:
        side = arrival_book.get("asks", [])

    predictions = plan_sliced_sweep(side, sizes)

    # Renumber predictions if this is a repair burst.
    for i, prediction in enumerate(
        predictions,
        start=starting_slice_no,
    ):
        prediction["slice_no"] = i

    min_interval = min_order_interval_seconds(
        advertised_rate
    )

    last_submit_start = None
    submitted_qty = 0
    submit_errors = 0

    for prediction in predictions:
        pacing_sleep_ms = pace_submission(
            last_submit_start,
            min_interval,
        )

        submit_start = time.perf_counter()
        last_submit_start = submit_start

        start_relative_ms = (
            submit_start - origin_mono
        ) * 1000.0

        timeline_mark(
            timeline,
            origin_mono,
            f"{role}_order_{prediction['slice_no']}_submit_start",
            {
                "qty": prediction["requested_qty"],
                "action": action,
                "pacing_sleep_ms": pacing_sleep_ms,
            },
        )

        response = {}
        error = None
        submit_latency_ms = None

        try:
            response, submit_latency_ms = send_market_order_fast(
                s,
                ticker,
                action,
                prediction["requested_qty"],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            submit_errors += 1

        response_relative_ms = (
            time.perf_counter() - origin_mono
        ) * 1000.0

        row = build_order_row(
            tender_row=tender_row,
            execution_id=execution_id,
            prediction=prediction,
            role=role,
            action=action,
            observed_at_utc=utc_now(),
            submit_start_relative_ms=start_relative_ms,
            submit_response_relative_ms=response_relative_ms,
            submit_latency_ms=submit_latency_ms,
            pacing_sleep_ms=pacing_sleep_ms,
            response=response,
            error=error,
        )
        order_rows.append(row)

        timeline_mark(
            timeline,
            origin_mono,
            f"{role}_order_{prediction['slice_no']}_submit_response",
            {
                "order_id": row["order_id"],
                "latency_ms": submit_latency_ms,
                "error": error,
            },
        )

        if error:
            # The outcome of an exception is potentially ambiguous. Stop adding
            # new exposure-reducing instructions, reconcile actual orders/position,
            # then use the repair path if objective residual inventory remains.
            break

        submitted_qty += prediction["requested_qty"]

    return {
        "submitted_qty": submitted_qty,
        "submit_errors": submit_errors,
        "min_interval_seconds": min_interval,
        "slice_count": len(predictions),
    }


def _execute_accepted_tender_v4(
    s,
    tender,
    tender_row,
    arrival_book,
    tick,
    detected_mono,
):
    global execution_in_progress

    execution_in_progress = True
    rate_hits_start = RATE_LIMIT_HITS

    origin_mono = detected_mono
    execution_id = (
        f"{RUN_ID}-{tender_row['heat_id']}-"
        f"T{tender_row['tender_id']}-{uuid.uuid4().hex[:6]}"
    )

    ticker = tender_row["ticker"]
    tender_action = tender_row["action"]
    hedge_action = tender_row["unwind_side"]
    qty = int(tender_row["quantity"])

    pre_securities = tender_row["_pre_securities"]
    pre_positions = positions_from_securities(
        pre_securities
    )
    target_position = pre_positions.get(ticker, 0)

    realized_before = security_realized(
        pre_securities,
        ticker,
    )

    advertised_rate = tender_row.get(
        "api_orders_per_second"
    )
    eff_rate = effective_order_rate(
        advertised_rate
    )
    min_interval = min_order_interval_seconds(
        advertised_rate
    )

    timeline = []
    observations = []
    order_rows = []

    timeline_mark(
        timeline,
        origin_mono,
        "analysis_complete",
        {
            "decision": tender_row["decision"],
            "arrival_unwind_vwap":
                tender_row["unwind_vwap"],
            "advertised_order_rate":
                advertised_rate,
        },
    )

    decision_complete_mono = time.perf_counter()
    accept_start_mono = time.perf_counter()

    timeline_mark(
        timeline,
        origin_mono,
        "accept_request_start",
    )

    failure_reason = None
    repair_qty = 0
    repair_order_count = 0
    stale_position_observed = False

    accept_transport_exception = None

    try:
        accept_result = accept_tender_fast(
            s,
            tender,
        )
    except Exception as exc:
        accept_transport_exception = (
            f"{type(exc).__name__}:{exc}"
        )
        accept_result = {
            "success": False,
            "request_at": utc_now(),
            "response_at": utc_now(),
            "latency_ms": None,
            "body": {
                "transport_exception":
                    accept_transport_exception
            },
        }

    accept_response_mono = time.perf_counter()

    if (
        not accept_result["success"]
        and accept_transport_exception is not None
    ):
        (
            resolved_success,
            resolved_position,
            resolve_reason,
        ) = resolve_ambiguous_tender_accept(
            s=s,
            tender=tender,
            position_before=target_position,
        )

        timeline_mark(
            timeline,
            origin_mono,
            "accept_transport_reconciliation",
            {
                "resolved_success": resolved_success,
                "observed_position": resolved_position,
                "reason": resolve_reason,
            },
        )

        if resolved_success is True:
            accept_result["success"] = True
            accept_result["body"][
                "recovered_after_transport_exception"
            ] = True
            failure_reason = None
        else:
            failure_reason = (
                "accept_transport_"
                f"{resolve_reason}:"
                f"{accept_transport_exception}"
            )

    timeline_mark(
        timeline,
        origin_mono,
        "accept_response",
        {
            "success": accept_result["success"],
            "latency_ms": accept_result["latency_ms"],
            "response": accept_result["body"],
        },
    )

    if not accept_result["success"]:
        failure_reason = (
            failure_reason
            or "tender_accept_failed"
        )

        summary = {
            "execution_id": execution_id,
            "run_id": RUN_ID,
            "heat_id": tender_row["heat_id"],
            "tender_id": tender_row["tender_id"],
            "ticker": ticker,
            "tender_action": tender_action,
            "quantity": qty,
            "tender_price": tender_row["tender_price"],
            "execution_mode": EXECUTION_MODE,
            "fee_model": FEE_MODEL_LABEL,
            "advertised_order_rate": advertised_rate,
            "effective_order_rate": eff_rate,
            "min_submit_interval_ms": min_interval * 1000.0,
            "predicted_arrival_unwind_vwap":
                tender_row["unwind_vwap"],
            "predicted_arrival_after_fee_edge":
                tender_row["fee_adjusted_edge_per_share"],
            "predicted_arrival_profit_after_fee":
                tender_row["immediate_profit_after_fees"],
            "favorable_imbalance_full":
                tender_row["favorable_imbalance_full"],
            "accept_success": False,
            "accept_latency_ms": accept_result["latency_ms"],
            "decision_to_accept_start_ms":
                (accept_start_mono - decision_complete_mono) * 1000.0,
            "accept_to_first_order_ms": None,
            "accept_to_last_order_ms": None,
            "burst_submission_span_ms": None,
            "accept_to_target_position_ms": None,
            "total_execution_ms":
                (time.perf_counter() - origin_mono) * 1000.0,
            "hedge_action": hedge_action,
            "hedge_slice_count": 0,
            "hedge_submitted_qty": 0,
            "hedge_reported_filled_qty": 0,
            "realized_hedge_vwap": None,
            "execution_error_per_share": None,
            "execution_error_dollars": None,
            "gross_block_profit": None,
            "profit_less_unwind_fee": None,
            "security_realized_before": realized_before,
            "security_realized_after": realized_before,
            "security_realized_delta": 0.0,
            "fee_model_error": None,
            "target_position": target_position,
            "final_position": target_position,
            "fully_flat": True,
            "all_orders_filled": False,
            "repair_qty": 0,
            "repair_order_count": 0,
            "rate_limit_hits_during_execution":
                RATE_LIMIT_HITS - rate_hits_start,
            "trader_fines_after": None,
            "failure_reason": failure_reason,
            "accept_response_json":
                json_dumps(accept_result["body"]),
            "pre_securities_json":
                json_dumps(pre_securities),
            "post_securities_json":
                json_dumps(pre_securities),
            "trader_after_json": json_dumps({}),
            "_detected_at_utc":
                tender_row["observed_at_utc"],
            "_started_at_utc":
                accept_result["request_at"],
            "_completed_at_utc": utc_now(),
        }

        execution_in_progress = False

        return {
            "summary": summary,
            "order_rows": order_rows,
            "timeline": timeline,
            "position_observations": observations,
        }

    # =====================================================================
    # HOT PATH: acceptance succeeded -> immediately submit exact hedge block.
    # NO position polling, book GET, SQLite write, CSV write or console print
    # is allowed between this point and the end of the initial submit burst.
    # =====================================================================

    pipeline = submit_pipeline(
        s=s,
        tender_row=tender_row,
        arrival_book=arrival_book,
        execution_id=execution_id,
        origin_mono=origin_mono,
        timeline=timeline,
        order_rows=order_rows,
        quantity=qty,
        action=hedge_action,
        role="PIPELINE",
        starting_slice_no=1,
        advertised_rate=advertised_rate,
    )

    burst_end_mono = time.perf_counter()

    timeline_mark(
        timeline,
        origin_mono,
        "initial_burst_submitted",
        {
            "submitted_qty": pipeline["submitted_qty"],
            "submit_errors": pipeline["submit_errors"],
        },
    )

    # Verification is intentionally POST-burst.
    (
        latest_securities,
        latest_position,
        all_filled,
        stale_position_observed,
    ) = wait_for_burst_settlement(
        s=s,
        ticker=ticker,
        target_position=target_position,
        order_rows=order_rows,
        origin_mono=origin_mono,
        observations=observations,
    )

    timeline_mark(
        timeline,
        origin_mono,
        "initial_burst_verified",
        {
            "position": latest_position,
            "target": target_position,
            "all_orders_filled": all_filled,
            "stale_position_observed":
                stale_position_observed,
        },
    )

    # Objective residual-repair path.
    #
    # If all submitted orders report fully filled, do not over-hedge merely
    # because /securities is lagging. Repair is allowed only when the position
    # has settled to a non-target value AND order records are not all filled.
    if (
        latest_position is not None
        and latest_position != target_position
        and not all_filled
    ):
        for repair_round in range(
            1,
            MAX_REPAIR_ROUNDS + 1,
        ):
            residual = latest_position - target_position

            if residual == 0:
                break

            repair_action = (
                "SELL" if residual > 0 else "BUY"
            )
            this_repair_qty = abs(residual)
            repair_qty += this_repair_qty

            repair_book = get_book(
                s,
                ticker,
            )

            starting_slice = len(order_rows) + 1

            repair_result = submit_pipeline(
                s=s,
                tender_row=tender_row,
                arrival_book=repair_book,
                execution_id=execution_id,
                origin_mono=origin_mono,
                timeline=timeline,
                order_rows=order_rows,
                quantity=this_repair_qty,
                action=repair_action,
                role=f"REPAIR_{repair_round}",
                starting_slice_no=starting_slice,
                advertised_rate=advertised_rate,
            )

            repair_order_count += len(
                split_quantity(
                    this_repair_qty,
                    MAX_ORDER_SIZE[ticker],
                )
            )

            (
                latest_securities,
                latest_position,
                all_filled,
                stale_now,
            ) = wait_for_burst_settlement(
                s=s,
                ticker=ticker,
                target_position=target_position,
                order_rows=order_rows,
                origin_mono=origin_mono,
                observations=observations,
            )

            stale_position_observed = (
                stale_position_observed
                or stale_now
            )

            if latest_position == target_position:
                break

    # One final refresh after the execution/repair path.
    try:
        post_securities = get_securities(s)
    except Exception:
        post_securities = latest_securities or []

    final_position = positions_from_securities(
        post_securities
    ).get(ticker, latest_position)

    update_order_rows_from_api(
        s,
        order_rows,
    )

    all_orders_filled = orders_fully_filled(
        order_rows
    )

    realized_hedge_vwap, reported_filled_qty = (
        weighted_realized_vwap(order_rows)
    )

    realized_after = security_realized(
        post_securities,
        ticker,
    )

    security_realized_delta = (
        None
        if realized_before is None
        or realized_after is None
        else realized_after - realized_before
    )

    gross = None
    less_unwind_fee = None

    # The block-P&L model is only strictly comparable when we have exactly the
    # tender quantity of reported hedge fills.
    if reported_filled_qty == qty:
        gross, less_unwind_fee = execution_profit_metrics(
            tender_action,
            tender_row["tender_price"],
            qty,
            realized_hedge_vwap,
        )

    predicted_arrival = tender_row["unwind_vwap"]

    if (
        realized_hedge_vwap is not None
        and predicted_arrival is not None
        and reported_filled_qty == qty
    ):
        if hedge_action == "SELL":
            execution_error_per_share = (
                predicted_arrival
                - realized_hedge_vwap
            )
        else:
            execution_error_per_share = (
                realized_hedge_vwap
                - predicted_arrival
            )

        execution_error_dollars = (
            execution_error_per_share * qty
        )
    else:
        execution_error_per_share = None
        execution_error_dollars = None

    fee_model_error = (
        None
        if security_realized_delta is None
        or less_unwind_fee is None
        else security_realized_delta
        - less_unwind_fee
    )

    try:
        trader_after = get_trader(s)
    except Exception:
        trader_after = {}

    fines_after = first_numeric(
        trader_after,
        ["fines", "fine", "penalties", "penalty"],
    )

    fully_flat = (
        final_position == target_position
    )

    if not fully_flat and failure_reason is None:
        failure_reason = (
            f"residual_position_{final_position}_"
            f"target_{target_position}"
        )

    if (
        failure_reason is None
        and not fully_flat
        and not all_orders_filled
    ):
        failure_reason = "residual_and_unfilled_orders"

    if (
        failure_reason is None
        and fully_flat
        and reported_filled_qty != qty
    ):
        failure_reason = (
            f"hedge_fill_qty_mismatch_"
            f"{reported_filled_qty}_vs_{qty}"
        )

    first_order_start = None
    last_order_response = None

    if order_rows:
        first_order_start = min(
            row["submit_start_relative_ms"]
            for row in order_rows
            if row["submit_start_relative_ms"] is not None
        )
        last_order_response = max(
            row["submit_response_relative_ms"]
            for row in order_rows
            if row["submit_response_relative_ms"] is not None
        )

    accept_response_relative_ms = (
        accept_response_mono - origin_mono
    ) * 1000.0

    accept_to_first_order_ms = (
        None
        if first_order_start is None
        else first_order_start
        - accept_response_relative_ms
    )

    accept_to_last_order_ms = (
        None
        if last_order_response is None
        else last_order_response
        - accept_response_relative_ms
    )

    burst_submission_span_ms = (
        None
        if first_order_start is None
        or last_order_response is None
        else last_order_response
        - first_order_start
    )

    # First observation at target gives us the post-accept-to-flat display lag.
    target_obs = [
        obs
        for obs in observations
        if obs["position"] == target_position
    ]

    accept_to_target_position_ms = None

    if target_obs:
        accept_to_target_position_ms = (
            min(
                obs["relative_ms"]
                for obs in target_obs
            )
            - accept_response_relative_ms
        )

    timeline_mark(
        timeline,
        origin_mono,
        "execution_complete",
        {
            "final_position": final_position,
            "target_position": target_position,
            "all_orders_filled": all_orders_filled,
            "realized_hedge_vwap":
                realized_hedge_vwap,
            "security_realized_delta":
                security_realized_delta,
        },
    )

    summary = {
        "execution_id": execution_id,
        "run_id": RUN_ID,
        "heat_id": tender_row["heat_id"],
        "tender_id": tender_row["tender_id"],
        "ticker": ticker,
        "tender_action": tender_action,
        "quantity": qty,
        "tender_price": tender_row["tender_price"],

        "execution_mode": EXECUTION_MODE,
        "fee_model": FEE_MODEL_LABEL,

        "advertised_order_rate": advertised_rate,
        "effective_order_rate": eff_rate,
        "min_submit_interval_ms":
            min_interval * 1000.0,

        "predicted_arrival_unwind_vwap":
            tender_row["unwind_vwap"],
        "predicted_arrival_after_fee_edge":
            tender_row["fee_adjusted_edge_per_share"],
        "predicted_arrival_profit_after_fee":
            tender_row["immediate_profit_after_fees"],
        "favorable_imbalance_full":
            tender_row["favorable_imbalance_full"],

        "accept_success": True,
        "accept_latency_ms":
            accept_result["latency_ms"],
        "decision_to_accept_start_ms":
            (accept_start_mono - decision_complete_mono)
            * 1000.0,
        "accept_to_first_order_ms":
            accept_to_first_order_ms,
        "accept_to_last_order_ms":
            accept_to_last_order_ms,
        "burst_submission_span_ms":
            burst_submission_span_ms,
        "accept_to_target_position_ms":
            accept_to_target_position_ms,
        "total_execution_ms":
            (
                time.perf_counter()
                - origin_mono
            ) * 1000.0,

        "hedge_action": hedge_action,
        "hedge_slice_count": len(order_rows),
        "hedge_submitted_qty": sum(
            row["requested_qty"]
            for row in order_rows
            if not row.get("error")
        ),
        "hedge_reported_filled_qty":
            reported_filled_qty,

        "realized_hedge_vwap":
            realized_hedge_vwap,
        "execution_error_per_share":
            execution_error_per_share,
        "execution_error_dollars":
            execution_error_dollars,

        "gross_block_profit": gross,
        "profit_less_unwind_fee":
            less_unwind_fee,

        "security_realized_before":
            realized_before,
        "security_realized_after":
            realized_after,
        "security_realized_delta":
            security_realized_delta,
        "fee_model_error":
            fee_model_error,

        "target_position": target_position,
        "final_position": final_position,
        "fully_flat": fully_flat,
        "all_orders_filled": all_orders_filled,

        "repair_qty": repair_qty,
        "repair_order_count":
            repair_order_count,

        "rate_limit_hits_during_execution":
            RATE_LIMIT_HITS - rate_hits_start,
        "trader_fines_after": fines_after,

        "failure_reason": failure_reason,

        "accept_response_json":
            json_dumps(accept_result["body"]),
        "pre_securities_json":
            json_dumps(pre_securities),
        "post_securities_json":
            json_dumps(post_securities),
        "trader_after_json":
            json_dumps(trader_after),

        "_detected_at_utc":
            tender_row["observed_at_utc"],
        "_started_at_utc":
            accept_result["request_at"],
        "_completed_at_utc":
            utc_now(),
    }

    execution_in_progress = False

    return {
        "summary": summary,
        "order_rows": order_rows,
        "timeline": timeline,
        "position_observations": observations,
    }


def execute_accepted_tender_v4(
    s,
    tender,
    tender_row,
    arrival_book,
    tick,
    detected_mono,
):
    global execution_in_progress

    execution_in_progress = True

    try:
        return _execute_accepted_tender_v4(
            s=s,
            tender=tender,
            tender_row=tender_row,
            arrival_book=arrival_book,
            tick=tick,
            detected_mono=detected_mono,
        )
    finally:
        execution_in_progress = False


def persist_execution_bundle(conn, bundle):
    summary = bundle["summary"]
    order_rows = bundle["order_rows"]
    timeline = bundle["timeline"]
    observations = bundle["position_observations"]

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO execution_sessions_v4 (
                execution_id, run_id, heat_id, tender_id,
                ticker, tender_action, quantity, tender_price,
                detected_at_utc, started_at_utc, completed_at_utc,
                execution_mode, fee_model,
                advertised_order_rate, effective_order_rate,
                min_submit_interval_ms,
                predicted_arrival_unwind_vwap,
                predicted_arrival_after_fee_edge,
                predicted_arrival_profit_after_fee,
                favorable_imbalance_full,
                accept_success,
                accept_latency_ms,
                decision_to_accept_start_ms,
                accept_to_first_order_ms,
                accept_to_last_order_ms,
                burst_submission_span_ms,
                accept_to_target_position_ms,
                total_execution_ms,
                hedge_action, hedge_slice_count,
                hedge_submitted_qty,
                hedge_reported_filled_qty,
                realized_hedge_vwap,
                execution_error_per_share,
                execution_error_dollars,
                gross_block_profit,
                profit_less_unwind_fee,
                security_realized_before,
                security_realized_after,
                security_realized_delta,
                fee_model_error,
                target_position, final_position,
                fully_flat, all_orders_filled,
                repair_qty, repair_order_count,
                rate_limit_hits_during_execution,
                trader_fines_after,
                accept_response_json,
                pre_securities_json,
                post_securities_json,
                trader_after_json,
                failure_reason
            ) VALUES (
                ?,?,?,?,?,?,?,?,
                ?,?,?,
                ?,?,
                ?,?,?,
                ?,?,?,?,
                ?,?,?,?,?,
                ?,?,
                ?,?,?,?,
                ?,?,?,
                ?,?,
                ?,?,?,?,
                ?,?,
                ?,?,
                ?,?,
                ?,?,
                ?,?,?,?,
                ?,?
            )
            """,
            (
                summary["execution_id"],
                summary["run_id"],
                summary["heat_id"],
                summary["tender_id"],
                summary["ticker"],
                summary["tender_action"],
                summary["quantity"],
                summary["tender_price"],

                summary["_detected_at_utc"],
                summary["_started_at_utc"],
                summary["_completed_at_utc"],

                summary["execution_mode"],
                summary["fee_model"],

                summary["advertised_order_rate"],
                summary["effective_order_rate"],
                summary["min_submit_interval_ms"],

                summary[
                    "predicted_arrival_unwind_vwap"
                ],
                summary[
                    "predicted_arrival_after_fee_edge"
                ],
                summary[
                    "predicted_arrival_profit_after_fee"
                ],
                summary[
                    "favorable_imbalance_full"
                ],

                int(summary["accept_success"]),
                summary["accept_latency_ms"],
                summary[
                    "decision_to_accept_start_ms"
                ],
                summary[
                    "accept_to_first_order_ms"
                ],
                summary[
                    "accept_to_last_order_ms"
                ],
                summary[
                    "burst_submission_span_ms"
                ],
                summary[
                    "accept_to_target_position_ms"
                ],
                summary["total_execution_ms"],

                summary["hedge_action"],
                summary["hedge_slice_count"],
                summary["hedge_submitted_qty"],
                summary[
                    "hedge_reported_filled_qty"
                ],

                summary["realized_hedge_vwap"],
                summary[
                    "execution_error_per_share"
                ],
                summary[
                    "execution_error_dollars"
                ],

                summary["gross_block_profit"],
                summary[
                    "profit_less_unwind_fee"
                ],

                summary[
                    "security_realized_before"
                ],
                summary[
                    "security_realized_after"
                ],
                summary[
                    "security_realized_delta"
                ],
                summary["fee_model_error"],

                summary["target_position"],
                summary["final_position"],
                int(summary["fully_flat"]),
                int(summary[
                    "all_orders_filled"
                ]),

                summary["repair_qty"],
                summary["repair_order_count"],

                summary[
                    "rate_limit_hits_during_execution"
                ],
                summary["trader_fines_after"],

                summary["accept_response_json"],
                summary["pre_securities_json"],
                summary["post_securities_json"],
                summary["trader_after_json"],
                summary["failure_reason"],
            ),
        )

        conn.executemany(
            """
            INSERT INTO order_events_v4 (
                run_id, heat_id, tender_id,
                execution_id,
                slice_no, order_role,
                observed_at_utc,
                ticker, action, requested_qty,
                predicted_slice_vwap,
                predicted_slice_worst,
                predicted_slice_covered,
                order_id,
                submit_start_relative_ms,
                submit_response_relative_ms,
                submit_latency_ms,
                pacing_sleep_ms,
                quantity_filled,
                actual_vwap, actual_price,
                signed_price_improvement,
                adverse_slippage_per_share,
                status,
                response_json, detail_json,
                error
            ) VALUES (
                ?,?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?,
                ?,?,?,?,?,
                ?,?,?,
                ?,?,
                ?,
                ?,?,?
            )
            """,
            [
                (
                    row["run_id"],
                    row["heat_id"],
                    row["tender_id"],
                    row["execution_id"],
                    row["slice_no"],
                    row["order_role"],
                    row["observed_at_utc"],
                    row["ticker"],
                    row["action"],
                    row["requested_qty"],
                    row[
                        "predicted_slice_vwap"
                    ],
                    row[
                        "predicted_slice_worst"
                    ],
                    int(row[
                        "predicted_slice_covered"
                    ]),
                    row["order_id"],
                    row[
                        "submit_start_relative_ms"
                    ],
                    row[
                        "submit_response_relative_ms"
                    ],
                    row["submit_latency_ms"],
                    row["pacing_sleep_ms"],
                    row["quantity_filled"],
                    row["actual_vwap"],
                    row["actual_price"],
                    row[
                        "signed_price_improvement"
                    ],
                    row[
                        "adverse_slippage_per_share"
                    ],
                    row["status"],
                    row["response_json"],
                    row["detail_json"],
                    row["error"],
                )
                for row in order_rows
            ],
        )

        conn.executemany(
            """
            INSERT INTO latency_events_v4 (
                run_id, heat_id, tender_id,
                execution_id,
                event_seq, event_name,
                observed_at_utc,
                relative_ms, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    RUN_ID,
                    summary["heat_id"],
                    summary["tender_id"],
                    summary["execution_id"],
                    row["event_seq"],
                    row["event_name"],
                    row["observed_at_utc"],
                    row["relative_ms"],
                    row["payload_json"],
                )
                for row in timeline
            ],
        )

        conn.executemany(
            """
            INSERT INTO position_observations_v4 (
                run_id, heat_id, tender_id,
                execution_id,
                observed_at_utc, relative_ms,
                ticker, position,
                target_position, phase
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    RUN_ID,
                    summary["heat_id"],
                    summary["tender_id"],
                    summary["execution_id"],
                    row["observed_at_utc"],
                    row["relative_ms"],
                    row["ticker"],
                    row["position"],
                    row["target_position"],
                    row["phase"],
                )
                for row in observations
            ],
        )

    append_csv(
        EXECUTION_CSV,
        EXECUTION_CSV_FIELDS,
        summary,
    )

    for row in order_rows:
        append_csv(
            ORDER_CSV,
            ORDER_CSV_FIELDS,
            row,
        )


def flatten_ticker_emergency(s, ticker):
    """
    Emergency end-of-heat flatten. This is intentionally separate from the
    normal tender execution dataset.
    """
    securities = get_securities(s)
    api_rate = advertised_order_rate(
        securities,
        ticker,
    )
    min_interval = min_order_interval_seconds(
        api_rate
    )
    last_submit = None

    for _ in range(MAX_HEDGE_SLICES):
        securities = get_securities(s)
        pos = positions_from_securities(
            securities
        )[ticker]

        if pos == 0:
            return

        action = (
            "SELL" if pos > 0 else "BUY"
        )
        qty = min(
            abs(pos),
            MAX_ORDER_SIZE[ticker],
        )

        pace_submission(
            last_submit,
            min_interval,
        )

        last_submit = time.perf_counter()

        try:
            send_market_order_fast(
                s,
                ticker,
                action,
                qty,
            )
        except Exception:
            pass


def flatten_all_emergency(s):
    positions = get_positions(s)

    for ticker, pos in positions.items():
        if pos:
            print(
                f"[EMERGENCY] Hard flatten "
                f"{ticker}: {pos:,}"
            )
            flatten_ticker_emergency(
                s,
                ticker,
            )


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
        "live_intervention": False,
    }


def mark_tracker_intervention(trackers, tender_id):
    if tender_id in trackers:
        trackers[tender_id]["live_intervention"] = True


def sample_trackers(s, conn, trackers, current_tick, current_heat_id):
    finished = []

    for tid, tr in list(trackers.items()):
        tender_row = tr["tender_row"]

        if tender_row["heat_id"] != current_heat_id:
            finished.append(tid)
            continue

        elapsed = current_tick - tender_row["tick"]

        if elapsed < 0 or elapsed > FOLLOWUP_TICKS:
            finished.append(tid)
            continue

        if elapsed == 0:
            continue

        if elapsed % FOLLOWUP_EVERY_TICKS != 0:
            continue

        if elapsed <= tr["last_sampled_elapsed"]:
            continue

        book = get_book(s, tender_row["ticker"])
        row = followup_row(
            tender_row,
            book,
            current_tick,
            live_intervention=tr["live_intervention"],
        )

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
# STARTUP VALIDATION / SELF-TESTS
# ============================================================================

def run_self_tests():
    book = {
        "bids": [
            {"price": 10.00, "quantity": 10_000, "quantity_filled": 0},
            {"price": 9.99, "quantity": 10_000, "quantity_filled": 0},
            {"price": 9.98, "quantity": 10_000, "quantity_filled": 0},
        ],
        "asks": [
            {"price": 10.01, "quantity": 10_000, "quantity_filled": 0},
            {"price": 10.02, "quantity": 10_000, "quantity_filled": 0},
            {"price": 10.03, "quantity": 10_000, "quantity_filled": 0},
        ],
    }

    assert split_quantity(72_000, 25_000) == [
        25_000, 25_000, 22_000
    ]
    assert split_quantity(74_000, 10_000) == [
        10_000, 10_000, 10_000, 10_000,
        10_000, 10_000, 10_000, 4_000,
    ]

    sizes = [10_000, 10_000, 10_000]
    preds = plan_sliced_sweep(
        book["bids"],
        sizes,
    )
    assert all(
        p["predicted_slice_covered"]
        for p in preds
    )

    aggregate = sum(
        p["predicted_slice_vwap"]
        * p["requested_qty"]
        for p in preds
    ) / sum(sizes)

    full, _, _, covered = sweep_vwap(
        book["bids"],
        sum(sizes),
    )

    assert covered
    assert abs(aggregate - full) < 1e-12

    generic = imbalance(book, None)
    assert favorable_imbalance("BUY", generic) == generic
    assert favorable_imbalance("SELL", generic) == -generic

    assert abs(
        effective_order_rate(200.0) - 180.0
    ) < 1e-12

    assert abs(
        min_order_interval_seconds(200.0)
        - (1.0 / 180.0)
    ) < 1e-12

    p, net, gross = risk_after_tender(
        {"CRZY": 0, "TAME": 0},
        "CRZY",
        "SELL",
        72_000,
    )
    assert p["CRZY"] == -72_000
    assert net == -72_000
    assert gross == 72_000

    print("V4 self-tests: PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LT3 V4 tender-pricing/research/execution engine"
        )
    )

    parser.add_argument(
        "--live-demo",
        action="store_true",
        help=(
            "Enable tender acceptance and market-order hedging. "
            "Without this flag V4 runs in paper mode."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run local pure-function smoke tests and exit "
            "without connecting to RIT."
        ),
    )

    return parser.parse_args()


def validate_startup():
    if (
        not API_KEY
        or API_KEY == "PASTE_YOUR_RIT_API_KEY_HERE"
    ):
        raise SystemExit(
            "Set API_KEY in this file or set the PowerShell "
            "environment variable RIT_API_KEY first."
        )

    if EXECUTION_MODE != "PIPELINED_MARKET_V4":
        raise SystemExit(
            "Unexpected EXECUTION_MODE. V4 is calibrated for "
            "PIPELINED_MARKET_V4 only."
        )

    if USE_IMBALANCE_FOR_EXECUTION:
        raise SystemExit(
            "V4 intentionally keeps imbalance research-only. "
            "Set USE_IMBALANCE_FOR_EXECUTION=False."
        )


def persist_tender_research(
    conn,
    tender_row,
    arrival_book,
    tick,
):
    """
    Persist the heavy research data AFTER the live hot path. The arrival book
    is the untouched snapshot captured before tender acceptance, so deferring
    the write does not change the data.
    """
    insert_tender(
        conn,
        tender_row,
    )

    append_csv(
        TENDER_CSV,
        TENDER_CSV_FIELDS,
        tender_row,
    )

    insert_policy_grid(
        conn,
        tender_row,
        tender_row["_limit_ok"],
    )

    arrival_followup = followup_row(
        tender_row,
        arrival_book,
        tick,
        live_intervention=False,
    )

    insert_followup(
        conn,
        arrival_followup,
    )

    insert_book_levels(
        conn,
        tender_row["heat_id"],
        tender_row["tender_id"],
        arrival_followup["observed_at_utc"],
        tick,
        0,
        tender_row["ticker"],
        arrival_book,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    global LIVE

    args = parse_args()

    if args.self_test:
        run_self_tests()
        return

    LIVE = bool(args.live_demo)
    validate_startup()
    conn = open_db()

    print("Connected to LT3 V4 research/execution engine.")
    print(f"RUN_ID={RUN_ID}")
    print(f"LIVE={LIVE}  BASE={BASE}")
    print(f"Database:      {DB_FILE}")
    print(f"Tender CSV:    {TENDER_CSV}")
    print(f"Execution CSV: {EXECUTION_CSV}")
    print(f"Order CSV:     {ORDER_CSV}")

    print(
        f"Current policy UNCHANGED: after-fee edge minus "
        f"${SAFETY_BUFFER_PER_SHARE:.2f} safety >= "
        f"${MIN_NET_EDGE_PER_SHARE:.2f}/sh and scored >= "
        f"${MIN_EXPECTED_PROFIT:,.0f}."
    )

    print(
        f"Execution: {EXECUTION_MODE}; dynamic order-rate pacing at "
        f"{ORDER_RATE_SAFETY_FRACTION:.0%} of the advertised case limit "
        f"(fallback {ORDER_RATE_FALLBACK_PER_SECOND:.1f}/sec)."
    )

    print(
        f"Tender poll target: every {POLL_SECONDS * 1000:.0f} ms. "
        f"Heavy SQLite/CSV persistence is deferred until after the "
        f"live hedge burst."
    )

    print(
        f"Follow-ups: every {FOLLOWUP_EVERY_TICKS} tick(s), "
        f"{FOLLOWUP_TICKS} ticks, "
        f"{BOOK_LEVELS_TO_STORE} levels/side."
    )

    if LIVE:
        print(
            "\n*** LIVE DEMO EXECUTION ENABLED ***\n"
            "Qualifying tenders WILL be accepted. A successful tender "
            "acceptance response immediately authorizes a pipelined "
            "max-size MARKET hedge. Position/fill verification occurs "
            "after the initial hedge burst.\n"
        )
    else:
        print(
            "\n[PAPER MODE] No tender/order actions will be submitted.\n"
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
                period = int(
                    case.get("period", 0) or 0
                )

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
                    heat is None
                    or tick_reset
                    or became_active
                ):
                    if heat is not None:
                        finalize_heat(
                            conn,
                            heat,
                            (
                                last_tick
                                if last_tick is not None
                                else tick
                            ),
                            (
                                last_period
                                if last_period is not None
                                else period
                            ),
                            (
                                "tick_reset"
                                if tick_reset
                                else "restart"
                            ),
                        )

                    heat_seq += 1
                    heat = new_heat(
                        conn,
                        heat_seq,
                        tick,
                        period,
                    )
                    seen.clear()
                    trackers.clear()

                if status != "ACTIVE":
                    if (
                        heat is not None
                        and last_status == "ACTIVE"
                    ):
                        finalize_heat(
                            conn,
                            heat,
                            (
                                last_tick
                                if last_tick is not None
                                else tick
                            ),
                            (
                                last_period
                                if last_period is not None
                                else period
                            ),
                            "inactive",
                        )

                        print(
                            f"### HEAT COMPLETE {heat['heat_id']} | "
                            f"tenders={heat['tender_count']} "
                            f"policy_accepts={heat['accept_count']} "
                            f"policy_declines={heat['decline_count']} "
                            f"scored=${heat['scored_profit_sum']:,.0f} ###"
                        )

                        heat = None
                        trackers.clear()
                        seen.clear()

                    last_tick = tick
                    last_status = status
                    last_period = period
                    time.sleep(0.10)
                    continue

                if heat is None:
                    heat_seq += 1
                    heat = new_heat(
                        conn,
                        heat_seq,
                        tick,
                        period,
                    )

                # End-of-period backstop. No new tender logic near the close;
                # in live mode any residual inventory is aggressively flattened.
                if tick >= HARD_FLATTEN_TICK:
                    if LIVE:
                        flatten_all_emergency(s)

                    last_tick = tick
                    last_status = status
                    last_period = period
                    time.sleep(0.10)
                    continue

                # =========================================================
                # TENDER DETECTION GETS PRIORITY OVER RESEARCH FOLLOW-UPS.
                # V3 sampled active trackers before checking /tenders. V4
                # reverses that ordering so logging can never intentionally
                # sit ahead of a new institutional order.
                # =========================================================
                active_tenders = get_tenders(s)

                for tender in active_tenders:
                    tid = int(tender["tender_id"])
                    key = (
                        heat["heat_id"],
                        tid,
                    )

                    if key in seen:
                        continue

                    detected_mono = time.perf_counter()
                    seen.add(key)

                    tender_row, arrival_book = analyze_tender(
                        s,
                        tender,
                        heat["heat_id"],
                        tick,
                        period,
                    )

                    # Update heat counters in memory only. No disk write yet.
                    heat["tender_count"] += 1

                    if tender_row["decision"] == "ACCEPT":
                        heat["accept_count"] += 1
                    else:
                        heat["decline_count"] += 1

                    if (
                        tender_row["decision"] == "ACCEPT"
                        and tender_row["scored_profit"] is not None
                    ):
                        heat["scored_profit_sum"] += max(
                            0.0,
                            tender_row["scored_profit"],
                        )

                    execution_bundle = None
                    decline_result = None
                    live_intervention = False

                    if LIVE:
                        if tender_row["decision"] == "ACCEPT":
                            # HOT PATH FIRST. No console, SQLite, CSV, book
                            # refresh or position wait before the burst.
                            try:
                                execution_bundle = execute_accepted_tender_v4(
                                    s=s,
                                    tender=tender,
                                    tender_row=tender_row,
                                    arrival_book=arrival_book,
                                    tick=tick,
                                    detected_mono=detected_mono,
                                )
                            except Exception as exc:
                                print(
                                    "[CRITICAL] V4 execution path raised "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                print(
                                    "[CRITICAL] Attempting emergency flatten "
                                    "before stopping."
                                )
                                try:
                                    flatten_all_emergency(s)
                                finally:
                                    raise

                            summary = execution_bundle["summary"]
                            tender_row["execution_id"] = (
                                summary["execution_id"]
                            )
                            tender_row["actual_accept_success"] = int(
                                bool(summary["accept_success"])
                            )
                            live_intervention = True

                        else:
                            # Decline first; persistence/printing comes after.
                            try:
                                decline_result = decline_tender(
                                    s,
                                    tid,
                                )
                            except Exception as exc:
                                decline_result = {
                                    "success": False,
                                    "error": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                }

                    # From this point onward the latency-sensitive action is done.
                    print_analysis(tender_row)

                    persist_tender_research(
                        conn,
                        tender_row,
                        arrival_book,
                        tick,
                    )

                    register_tracker(
                        trackers,
                        tender_row,
                    )

                    if live_intervention:
                        mark_tracker_intervention(
                            trackers,
                            tid,
                        )

                    if execution_bundle is not None:
                        persist_execution_bundle(
                            conn,
                            execution_bundle,
                        )

                        summary = execution_bundle["summary"]

                        print(
                            f"[V4 LIVE] Tender {tid} | "
                            f"flat={summary['fully_flat']} | "
                            f"ordersFilled={summary['all_orders_filled']} | "
                            f"accept->first="
                            f"{fmt_price(summary['accept_to_first_order_ms'], 1)}ms | "
                            f"burst={fmt_price(summary['burst_submission_span_ms'], 1)}ms | "
                            f"execError="
                            f"{fmt_price(summary['execution_error_per_share'], 5)}/sh | "
                            f"realizedDelta="
                            f"{fmt_money(summary['security_realized_delta'])} | "
                            f"feeModelError="
                            f"{fmt_money(summary['fee_model_error'])} | "
                            f"rateHits="
                            f"{summary['rate_limit_hits_during_execution']}"
                        )

                    elif LIVE:
                        print(
                            f"[V4 LIVE] Tender {tid} declined: "
                            f"{decline_result}"
                        )

                    else:
                        print(
                            "[PAPER] No action sent. Tracking book "
                            "for 30 ticks."
                        )

                # Research work runs AFTER tender detection/action.
                sample_trackers(
                    s,
                    conn,
                    trackers,
                    tick,
                    heat["heat_id"],
                )

                last_tick = tick
                last_status = status
                last_period = period
                time.sleep(POLL_SECONDS)

    finally:
        if heat is not None:
            finalize_heat(
                conn,
                heat,
                (
                    last_tick
                    if last_tick is not None
                    else 0
                ),
                (
                    last_period
                    if last_period is not None
                    else 0
                ),
                "script_stop",
            )

        try:
            conn.commit()
            # Merge WAL into the main DB so a cleanly stopped database can be
            # copied/analyzed without leaving recent rows only in -wal.
            conn.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )
        except Exception:
            pass

        conn.close()


if __name__ == "__main__":
    signal.signal(
        signal.SIGINT,
        signal_handler,
    )

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped cleanly.")

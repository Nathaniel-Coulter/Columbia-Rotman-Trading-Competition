"""
LT3 tender-pricing / research / live-execution calibration engine
Rotman Interactive Trader
Version: 3.0

PURPOSE
-------
V3 preserves the V2 research logger and adds a conservative live-execution
calibration layer. It is designed to answer the questions that paper mode
cannot:

1) What execution VWAP do we actually receive after accepting a tender?
2) How far is realized execution from the untouched-book VWAP we predicted?
3) How quickly does RIT reflect the tender position and each market-order fill?
4) How does P&L move before tender -> after accept -> after each slice -> flat?
5) Does the stated $0.02/share fee behave like one fee or two in RIT?
6) Do CRZY/TAME and BUY/SELL tenders exhibit different execution error?
7) Which cent thresholds WOULD have accepted each tender, without changing the
   live policy used for the calibration run?
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

    lt3_training.sqlite       main research database (V2 + V3)
    lt3_tenders.csv           compact tender master table
    lt3_executions.csv        one row per accepted live tender
    lt3_orders.csv            one row per submitted hedge slice

The SQLite database is the authoritative dataset. CSVs are conveniences.

The schema is migration-safe: if you point V3 at the existing V2 database,
V3 adds its new tables/columns and preserves all historical V2 data.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# ============================================================================
# CONFIG
# ============================================================================

VERSION = "3.0"
PORT = 9999
API_KEY = os.getenv("RIT_API_KEY", "PASTE_YOUR_RIT_API_KEY_HERE")
BASE = f"http://localhost:{PORT}/v1"

# ------------------------- LIVE SWITCH --------------------------------------
# Start False. For the demo-server execution-calibration run, set:
#     LIVE = True
#     LIVE_CONFIRMATION = "YES_DEMO"
#
# This extra confirmation prevents an accidental switch to live order submission
# merely because one boolean was edited.

LIVE = True
LIVE_CONFIRMATION = "YES_DEMO"

#LIVE = False
#LIVE_CONFIRMATION = "NO"

# ------------------------- LOOP / CASE --------------------------------------
POLL_SECONDS = 0.20
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
# These DO NOT change live decisions. V3 logs whether each tender would pass
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
EXECUTION_MODE = "IMMEDIATE_MARKET"

# How long to wait for RIT to reflect tender position before beginning hedge.
TENDER_POSITION_TIMEOUT_SECONDS = 1.50

# How long to wait after each market-order slice for a position change / fill.
ORDER_FILL_TIMEOUT_SECONDS = 1.50

# Polling used only during an execution sequence. Fast enough to measure
# latency while being friendly to the local API.
EXECUTION_POLL_SECONDS = 0.02

# Tiny pause between completed slices. Set 0 to fire the next slice as soon as
# the prior slice is observed; 0.01 is a useful calibration default.
BETWEEN_SLICE_DELAY_SECONDS = 0.00

# Explicit pacing cushion: no more than ~9.5 order submissions/sec.
# This is separate from the generic HTTP 429 retry handler.
MIN_ORDER_SUBMIT_INTERVAL_SECONDS = 0.105
_last_order_submit_monotonic = 0.0

# Maximum hedge-loop iterations. An LT3 tender should need far fewer.
MAX_HEDGE_SLICES = 50

# If the remaining position is nonzero after a failed order/timeout, V3 retries
# based on the ACTUAL current position instead of assuming the prior order filled.
RETRY_ON_FILL_TIMEOUT = True

# ------------------------- PATHS --------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = LOG_DIR / "lt3_training.sqlite"
TENDER_CSV = LOG_DIR / "lt3_tenders.csv"
EXECUTION_CSV = LOG_DIR / "lt3_executions.csv"
ORDER_CSV = LOG_DIR / "lt3_orders.csv"

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
    # First Ctrl+C requests a graceful database close. A second Ctrl+C uses the
    # default handler if something is genuinely stuck.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True


def rit_session():
    s = requests.Session()
    s.headers.update({"X-API-Key": API_KEY})
    return s


def request(s, method, path, allow_404=False, **kwargs):
    """
    HTTP wrapper:
    - automatically respects common RIT 429 wait responses;
    - retries transient requests exceptions;
    - provides a clear 401 error;
    - optionally returns None for 404 (used by order-detail fallback logic).
    """
    url = BASE + path

    while not shutdown:
        try:
            r = s.request(method, url, timeout=2, **kwargs)
        except requests.RequestException:
            time.sleep(0.10)
            continue

        if r.status_code == 429:
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
    "predicted_arrival_unwind_vwap",
    "predicted_arrival_after_fee_edge",
    "predicted_arrival_profit_after_fee",
    "favorable_imbalance_full",
    "accept_latency_ms", "tender_position_latency_ms",
    "hedge_action", "hedge_requested_qty",
    "hedge_observed_filled_qty", "hedge_slice_count",
    "realized_hedge_vwap",
    "execution_error_per_share", "execution_error_dollars",
    "gross_block_profit", "profit_less_unwind_fee",
    "profit_less_two_2c_fees",
    "pnl_before", "pnl_after_accept", "pnl_after_flat", "rit_pnl_delta",
    "inferred_fee_error_one_side", "inferred_fee_error_two_sides",
    "final_position", "fully_flat", "failure_reason",
]

ORDER_CSV_FIELDS = [
    "run_id", "heat_id", "tender_id", "execution_id",
    "slice_no", "observed_at_utc", "ticker", "action",
    "requested_qty", "position_before", "position_after",
    "observed_filled_qty",
    "predicted_slice_vwap", "predicted_slice_worst",
    "predicted_slice_covered",
    "order_id", "submit_latency_ms", "fill_observation_latency_ms",
    "actual_vwap", "actual_price",
    "signed_price_improvement", "adverse_slippage_per_share",
    "status", "error",
]


def append_csv(path, fieldnames, row):
    new_file = not path.exists()
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

def accept_tender(s, tender):
    t0 = monotonic_ms()
    request_at = utc_now()

    r = request(
        s,
        "POST",
        f"/tenders/{int(tender['tender_id'])}",
        params={"price": float(tender["price"])},
    )

    response_at = utc_now()
    latency = monotonic_ms() - t0

    try:
        body = r.json()
    except Exception:
        body = {}

    success = bool(body.get("success", r.ok))

    return {
        "success": success,
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


def pace_order_submission():
    """Enforce a small cushion below 10 order submissions per second."""
    global _last_order_submit_monotonic

    now = time.perf_counter()
    wait = (
        MIN_ORDER_SUBMIT_INTERVAL_SECONDS
        - (now - _last_order_submit_monotonic)
    )
    if wait > 0:
        time.sleep(wait)

    _last_order_submit_monotonic = time.perf_counter()


def send_market_order(s, ticker, action, qty):
    pace_order_submission()

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


def wait_for_tender_position(
    s, ticker, position_before, expected_position, timeout_seconds
):
    """
    Wait for the accepted tender to appear in the portfolio. Returns:
        observed_position, latency_ms, reached_expected
    """
    start = monotonic_ms()
    deadline = time.perf_counter() + timeout_seconds
    latest = position_before

    while time.perf_counter() < deadline and not shutdown:
        latest = get_positions(s)[ticker]

        # Exact expected position is ideal. Any change is still useful and will
        # be handled safely by the hedge loop using the actual position.
        if latest == expected_position:
            return latest, monotonic_ms() - start, True

        if latest != position_before:
            return latest, monotonic_ms() - start, False

        time.sleep(EXECUTION_POLL_SECONDS)

    return latest, monotonic_ms() - start, latest == expected_position


def wait_for_position_target(
    s, ticker, position_before, action, requested_qty, timeout_seconds
):
    """
    Wait for the full requested market-order position change.

    A partial change is remembered, but we continue polling until either the
    requested quantity is reflected or timeout occurs. This prevents sending
    another slice merely because a partial fill appeared first.
    """
    signed_change = requested_qty if action == "BUY" else -requested_qty
    expected_position = position_before + signed_change

    start = monotonic_ms()
    deadline = time.perf_counter() + timeout_seconds
    latest = position_before

    while time.perf_counter() < deadline and not shutdown:
        latest = get_positions(s)[ticker]

        if latest == expected_position:
            return latest, monotonic_ms() - start, True

        time.sleep(EXECUTION_POLL_SECONDS)

    return latest, monotonic_ms() - start, latest == expected_position


def extract_reported_filled_qty(response, detail):
    merged = {}
    if isinstance(response, dict):
        merged.update(response)
    if isinstance(detail, dict):
        merged.update(detail)

    for name in (
        "quantity_filled", "filled", "filled_qty",
        "filled_quantity", "quantity_transacted"
    ):
        q = safe_int(merged.get(name))
        if q is not None:
            return max(0, q)
    return None


def extract_order_actuals(response, detail):
    """
    Pull actual fill metrics from either the submission response or a later
    order-detail query. Missing fields stay None; no invented values.
    """
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

    status = (
        merged.get("status")
        or merged.get("order_status")
        or merged.get("state")
    )

    return order_id, actual_vwap, actual_price, status


def signed_execution_metrics(action, predicted_vwap, actual_vwap):
    """
    Price improvement is positive when actual execution is BETTER than predicted.
    Adverse slippage is the opposite sign and positive when actual is WORSE.

    SELL hedge: higher actual price is better.
    BUY hedge : lower actual price is better.
    """
    if predicted_vwap is None or actual_vwap is None:
        return None, None

    if action == "SELL":
        improvement = actual_vwap - predicted_vwap
    else:
        improvement = predicted_vwap - actual_vwap

    return improvement, -improvement


def log_order_event(conn, row):
    conn.execute(
        """
        INSERT INTO order_events (
            run_id, heat_id, tender_id, execution_id,
            slice_no, observed_at_utc,
            ticker, action, requested_qty,
            position_before, position_after, observed_filled_qty,
            predicted_slice_vwap, predicted_slice_worst,
            predicted_slice_covered,
            order_id, submit_latency_ms, fill_observation_latency_ms,
            response_json, detail_json,
            actual_vwap, actual_price,
            signed_price_improvement, adverse_slippage_per_share,
            status, error
        ) VALUES (
            ?,?,?,?,
            ?,?,
            ?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,
            ?,?,
            ?,?,
            ?,?
        )
        """,
        (
            row["run_id"], row["heat_id"], row["tender_id"],
            row["execution_id"],
            row["slice_no"], row["observed_at_utc"],
            row["ticker"], row["action"], row["requested_qty"],
            row["position_before"], row["position_after"],
            row["observed_filled_qty"],
            row["predicted_slice_vwap"], row["predicted_slice_worst"],
            int(row["predicted_slice_covered"]),
            row["order_id"], row["submit_latency_ms"],
            row["fill_observation_latency_ms"],
            row["response_json"], row["detail_json"],
            row["actual_vwap"], row["actual_price"],
            row["signed_price_improvement"],
            row["adverse_slippage_per_share"],
            row["status"], row["error"],
        ),
    )
    conn.commit()
    append_csv(ORDER_CSV, ORDER_CSV_FIELDS, row)


def create_execution_session(conn, tender_row, execution_id, pnl_before):
    conn.execute(
        """
        INSERT INTO execution_sessions (
            execution_id, run_id, heat_id, tender_id,
            ticker, tender_action, quantity, tender_price,
            started_at_utc,
            predicted_arrival_unwind_vwap,
            predicted_arrival_after_fee_edge,
            predicted_arrival_profit_after_fee,
            favorable_imbalance_full,
            position_before_accept,
            hedge_action,
            pnl_before
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            execution_id, RUN_ID, tender_row["heat_id"],
            tender_row["tender_id"],
            tender_row["ticker"], tender_row["action"],
            tender_row["quantity"], tender_row["tender_price"],
            utc_now(),
            tender_row["unwind_vwap"],
            tender_row["fee_adjusted_edge_per_share"],
            tender_row["immediate_profit_after_fees"],
            tender_row["favorable_imbalance_full"],
            tender_row["actual_positions"].get(tender_row["ticker"], 0),
            tender_row["unwind_side"],
            pnl_before,
        ),
    )
    conn.commit()


def update_execution_accept(
    conn, execution_id, accept_result, pos_after, position_latency_ms,
    pnl_after_accept
):
    conn.execute(
        """
        UPDATE execution_sessions
        SET accept_request_at_utc = ?,
            accept_response_at_utc = ?,
            accept_latency_ms = ?,
            accept_success = ?,
            accept_response_json = ?,
            position_after_accept = ?,
            tender_position_latency_ms = ?,
            pnl_after_accept = ?
        WHERE execution_id = ?
        """,
        (
            accept_result["request_at"],
            accept_result["response_at"],
            accept_result["latency_ms"],
            int(accept_result["success"]),
            json_dumps(accept_result["body"]),
            pos_after,
            position_latency_ms,
            pnl_after_accept,
            execution_id,
        ),
    )
    conn.commit()


def finalize_execution_session(conn, summary):
    conn.execute(
        """
        UPDATE execution_sessions
        SET completed_at_utc = ?,
            hedge_requested_qty = ?,
            hedge_observed_filled_qty = ?,
            hedge_slice_count = ?,
            realized_hedge_vwap = ?,
            execution_error_per_share = ?,
            execution_error_dollars = ?,
            gross_block_profit = ?,
            profit_less_unwind_fee = ?,
            profit_less_two_2c_fees = ?,
            pnl_after_flat = ?,
            rit_pnl_delta = ?,
            inferred_fee_error_one_side = ?,
            inferred_fee_error_two_sides = ?,
            final_position = ?,
            fully_flat = ?,
            failure_reason = ?
        WHERE execution_id = ?
        """,
        (
            utc_now(),
            summary["hedge_requested_qty"],
            summary["hedge_observed_filled_qty"],
            summary["hedge_slice_count"],
            summary["realized_hedge_vwap"],
            summary["execution_error_per_share"],
            summary["execution_error_dollars"],
            summary["gross_block_profit"],
            summary["profit_less_unwind_fee"],
            summary["profit_less_two_2c_fees"],
            summary["pnl_after_flat"],
            summary["rit_pnl_delta"],
            summary["inferred_fee_error_one_side"],
            summary["inferred_fee_error_two_sides"],
            summary["final_position"],
            int(summary["fully_flat"]),
            summary["failure_reason"],
            summary["execution_id"],
        ),
    )
    conn.commit()
    append_csv(EXECUTION_CSV, EXECUTION_CSV_FIELDS, summary)


def execution_profit_metrics(
    tender_action, tender_price, quantity, realized_hedge_vwap
):
    if realized_hedge_vwap is None or quantity <= 0:
        return None, None, None

    if tender_action == "BUY":
        gross = (realized_hedge_vwap - tender_price) * quantity
    else:
        gross = (tender_price - realized_hedge_vwap) * quantity

    one_side = gross - MARKET_FEE_PER_SHARE * quantity

    # Explicit diagnostic for the ambiguity we want to test:
    # what if RIT effectively charges $0.02/share on BOTH the tender and hedge?
    two_2c = gross - 0.04 * quantity

    return gross, one_side, two_2c


def execute_accepted_tender(
    s, conn, tender, tender_row, tick
):
    """
    Conservative execution calibration:
      1) record P&L before;
      2) accept tender;
      3) wait for accepted position to appear;
      4) only then send max-size MARKET slices sequentially until flat;
      5) capture actual order responses, observed position deltas and P&L.

    No execution choice is influenced by imbalance in V3. Imbalance remains a
    feature for later V4 experimentation.
    """
    execution_id = (
        f"{RUN_ID}-{tender_row['heat_id']}-"
        f"T{tender_row['tender_id']}-{uuid.uuid4().hex[:6]}"
    )

    ticker = tender_row["ticker"]
    tender_action = tender_row["action"]
    qty = tender_row["quantity"]
    hedge_action = tender_row["unwind_side"]

    # Baseline portfolio/P&L immediately before accepting.
    pre_snap = insert_pnl_snapshot(
        conn, s, tender_row["heat_id"], tender_row["tender_id"],
        execution_id, tick, "before_accept"
    )
    pnl_before = best_available_pnl(pre_snap)

    create_execution_session(
        conn, tender_row, execution_id, pnl_before
    )
    update_tender_execution_link(
        conn, tender_row, execution_id, None
    )

    pos_before = pre_snap["positions"].get(ticker, 0)

    signed_tender = qty if tender_action == "BUY" else -qty
    expected_post_accept = pos_before + signed_tender

    # ------------------------- ACCEPT FIRST -----------------------------
    accept_result = accept_tender(s, tender)

    if not accept_result["success"]:
        update_tender_execution_link(
            conn, tender_row, execution_id, 0
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
            "predicted_arrival_unwind_vwap": tender_row["unwind_vwap"],
            "predicted_arrival_after_fee_edge":
                tender_row["fee_adjusted_edge_per_share"],
            "predicted_arrival_profit_after_fee":
                tender_row["immediate_profit_after_fees"],
            "favorable_imbalance_full":
                tender_row["favorable_imbalance_full"],
            "accept_latency_ms": accept_result["latency_ms"],
            "tender_position_latency_ms": None,
            "hedge_action": hedge_action,
            "hedge_requested_qty": 0,
            "hedge_observed_filled_qty": 0,
            "hedge_slice_count": 0,
            "realized_hedge_vwap": None,
            "execution_error_per_share": None,
            "execution_error_dollars": None,
            "gross_block_profit": None,
            "profit_less_unwind_fee": None,
            "profit_less_two_2c_fees": None,
            "pnl_before": pnl_before,
            "pnl_after_accept": None,
            "pnl_after_flat": None,
            "rit_pnl_delta": None,
            "inferred_fee_error_one_side": None,
            "inferred_fee_error_two_sides": None,
            "final_position": pos_before,
            "fully_flat": pos_before == 0,
            "failure_reason": "tender_accept_failed",
        }
        finalize_execution_session(conn, summary)
        return summary

    update_tender_execution_link(
        conn, tender_row, execution_id, 1
    )

    pos_after_accept, tender_pos_latency, exact_expected = (
        wait_for_tender_position(
            s, ticker, pos_before, expected_post_accept,
            TENDER_POSITION_TIMEOUT_SECONDS
        )
    )

    accept_snap = insert_pnl_snapshot(
        conn, s, tender_row["heat_id"], tender_row["tender_id"],
        execution_id, tick, "after_accept"
    )
    pnl_after_accept = best_available_pnl(accept_snap)

    update_execution_accept(
        conn, execution_id, accept_result, pos_after_accept,
        tender_pos_latency, pnl_after_accept
    )

    print(
        f"[LIVE] Accepted tender {tender_row['tender_id']} in "
        f"{accept_result['latency_ms']:.1f} ms | "
        f"{ticker} pos {pos_before:,} -> {pos_after_accept:,} "
        f"(expected {expected_post_accept:,})"
    )

    # ------------------------- HEDGE -----------------------------------
    weighted_actual_notional = 0.0
    actual_vwap_qty = 0
    hedge_requested_qty = 0
    observed_filled_total = 0
    slice_count = 0
    failure_reason = None

    for slice_no in range(1, MAX_HEDGE_SLICES + 1):
        current_positions = get_positions(s)
        pos_before_order = current_positions[ticker]

        if pos_before_order == 0:
            break

        # We flatten the ACTUAL position, not an assumed tender quantity.
        order_action = "SELL" if pos_before_order > 0 else "BUY"
        requested_qty = min(
            abs(pos_before_order),
            MAX_ORDER_SIZE[ticker],
        )

        # Capture the live book immediately before this specific slice.
        slice_book = get_book(s, ticker)
        slice_side = (
            slice_book["bids"] if order_action == "SELL"
            else slice_book["asks"]
        )
        predicted_slice_vwap, predicted_slice_worst, _, predicted_covered = (
            sweep_vwap(slice_side, requested_qty)
        )

        insert_pnl_snapshot(
            conn, s, tender_row["heat_id"], tender_row["tender_id"],
            execution_id, tick, f"before_order_{slice_no}"
        )

        response = {}
        detail = {}
        error = None
        submit_latency_ms = None
        fill_latency_ms = None
        pos_after_order = pos_before_order
        observed_filled_qty = 0
        order_id = None
        actual_vwap = None
        actual_price = None
        status = None

        try:
            response, submit_latency_ms = send_market_order(
                s, ticker, order_action, requested_qty
            )
            order_id = safe_int(response.get("order_id"))

            pos_after_order, fill_latency_ms, full_position_change = (
                wait_for_position_target(
                    s, ticker, pos_before_order, order_action,
                    requested_qty, ORDER_FILL_TIMEOUT_SECONDS
                )
            )

            position_delta_qty = abs(
                pos_after_order - pos_before_order
            )

            # Query the completed order record for fill VWAP/status.
            if order_id is not None:
                detail = get_order_detail(s, order_id)

            (
                order_id2,
                actual_vwap,
                actual_price,
                status,
            ) = extract_order_actuals(response, detail)

            reported_filled_qty = extract_reported_filled_qty(
                response, detail
            )

            # Position delta is the safest portfolio truth; the order record can
            # be more specific if RIT returns quantity_filled.
            if reported_filled_qty is not None:
                observed_filled_qty = max(
                    position_delta_qty, reported_filled_qty
                )
            else:
                observed_filled_qty = position_delta_qty

            if order_id is None:
                order_id = order_id2

            if not full_position_change:
                if observed_filled_qty == 0:
                    error = "position_did_not_change_before_timeout"
                else:
                    error = (
                        "partial_position_change_before_timeout:"
                        f"{observed_filled_qty}/{requested_qty}"
                    )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        improvement, adverse = signed_execution_metrics(
            order_action, predicted_slice_vwap, actual_vwap
        )

        order_row = {
            "run_id": RUN_ID,
            "heat_id": tender_row["heat_id"],
            "tender_id": tender_row["tender_id"],
            "execution_id": execution_id,

            "slice_no": slice_no,
            "observed_at_utc": utc_now(),

            "ticker": ticker,
            "action": order_action,
            "requested_qty": requested_qty,

            "position_before": pos_before_order,
            "position_after": pos_after_order,
            "observed_filled_qty": observed_filled_qty,

            "predicted_slice_vwap": predicted_slice_vwap,
            "predicted_slice_worst": predicted_slice_worst,
            "predicted_slice_covered": predicted_covered,

            "order_id": order_id,
            "submit_latency_ms": submit_latency_ms,
            "fill_observation_latency_ms": fill_latency_ms,

            "response_json": json_dumps(response),
            "detail_json": json_dumps(detail),

            "actual_vwap": actual_vwap,
            "actual_price": actual_price,

            "signed_price_improvement": improvement,
            "adverse_slippage_per_share": adverse,

            "status": status,
            "error": error,
        }

        log_order_event(conn, order_row)

        slice_count += 1
        hedge_requested_qty += requested_qty
        observed_filled_total += observed_filled_qty

        if actual_vwap is not None and observed_filled_qty > 0:
            weighted_actual_notional += (
                actual_vwap * observed_filled_qty
            )
            actual_vwap_qty += observed_filled_qty

        print(
            f"  [LIVE] {order_action} {requested_qty:,} {ticker} | "
            f"pred={fmt_price(predicted_slice_vwap, 5)} "
            f"actual={fmt_price(actual_vwap, 5)} | "
            f"pos {pos_before_order:,}->{pos_after_order:,} | "
            f"submit={fmt_price(submit_latency_ms, 1)}ms "
            f"fillObs={fmt_price(fill_latency_ms, 1)}ms"
        )

        insert_pnl_snapshot(
            conn, s, tender_row["heat_id"], tender_row["tender_id"],
            execution_id, tick, f"after_order_{slice_no}"
        )

        if error and observed_filled_qty == 0 and not RETRY_ON_FILL_TIMEOUT:
            failure_reason = error
            break

        if BETWEEN_SLICE_DELAY_SECONDS > 0:
            time.sleep(BETWEEN_SLICE_DELAY_SECONDS)
        # send_market_order() separately enforces MIN_ORDER_SUBMIT_INTERVAL_SECONDS.

    final_snap = insert_pnl_snapshot(
        conn, s, tender_row["heat_id"], tender_row["tender_id"],
        execution_id, tick, "after_flat"
    )

    final_position = final_snap["positions"].get(ticker, 0)
    fully_flat = final_position == 0
    pnl_after_flat = best_available_pnl(final_snap)

    if not fully_flat and failure_reason is None:
        failure_reason = (
            f"residual_position_{final_position}"
        )

    realized_hedge_vwap = (
        weighted_actual_notional / actual_vwap_qty
        if actual_vwap_qty > 0
        else None
    )

    # If RIT did not expose fill VWAP, preserve the session anyway. We do NOT
    # substitute book prices for realized prices.
    predicted_arrival = tender_row["unwind_vwap"]

    if (
        realized_hedge_vwap is not None
        and predicted_arrival is not None
    ):
        if hedge_action == "SELL":
            execution_error_per_share = (
                predicted_arrival - realized_hedge_vwap
            )
        else:
            execution_error_per_share = (
                realized_hedge_vwap - predicted_arrival
            )
        execution_error_dollars = (
            execution_error_per_share * actual_vwap_qty
        )
    else:
        execution_error_per_share = None
        execution_error_dollars = None

    gross, less_one_fee, less_two_2c = execution_profit_metrics(
        tender_action,
        tender_row["tender_price"],
        actual_vwap_qty,
        realized_hedge_vwap,
    )

    rit_pnl_delta = (
        None
        if pnl_before is None or pnl_after_flat is None
        else pnl_after_flat - pnl_before
    )

    inferred_fee_error_one_side = (
        None
        if rit_pnl_delta is None or less_one_fee is None
        else rit_pnl_delta - less_one_fee
    )

    inferred_fee_error_two_sides = (
        None
        if rit_pnl_delta is None or less_two_2c is None
        else rit_pnl_delta - less_two_2c
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

        "predicted_arrival_unwind_vwap":
            tender_row["unwind_vwap"],
        "predicted_arrival_after_fee_edge":
            tender_row["fee_adjusted_edge_per_share"],
        "predicted_arrival_profit_after_fee":
            tender_row["immediate_profit_after_fees"],
        "favorable_imbalance_full":
            tender_row["favorable_imbalance_full"],

        "accept_latency_ms": accept_result["latency_ms"],
        "tender_position_latency_ms": tender_pos_latency,

        "hedge_action": hedge_action,
        "hedge_requested_qty": hedge_requested_qty,
        "hedge_observed_filled_qty": observed_filled_total,
        "hedge_slice_count": slice_count,

        "realized_hedge_vwap": realized_hedge_vwap,
        "execution_error_per_share": execution_error_per_share,
        "execution_error_dollars": execution_error_dollars,

        "gross_block_profit": gross,
        "profit_less_unwind_fee": less_one_fee,
        "profit_less_two_2c_fees": less_two_2c,

        "pnl_before": pnl_before,
        "pnl_after_accept": pnl_after_accept,
        "pnl_after_flat": pnl_after_flat,
        "rit_pnl_delta": rit_pnl_delta,

        "inferred_fee_error_one_side":
            inferred_fee_error_one_side,
        "inferred_fee_error_two_sides":
            inferred_fee_error_two_sides,

        "final_position": final_position,
        "fully_flat": fully_flat,
        "failure_reason": failure_reason,
    }

    finalize_execution_session(conn, summary)

    print(
        f"[LIVE] Execution complete | "
        f"actual hedge VWAP={fmt_price(realized_hedge_vwap, 6)} | "
        f"arrival prediction={fmt_price(predicted_arrival, 6)} | "
        f"adverse error={fmt_price(execution_error_per_share, 5)}/sh | "
        f"RIT P&L delta={fmt_money(rit_pnl_delta)} | "
        f"flat={fully_flat}"
    )

    return summary


def flatten_ticker_emergency(s, ticker):
    """
    Emergency end-of-heat flatten for any residual position. Not part of the
    tender execution-calibration dataset.
    """
    max_size = MAX_ORDER_SIZE[ticker]

    for _ in range(MAX_HEDGE_SLICES):
        pos = get_positions(s)[ticker]
        if pos == 0:
            return

        action = "SELL" if pos > 0 else "BUY"
        qty = min(abs(pos), max_size)

        try:
            send_market_order(s, ticker, action, qty)
        except Exception:
            pass

        time.sleep(0.03)


def flatten_all_emergency(s):
    p = get_positions(s)
    for ticker, pos in p.items():
        if pos:
            print(
                f"[EMERGENCY] Hard flatten {ticker}: {pos:,}"
            )
            flatten_ticker_emergency(s, ticker)


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
# STARTUP VALIDATION
# ============================================================================

def validate_startup():
    if not API_KEY or API_KEY == "PASTE_YOUR_RIT_API_KEY_HERE":
        raise SystemExit(
            "Set API_KEY in this file or set PowerShell env var "
            "RIT_API_KEY first."
        )

    if LIVE and LIVE_CONFIRMATION != "YES_DEMO":
        raise SystemExit(
            "LIVE=True but LIVE_CONFIRMATION is not 'YES_DEMO'. "
            "For a demo execution-calibration run set both explicitly."
        )

    if EXECUTION_MODE != "IMMEDIATE_MARKET":
        raise SystemExit(
            "V3 supports only EXECUTION_MODE='IMMEDIATE_MARKET'. "
            "Passive execution is intentionally deferred until the live "
            "calibration data has been analyzed."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    validate_startup()
    conn = open_db()

    print("Connected to LT3 V3 research/execution engine.")
    print(f"RUN_ID={RUN_ID}")
    print(f"LIVE={LIVE}  BASE={BASE}")
    print(f"Database:      {DB_FILE}")
    print(f"Tender CSV:    {TENDER_CSV}")
    print(f"Execution CSV: {EXECUTION_CSV}")
    print(f"Order CSV:     {ORDER_CSV}")
    print(
        f"Current policy: after-fee edge minus "
        f"${SAFETY_BUFFER_PER_SHARE:.2f} safety >= "
        f"${MIN_NET_EDGE_PER_SHARE:.2f}/sh and scored >= "
        f"${MIN_EXPECTED_PROFIT:,.0f}."
    )
    print(
        f"Counterfactual grid: {len(AFTER_FEE_EDGE_THRESHOLDS)} "
        f"edge thresholds x {len(SAFETY_BUFFER_GRID)} safety buffers."
    )
    print(
        f"Follow-ups: every {FOLLOWUP_EVERY_TICKS} tick(s), "
        f"{FOLLOWUP_TICKS} ticks, {BOOK_LEVELS_TO_STORE} levels/side."
    )
    print(
        f"Order pacing: >= {MIN_ORDER_SUBMIT_INTERVAL_SECONDS:.3f}s "
        f"between MARKET order submissions."
    )

    if LIVE:
        print(
            "\n*** LIVE DEMO CALIBRATION ENABLED ***\n"
            "Qualifying tenders WILL be accepted and immediately hedged "
            "with sequential max-size MARKET orders.\n"
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
                period = int(case.get("period", 0) or 0)

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
                    time.sleep(0.20)
                    continue

                if heat is None:
                    heat_seq += 1
                    heat = new_heat(conn, heat_seq, tick, period)

                # Post-tender research observations.
                sample_trackers(
                    s, conn, trackers, tick, heat["heat_id"]
                )

                # End-of-period backstop. In paper mode this does nothing.
                if tick >= HARD_FLATTEN_TICK:
                    if LIVE:
                        flatten_all_emergency(s)
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
                    append_csv(
                        TENDER_CSV, TENDER_CSV_FIELDS, tender_row
                    )

                    insert_policy_grid(
                        conn, tender_row, tender_row["_limit_ok"]
                    )

                    # Clean untouched-book arrival snapshot.
                    arrival_followup = followup_row(
                        tender_row,
                        arrival_book,
                        tick,
                        live_intervention=False,
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
                        if tender_row["decision"] == "ACCEPT":
                            heat["scored_profit_sum"] += max(
                                0.0, tender_row["scored_profit"]
                            )

                    if not LIVE:
                        print(
                            "[PAPER] No action sent. Tracking book for "
                            "30 ticks."
                        )
                        continue

                    # ------------------------- LIVE ----------------------
                    if tender_row["decision"] == "ACCEPT":
                        # Mark subsequent followups as market-impacted by us.
                        mark_tracker_intervention(trackers, tid)

                        summary = execute_accepted_tender(
                            s, conn, tender, tender_row, tick
                        )

                        print(
                            f"[LIVE SUMMARY] Tender {tid}: "
                            f"flat={summary['fully_flat']} | "
                            f"execError={fmt_price(summary['execution_error_per_share'], 5)}/sh | "
                            f"P&L delta={fmt_money(summary['rit_pnl_delta'])}"
                        )

                    else:
                        result = decline_tender(s, tid)
                        print(
                            f"[LIVE] Tender {tid} declined: {result}"
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

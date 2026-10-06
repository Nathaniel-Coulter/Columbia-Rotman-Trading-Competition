# fullsample_g1_trade_frequency.py
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

import polars as pl


# ============================================================
# Paths
# ============================================================

FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\fullsample_g1_trade_frequency"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENT_CANDIDATES = OUTPUT_DIR / "g1_fullsample_entry_candidates.csv"
OUT_ACTUAL_ENTRIES = OUTPUT_DIR / "g1_fullsample_actual_entries.csv"
OUT_DAILY = OUTPUT_DIR / "g1_fullsample_daily_trade_counts.csv"
OUT_WEEKLY = OUTPUT_DIR / "g1_fullsample_weekly_trade_counts.csv"
OUT_SUMMARY = OUTPUT_DIR / "g1_fullsample_trade_frequency_summary.csv"


# ============================================================
# Strategy definition: G1_t4_s6_t900
# ============================================================

HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "upper"

# Parent rule = UNION_A2_C2
RULE_A2 = [
    ("depth_near_ask", "between", 4.0, 128.0),
    ("cancel_rate_ask", "between", 6.6, 427.0),
]

RULE_C2 = [
    ("cancel_rate_ask", "between", 6.6, 427.0),
    ("bid_depth_curvature", "between", -6.625, 11.5),
]

# Gate G1 = ALL(best_ask_queue_norm high, aggressive_buy_volume low)
FINAL_CONFIRMATIONS: Dict[str, Tuple[str, float, float]] = {
    "best_ask_queue_norm": ("high_bin", 0.9811400683599983, 16.909106355664214),
    "aggressive_buy_volume": ("low_bin", 70.0, 507.0),
}

GATE_NAME = "G1_ALL2_bestask_aggbuy"

ENTRY_ADVERSE_PTS = 1.0
TARGET_PTS = 4.0
STOP_PTS = 6.0
TIME_EXIT_S = 900
ONE_TRADE_AT_A_TIME = True


# ============================================================
# Helpers
# ============================================================

def iter_day_parquets(root: Path):
    for ydir in sorted(root.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            for ddir in sorted(mdir.glob("day=*")):
                p = ddir / "part-000.parquet"
                if not p.exists():
                    continue
                year = int(ydir.name.split("=")[1])
                month = int(mdir.name.split("=")[1])
                day = int(ddir.name.split("=")[1])
                yield year, month, day, p


def build_day_to_path(root: Path) -> Dict[str, Path]:
    out = {}
    for y, m, d, p in iter_day_parquets(root):
        out[f"{y:04d}-{m:02d}-{d:02d}"] = p
    return out


def load_trade_day(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    required = {"ts", "type", "price"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required trade cols: {missing}")

    return (
        df.filter(
            (pl.col("type") == 2) &
            pl.col("price").is_not_null()
        )
        .select(["ts", "price"])
        .sort("ts")
    )


def passes_condition(v: Any, op: str, lo: float, hi: float | None = None) -> bool:
    if v is None:
        return False
    try:
        x = float(v)
    except Exception:
        return False

    if op == "between":
        assert hi is not None
        return lo <= x <= hi

    raise ValueError(f"Unsupported op: {op}")


def event_passes_rule(row: Dict[str, Any], conds: List[Tuple[str, str, float, float]]) -> bool:
    for feat, op, lo, hi in conds:
        if not passes_condition(row.get(feat), op, lo, hi):
            return False
    return True


def confirmation_pass(row: Dict[str, Any], feat: str, lo: float, hi: float) -> bool:
    return passes_condition(row.get(feat), "between", lo, hi)


def g1_gate_pass(row: Dict[str, Any]) -> bool:
    return all(
        confirmation_pass(row, feat, lo, hi)
        for feat, (_label, lo, hi) in FINAL_CONFIRMATIONS.items()
    )


def safe_float(x):
    return None if x is None else float(x)


def safe_int(x):
    return None if x is None else int(x)


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(FULL_TICK_ROOT)

    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    df = pl.read_parquet(FEATURES_PATH).sort(["date_ymd", "ts"])
    df = df.filter(
        pl.col("band_k").is_in(BAND_FILTER) &
        (pl.col("side") == SIDE_FILTER)
    )

    rows = df.to_dicts()

    print("\n--------------------------------------")
    print("Full-Sample G1 Trade Frequency")
    print("--------------------------------------")
    print(f"Rows analyzed          = {len(rows):,}")
    print(f"BAND_FILTER            = {BAND_FILTER}")
    print(f"SIDE_FILTER            = {SIDE_FILTER}")
    print(f"PARENT_RULE            = UNION_A2_C2")
    print(f"GATE_NAME              = {GATE_NAME}")
    print(f"ENTRY_ADVERSE_PTS      = {ENTRY_ADVERSE_PTS}")
    print(f"TARGET_PTS             = {TARGET_PTS}")
    print(f"STOP_PTS               = {STOP_PTS}")
    print(f"TIME_EXIT_S            = {TIME_EXIT_S}")
    print(f"ONE_TRADE_AT_A_TIME    = {ONE_TRADE_AT_A_TIME}")

    # --------------------------------------------------------
    # 1. Parent + gate filter on feature rows
    # --------------------------------------------------------
    candidate_rows = []
    for r in rows:
        parent_pass = event_passes_rule(r, RULE_A2) or event_passes_rule(r, RULE_C2)
        if not parent_pass:
            continue
        if not g1_gate_pass(r):
            continue
        candidate_rows.append(r)

    if not candidate_rows:
        raise ValueError("No rows passed parent rule + G1 gate.")

    # Keep chronological order
    candidate_rows = sorted(candidate_rows, key=lambda r: (r["date_ymd"], r["ts"]))

    print(f"Candidate events       = {len(candidate_rows):,}")

    # --------------------------------------------------------
    # 2. Build entry candidates from raw path
    #    We simulate exit only to enforce one-trade-at-a-time.
    # --------------------------------------------------------
    day_cache: Dict[str, Tuple[List[Any], List[float]]] = {}

    def get_day_series(day_ymd: str) -> Tuple[List[Any], List[float]]:
        if day_ymd in day_cache:
            return day_cache[day_ymd]
        p = day_to_path.get(day_ymd)
        if p is None or not p.exists():
            raise FileNotFoundError(f"Missing parquet for day {day_ymd}")
        tr = load_trade_day(p)
        ts_list = tr["ts"].to_list()
        px_list = [float(x) for x in tr["price"].to_list()]
        day_cache[day_ymd] = (ts_list, px_list)
        return ts_list, px_list

    entry_candidate_records: List[Dict[str, Any]] = []
    actual_entry_records: List[Dict[str, Any]] = []

    next_available_ts = None

    for r in candidate_rows:
        day_ymd = r["date_ymd"]
        event_ts = r["ts"]
        entry_price_trigger = r.get("entry_price")

        if event_ts is None or entry_price_trigger is None:
            continue

        # one-trade-at-a-time filter
        if ONE_TRADE_AT_A_TIME and next_available_ts is not None and event_ts < next_available_ts:
            continue

        entry_price_trigger = float(entry_price_trigger)

        ts_list, px_list = get_day_series(day_ymd)

        j0 = bisect_right(ts_list, event_ts) - 1
        if j0 < 0:
            continue

        entry_idx = None
        entry_ts = None
        entry_px = None
        horizon_end_idx = j0

        # find refined entry after adverse move
        for j in range(j0, len(ts_list)):
            dt_s = (ts_list[j] - event_ts).total_seconds()
            if dt_s > HORIZON_S:
                break
            horizon_end_idx = j

            px = px_list[j]
            direction = -1.0  # upper-band short reversion
            signed_move_from_event = (px - entry_price_trigger) * direction
            adverse_move = -signed_move_from_event

            if adverse_move >= ENTRY_ADVERSE_PTS:
                entry_idx = j
                entry_ts = ts_list[j]
                entry_px = px
                break

        if entry_idx is None:
            continue

        delay_s = (entry_ts - event_ts).total_seconds()

        entry_candidate_records.append({
            "event_id": r.get("event_id"),
            "date_ymd": day_ymd,
            "event_ts": event_ts,
            "entry_ts": entry_ts,
            "entry_delay_s": delay_s,
            "entry_price_trigger": entry_price_trigger,
            "entry_price_refined": entry_px,
        })

        # ----------------------------------------------------
        # exit simulation only to know when strategy is free again
        # ----------------------------------------------------
        exit_ts = None
        exit_reason = "time"
        end_idx = horizon_end_idx

        target_px = entry_px - TARGET_PTS
        stop_px = entry_px + STOP_PTS

        for j in range(entry_idx, len(ts_list)):
            dt_s = (ts_list[j] - entry_ts).total_seconds()
            if dt_s > TIME_EXIT_S:
                break

            end_idx = j
            px = px_list[j]

            # short logic
            if px <= target_px:
                exit_ts = ts_list[j]
                exit_reason = "target"
                break
            if px >= stop_px:
                exit_ts = ts_list[j]
                exit_reason = "stop"
                break

        if exit_ts is None:
            exit_ts = ts_list[end_idx]
            exit_reason = "time"

        next_available_ts = exit_ts

        actual_entry_records.append({
            "event_id": r.get("event_id"),
            "date_ymd": day_ymd,
            "event_ts": event_ts,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "exit_reason_for_lockout_only": exit_reason,
            "entry_delay_s": delay_s,
            "entry_price_trigger": entry_price_trigger,
            "entry_price_refined": entry_px,
        })

    if not actual_entry_records:
        raise ValueError("No actual entries were produced.")

    candidate_df = pl.from_dicts(entry_candidate_records).sort("entry_ts")
    actual_df = pl.from_dicts(actual_entry_records).sort("entry_ts")

    candidate_df.write_csv(OUT_EVENT_CANDIDATES)
    actual_df.write_csv(OUT_ACTUAL_ENTRIES)

    # --------------------------------------------------------
    # 3. Daily counts
    # --------------------------------------------------------
    actual_df = actual_df.with_columns(
        pl.col("date_ymd").cast(pl.Utf8).alias("trade_day")
    )

    daily_active = (
        actual_df.group_by("trade_day")
        .agg(pl.len().alias("n_trades"))
        .sort("trade_day")
        .with_columns(
            pl.col("trade_day").str.strptime(pl.Date, "%Y-%m-%d").alias("trade_date")
        )
        .with_columns([
            pl.col("trade_date").dt.strftime("%G").cast(pl.Int32).alias("iso_year"),
            pl.col("trade_date").dt.week().alias("iso_week"),
            pl.col("trade_date").dt.weekday().alias("weekday_num"),
        ])
        .select([
            "trade_day",
            "trade_date",
            "iso_year",
            "iso_week",
            "weekday_num",
            "n_trades",
        ])
    )

    # full trading-day universe from feature file
    all_days_df = (
        df.select(pl.col("date_ymd").cast(pl.Utf8).alias("trade_day"))
        .unique()
        .sort("trade_day")
        .with_columns(
            pl.col("trade_day").str.strptime(pl.Date, "%Y-%m-%d").alias("trade_date")
        )
        .with_columns([
            pl.col("trade_date").dt.strftime("%G").cast(pl.Int32).alias("iso_year"),
            pl.col("trade_date").dt.week().alias("iso_week"),
            pl.col("trade_date").dt.weekday().alias("weekday_num"),
        ])
    )

    daily_full = (
        all_days_df.join(
            daily_active.select(["trade_day", "n_trades"]),
            on="trade_day",
            how="left",
        )
        .with_columns(
            pl.col("n_trades").fill_null(0)
        )
        .select([
            "trade_day",
            "trade_date",
            "iso_year",
            "iso_week",
            "weekday_num",
            "n_trades",
        ])
        .sort("trade_day")
    )

    daily_full.write_csv(OUT_DAILY)

    # --------------------------------------------------------
    # 4. Weekly counts
    # --------------------------------------------------------
    weekly_full = (
        daily_full.group_by(["iso_year", "iso_week"])
        .agg([
            pl.len().alias("n_trading_days_total"),
            (pl.col("n_trades") > 0).sum().alias("n_trading_days_with_trades"),
            pl.col("n_trades").sum().alias("n_trades_week"),
            pl.col("trade_day").min().alias("week_start_day"),
            pl.col("trade_day").max().alias("week_end_day"),
        ])
        .sort(["iso_year", "iso_week"])
    )

    weekly_full.write_csv(OUT_WEEKLY)

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------
    total_trades = actual_df.height
    active_days = daily_active.height
    all_days = daily_full.height
    active_weeks = weekly_full.filter(pl.col("n_trades_week") > 0).height
    all_weeks = weekly_full.height

    summary = pl.DataFrame([{
        "strategy_name": "G1_t4_s6_t900",
        "gate_name": GATE_NAME,
        "total_actual_entries": total_trades,

        "n_candidate_events_before_lockout": len(candidate_rows),
        "n_candidate_events_after_lockout": candidate_df.height,

        "n_active_trade_days": active_days,
        "n_all_trading_days": all_days,

        "n_active_trade_weeks": active_weeks,
        "n_all_trading_weeks": all_weeks,

        # active-day stats
        "avg_trades_per_active_day": safe_float(daily_active["n_trades"].mean()),
        "median_trades_per_active_day": safe_float(daily_active["n_trades"].median()),
        "min_trades_per_active_day": safe_int(daily_active["n_trades"].min()),
        "max_trades_per_active_day": safe_int(daily_active["n_trades"].max()),

        # full-day stats
        "avg_trades_per_all_trading_day": safe_float(daily_full["n_trades"].mean()),
        "median_trades_per_all_trading_day": safe_float(daily_full["n_trades"].median()),
        "min_trades_per_all_trading_day": safe_int(daily_full["n_trades"].min()),
        "max_trades_per_all_trading_day": safe_int(daily_full["n_trades"].max()),

        # active-week stats
        "avg_trades_per_active_week": safe_float(
            weekly_full.filter(pl.col("n_trades_week") > 0)["n_trades_week"].mean()
        ),
        "median_trades_per_active_week": safe_float(
            weekly_full.filter(pl.col("n_trades_week") > 0)["n_trades_week"].median()
        ),

        # full-week stats
        "avg_trades_per_all_trading_week": safe_float(weekly_full["n_trades_week"].mean()),
        "median_trades_per_all_trading_week": safe_float(weekly_full["n_trades_week"].median()),
        "min_trades_per_all_trading_week": safe_int(weekly_full["n_trades_week"].min()),
        "max_trades_per_all_trading_week": safe_int(weekly_full["n_trades_week"].max()),

        "avg_trade_days_per_week": safe_float(weekly_full["n_trading_days_with_trades"].mean()),
        "median_trade_days_per_week": safe_float(weekly_full["n_trading_days_with_trades"].median()),
    }])

    summary.write_csv(OUT_SUMMARY)

    print("\nDone.")
    print(f"Wrote: {OUT_EVENT_CANDIDATES}")
    print(f"Wrote: {OUT_ACTUAL_ENTRIES}")
    print(f"Wrote: {OUT_DAILY}")
    print(f"Wrote: {OUT_WEEKLY}")
    print(f"Wrote: {OUT_SUMMARY}")

    print("\nSummary row:")
    print(summary)

    print("\nFirst 20 weekly rows:")
    print(weekly_full.head(20))


if __name__ == "__main__":
    main()
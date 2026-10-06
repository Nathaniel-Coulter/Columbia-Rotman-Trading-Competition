# trade_frequency_summary.py
from __future__ import annotations

from pathlib import Path
import polars as pl


# ============================================================
# Paths
# ============================================================

TRADE_LEVEL_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_trade_level.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\trade_frequency_summary"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILTERED = OUTPUT_DIR / "g2_t5_trade_subset.csv"
OUT_DAILY = OUTPUT_DIR / "g2_t5_daily_trade_counts.csv"
OUT_WEEKLY = OUTPUT_DIR / "g2_t5_weekly_trade_counts.csv"
OUT_SUMMARY = OUTPUT_DIR / "g2_t5_trade_frequency_summary.csv"


# ============================================================
# Strategy definition
# ============================================================

GATE_NAME = "G2_VOTE2OF3_curv_cancel_spread"
TARGET_PTS = 5.0
STOP_PTS = 8.0
TIME_EXIT_S = 600


# ============================================================
# Helpers
# ============================================================

def safe_float(x):
    return None if x is None else float(x)


def safe_int(x):
    return None if x is None else int(x)


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not TRADE_LEVEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRADE_LEVEL_PATH}")

    df = pl.read_csv(
        TRADE_LEVEL_PATH,
        try_parse_dates=True,
        infer_schema_length=10000,
    )

    required_cols = [
        "gate_name",
        "target_pts",
        "stop_pts",
        "time_exit_s",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------------------------------------------
    # derive date column
    # --------------------------------------------------------
    if "date_ymd" in df.columns:
        df = df.with_columns(
            pl.col("date_ymd").cast(pl.Utf8).alias("trade_day")
        )
    elif "entry_ts" in df.columns:
        if df.schema["entry_ts"] == pl.Utf8:
            df = df.with_columns(
                pl.col("entry_ts").str.slice(0, 10).alias("trade_day")
            )
        else:
            df = df.with_columns(
                pl.col("entry_ts").dt.strftime("%Y-%m-%d").alias("trade_day")
            )
    elif "ts" in df.columns:
        if df.schema["ts"] == pl.Utf8:
            df = df.with_columns(
                pl.col("ts").str.slice(0, 10).alias("trade_day")
            )
        else:
            df = df.with_columns(
                pl.col("ts").dt.strftime("%Y-%m-%d").alias("trade_day")
            )
    else:
        raise ValueError("Could not derive trade_day. Need one of: date_ymd, entry_ts, ts")

    # --------------------------------------------------------
    # filter to G2_t5_s8_t600
    # --------------------------------------------------------
    g2 = df.filter(
        (pl.col("gate_name") == GATE_NAME) &
        (pl.col("target_pts") == TARGET_PTS) &
        (pl.col("stop_pts") == STOP_PTS) &
        (pl.col("time_exit_s") == TIME_EXIT_S)
    )

    if g2.height == 0:
        raise ValueError("No trades matched G2_t5_s8_t600")

    g2.write_csv(OUT_FILTERED)

    # --------------------------------------------------------
    # daily counts
    # --------------------------------------------------------
    daily = (
        g2.group_by("trade_day")
        .agg(pl.len().alias("n_trades"))
        .sort("trade_day")
        .with_columns(
            pl.col("trade_day").str.strptime(pl.Date, "%Y-%m-%d").alias("trade_date")
        )
        .with_columns([
            pl.col("trade_date").dt.year().alias("year"),
            pl.col("trade_date").dt.week().alias("iso_week"),
            pl.col("trade_date").dt.weekday().alias("weekday_num"),
        ])
        .select([
            "trade_day",
            "trade_date",
            "year",
            "iso_week",
            "weekday_num",
            "n_trades",
        ])
    )

    daily.write_csv(OUT_DAILY)

    # --------------------------------------------------------
    # weekly counts
    # --------------------------------------------------------
    weekly = (
        daily.group_by(["year", "iso_week"])
        .agg([
            pl.len().alias("n_trading_days_with_trades"),
            pl.col("n_trades").sum().alias("n_trades_week"),
            pl.col("trade_day").min().alias("week_start_day"),
            pl.col("trade_day").max().alias("week_end_day"),
        ])
        .sort(["year", "iso_week"])
    )

    weekly.write_csv(OUT_WEEKLY)

    # --------------------------------------------------------
    # summary stats
    # --------------------------------------------------------
    total_trades = g2.height
    n_days = daily.height
    n_weeks = weekly.height

    summary = pl.DataFrame([{
        "strategy_name": "G2_t5_s8_t600",
        "gate_name": GATE_NAME,
        "target_pts": TARGET_PTS,
        "stop_pts": STOP_PTS,
        "time_exit_s": TIME_EXIT_S,
        "total_trades": total_trades,
        "n_trade_days": n_days,
        "n_trade_weeks": n_weeks,

        "avg_trades_per_day": safe_float(daily["n_trades"].mean()),
        "median_trades_per_day": safe_float(daily["n_trades"].median()),
        "min_trades_per_day": safe_int(daily["n_trades"].min()),
        "max_trades_per_day": safe_int(daily["n_trades"].max()),

        "avg_trades_per_week": safe_float(weekly["n_trades_week"].mean()),
        "median_trades_per_week": safe_float(weekly["n_trades_week"].median()),
        "min_trades_per_week": safe_int(weekly["n_trades_week"].min()),
        "max_trades_per_week": safe_int(weekly["n_trades_week"].max()),

        "avg_trade_days_per_week": safe_float(weekly["n_trading_days_with_trades"].mean()),
        "median_trade_days_per_week": safe_float(weekly["n_trading_days_with_trades"].median()),
    }])

    summary.write_csv(OUT_SUMMARY)

    print("\n--------------------------------------")
    print("Trade Frequency Summary")
    print("--------------------------------------")
    print("Strategy = G2_t5_s8_t600")
    print(f"Total trades         = {total_trades}")
    print(f"Trade days           = {n_days}")
    print(f"Trade weeks          = {n_weeks}")
    print(f"Avg trades / day     = {safe_float(daily['n_trades'].mean()):.4f}")
    print(f"Median trades / day  = {safe_float(daily['n_trades'].median()):.4f}")
    print(f"Avg trades / week    = {safe_float(weekly['n_trades_week'].mean()):.4f}")
    print(f"Median trades / week = {safe_float(weekly['n_trades_week'].median()):.4f}")

    print("\nWrote:")
    print(OUT_FILTERED)
    print(OUT_DAILY)
    print(OUT_WEEKLY)
    print(OUT_SUMMARY)

    print("\nSummary row:")
    print(summary)

    print("\nFirst 20 weekly rows:")
    print(weekly.head(20))


if __name__ == "__main__":
    main()
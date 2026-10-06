# trade_level_exit_diagnostics.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import math

import polars as pl

# ============================================================
# Paths
# ============================================================
TRADE_LEVEL_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_trade_level.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\trade_level_exit_diagnostics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILTERED_TRADES = OUTPUT_DIR / "diagnostic_trade_subset.csv"
OUT_FOLD_SUMMARY = OUTPUT_DIR / "diagnostic_fold_summary.csv"
OUT_STRATEGY_SUMMARY = OUTPUT_DIR / "diagnostic_strategy_summary.csv"

# ============================================================
# Strategy shortlist
# Edit this freely
# ============================================================
STRATEGY_SHORTLIST: List[Dict[str, Any]] = [
    {
        "strategy_name": "G1_t10_s8_t600",
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "target_pts": 10.0,
        "stop_pts": 8.0,
        "time_exit_s": 600,
    },
    {
        "strategy_name": "G1_t4_s6_t900",
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "target_pts": 4.0,
        "stop_pts": 6.0,
        "time_exit_s": 900,
    },
    {
        "strategy_name": "G1_t4_s7_t900",
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "target_pts": 4.0,
        "stop_pts": 7.0,
        "time_exit_s": 900,
    },
    {
        "strategy_name": "G1_t4_s8_t900",
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "target_pts": 4.0,
        "stop_pts": 8.0,
        "time_exit_s": 900,
    },
    {
        "strategy_name": "G1_t5_s8_t900",
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "target_pts": 5.0,
        "stop_pts": 8.0,
        "time_exit_s": 900,
    },
]
# minimum count just to avoid over-reading tiny buckets
MIN_TRADES_TOTAL = 20

# ============================================================
# Helpers
# ============================================================
def pct(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den

def safe_quantile(s: pl.Series, q: float) -> float:
    if len(s) == 0:
        return math.nan
    x = s.quantile(q)
    return float(x) if x is not None else math.nan

def safe_mean(df: pl.DataFrame, col: str) -> float:
    if df.height == 0:
        return math.nan
    x = df[col].mean()
    return float(x) if x is not None else math.nan

def safe_std(df: pl.DataFrame, col: str) -> float:
    if df.height <= 1:
        return 0.0
    x = df[col].std()
    return float(x) if x is not None else 0.0


def top_k_share_of_total_profit(pnls: List[float], k: int) -> float:
    wins = sorted([x for x in pnls if x > 0], reverse=True)
    total_profit = sum(wins)
    if total_profit <= 0:
        return math.nan
    return sum(wins[:k]) / total_profit


def bottom_k_share_of_total_loss(pnls: List[float], k: int) -> float:
    losses = sorted([x for x in pnls if x < 0])  # most negative first
    gross_loss = abs(sum(losses))
    if gross_loss <= 0:
        return math.nan
    return abs(sum(losses[:k])) / gross_loss


def summarize_trade_block(df: pl.DataFrame) -> Dict[str, Any]:
    n = df.height
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": math.nan,
            "avg_pnl_pts": math.nan,
            "median_pnl_pts": math.nan,
            "sum_pnl_pts": math.nan,
            "std_pnl_pts": math.nan,
            "avg_mfe_pts": math.nan,
            "avg_mae_pts": math.nan,
            "avg_end_ret_pts": math.nan,
            "median_hold_s": math.nan,
            "avg_hold_s": math.nan,
            "target_exit_rate": math.nan,
            "stop_exit_rate": math.nan,
            "time_exit_rate": math.nan,
            "pnl_p10": math.nan,
            "pnl_p25": math.nan,
            "pnl_p50": math.nan,
            "pnl_p75": math.nan,
            "pnl_p90": math.nan,
            "mfe_p50": math.nan,
            "mfe_p90": math.nan,
            "mae_p50": math.nan,
            "mae_p90": math.nan,
            "top1_profit_share": math.nan,
            "top3_profit_share": math.nan,
            "top5_profit_share": math.nan,
            "worst1_loss_share": math.nan,
            "worst3_loss_share": math.nan,
            "worst5_loss_share": math.nan,
            "profit_factor_proxy": math.nan,
        }

    pnl = df["pnl_pts"]
    mfe = df["mfe_pts"]
    mae = df["mae_pts"]
    hold = df["hold_s"]

    pnl_list = [float(x) for x in pnl.to_list()]
    wins = [x for x in pnl_list if x > 0]
    losses = [x for x in pnl_list if x < 0]

    total_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor_proxy = total_profit / gross_loss if gross_loss > 0 else math.nan

    exit_reason_counts = (
        df.group_by("exit_reason")
        .agg(pl.len().alias("n"))
        .to_dicts()
    )
    exit_reason_map = {r["exit_reason"]: int(r["n"]) for r in exit_reason_counts}

    return {
        "n_trades": n,
        "win_rate": float(df.select((pl.col("pnl_pts") > 0).mean()).item()),
        "avg_pnl_pts": float(pnl.mean()),
        "median_pnl_pts": float(pnl.median()),
        "sum_pnl_pts": float(pnl.sum()),
        "std_pnl_pts": float(pnl.std()) if n > 1 and pnl.std() is not None else 0.0,
        "avg_mfe_pts": float(mfe.mean()),
        "avg_mae_pts": float(mae.mean()),
        "avg_end_ret_pts": float(df["end_ret_pts"].mean()),
        "median_hold_s": float(hold.median()),
        "avg_hold_s": float(hold.mean()),
        "target_exit_rate": pct(exit_reason_map.get("target", 0), n),
        "stop_exit_rate": pct(exit_reason_map.get("stop", 0), n),
        "time_exit_rate": pct(exit_reason_map.get("time", 0), n),
        "pnl_p10": safe_quantile(pnl, 0.10),
        "pnl_p25": safe_quantile(pnl, 0.25),
        "pnl_p50": safe_quantile(pnl, 0.50),
        "pnl_p75": safe_quantile(pnl, 0.75),
        "pnl_p90": safe_quantile(pnl, 0.90),
        "mfe_p50": safe_quantile(mfe, 0.50),
        "mfe_p90": safe_quantile(mfe, 0.90),
        "mae_p50": safe_quantile(mae, 0.50),
        "mae_p90": safe_quantile(mae, 0.90),
        "top1_profit_share": top_k_share_of_total_profit(pnl_list, 1),
        "top3_profit_share": top_k_share_of_total_profit(pnl_list, 3),
        "top5_profit_share": top_k_share_of_total_profit(pnl_list, 5),
        "worst1_loss_share": bottom_k_share_of_total_loss(pnl_list, 1),
        "worst3_loss_share": bottom_k_share_of_total_loss(pnl_list, 3),
        "worst5_loss_share": bottom_k_share_of_total_loss(pnl_list, 5),
        "profit_factor_proxy": profit_factor_proxy,
    }


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
        "fold_idx",
        "gate_name",
        "target_pts",
        "stop_pts",
        "time_exit_s",
        "exit_reason",
        "pnl_pts",
        "mfe_pts",
        "mae_pts",
        "end_ret_pts",
        "hold_s",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\n--------------------------------------")
    print("Trade-Level Exit Diagnostics")
    print("--------------------------------------")
    print(f"Rows in trade file = {df.height:,}")
    print(f"N shortlist        = {len(STRATEGY_SHORTLIST)}")

    filtered_frames = []

    for s in STRATEGY_SHORTLIST:
        sub = df.filter(
            (pl.col("gate_name") == s["gate_name"]) &
            (pl.col("target_pts") == s["target_pts"]) &
            (pl.col("stop_pts") == s["stop_pts"]) &
            (pl.col("time_exit_s") == s["time_exit_s"])
        ).with_columns(
            pl.lit(s["strategy_name"]).alias("strategy_name")
        )
        filtered_frames.append(sub)

    diag_df = pl.concat(filtered_frames, how="vertical") if filtered_frames else pl.DataFrame()

    if diag_df.height == 0:
        raise ValueError("No trades matched the shortlist.")

    diag_df.write_csv(OUT_FILTERED_TRADES)

    # --------------------------------------------------------
    # Fold-level diagnostics
    # --------------------------------------------------------
    fold_records: List[Dict[str, Any]] = []

    for s in STRATEGY_SHORTLIST:
        sdf = diag_df.filter(pl.col("strategy_name") == s["strategy_name"])
        if sdf.height == 0:
            continue

        for fold_idx in sorted(sdf["fold_idx"].unique().to_list()):
            fdf = sdf.filter(pl.col("fold_idx") == fold_idx)
            stats = summarize_trade_block(fdf)

            fold_records.append({
                "strategy_name": s["strategy_name"],
                "gate_name": s["gate_name"],
                "target_pts": s["target_pts"],
                "stop_pts": s["stop_pts"],
                "time_exit_s": s["time_exit_s"],
                "fold_idx": int(fold_idx),
                **stats,
            })

    fold_df = pl.from_dicts(fold_records) if fold_records else pl.DataFrame()
    if fold_df.height > 0:
        fold_df.write_csv(OUT_FOLD_SUMMARY)

    # --------------------------------------------------------
    # Strategy-level diagnostics
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for s in STRATEGY_SHORTLIST:
        sdf = diag_df.filter(pl.col("strategy_name") == s["strategy_name"])
        if sdf.height < MIN_TRADES_TOTAL:
            continue

        stats = summarize_trade_block(sdf)

        if fold_df.height > 0:
            fs = fold_df.filter(pl.col("strategy_name") == s["strategy_name"])
            std_fold_pnl = safe_std(fs, "avg_pnl_pts")
            std_fold_win = safe_std(fs, "win_rate")
            mean_fold_trades = safe_mean(fs, "n_trades")
        else:
            std_fold_pnl = math.nan
            std_fold_win = math.nan
            mean_fold_trades = math.nan

        concentration_penalty = 0.0
        if not math.isnan(stats["top3_profit_share"]):
            concentration_penalty += 0.45 * stats["top3_profit_share"]
        if not math.isnan(stats["worst3_loss_share"]):
            concentration_penalty += 0.20 * stats["worst3_loss_share"]

        consistency_score = (
            1.10 * stats["avg_pnl_pts"]
            + 0.45 * stats["win_rate"]
            + 0.03 * stats["avg_end_ret_pts"]
            - 0.03 * stats["avg_mae_pts"]
            - 0.20 * (0.0 if math.isnan(stats["time_exit_rate"]) else stats["time_exit_rate"])
            - 0.20 * (0.0 if math.isnan(std_fold_win) else std_fold_win)
            - 0.06 * (0.0 if math.isnan(std_fold_pnl) else std_fold_pnl)
            - concentration_penalty
        )

        summary_records.append({
            "strategy_name": s["strategy_name"],
            "gate_name": s["gate_name"],
            "target_pts": s["target_pts"],
            "stop_pts": s["stop_pts"],
            "time_exit_s": s["time_exit_s"],
            "mean_fold_trades": mean_fold_trades,
            "std_fold_win_rate": std_fold_win,
            "std_fold_avg_pnl_pts": std_fold_pnl,
            **stats,
            "consistency_score": consistency_score,
        })

    summary_df = (
        pl.from_dicts(summary_records)
        .sort(["consistency_score", "avg_pnl_pts", "win_rate"], descending=[True, True, True])
        if summary_records else pl.DataFrame()
    )

    if summary_df.height == 0:
        raise ValueError("No strategy summaries were produced.")

    summary_df.write_csv(OUT_STRATEGY_SUMMARY)

    print("\nDone.")
    print(f"Wrote: {OUT_FILTERED_TRADES}")
    print(f"Wrote: {OUT_FOLD_SUMMARY}")
    print(f"Wrote: {OUT_STRATEGY_SUMMARY}")

    print("\nTop strategy diagnostics:")
    print(summary_df)

    print("\nTop fold diagnostics:")
    print(fold_df.sort(["strategy_name", "fold_idx"]).head(30))


if __name__ == "__main__":
    main()
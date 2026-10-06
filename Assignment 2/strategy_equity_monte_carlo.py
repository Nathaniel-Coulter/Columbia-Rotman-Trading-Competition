# strategy_equity_monte_carlo.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import math
import random

import polars as pl


# ============================================================
# Paths
# ============================================================

TRADE_LEVEL_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_trade_level.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_equity_monte_carlo"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRADE_SUBSET = OUTPUT_DIR / "equity_mc_trade_subset.csv"
OUT_PATH_LEVEL = OUTPUT_DIR / "equity_mc_path_level.csv"
OUT_SUMMARY = OUTPUT_DIR / "equity_mc_summary.csv"


# ============================================================
# Monte Carlo settings
# ============================================================

N_PATHS = 1000
BASE_SEED = 42

# If True, resample trades with replacement (bootstrap).
# If False, just shuffle the observed trade order without replacement.
USE_BOOTSTRAP = True

# Optional: scale pnl by number of contracts if you want later.
CONTRACT_MULTIPLIER = 1.0


# ============================================================
# Strategy shortlist
# ============================================================

STRATEGY_SHORTLIST: List[Dict[str, Any]] = [
    {
        "strategy_name": "G1_t4_s6_t900",
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "target_pts": 4.0,
        "stop_pts": 6.0,
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


# ============================================================
# Helpers
# ============================================================

def safe_mean(xs: List[float]) -> float:
    if not xs:
        return math.nan
    return sum(xs) / len(xs)


def safe_std(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def safe_quantile(xs: List[float], q: float) -> float:
    if not xs:
        return math.nan
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def max_drawdown_from_equity(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > worst_dd:
            worst_dd = dd
    return worst_dd


def longest_loss_streak(pnls: List[float]) -> int:
    worst = 0
    cur = 0
    for x in pnls:
        if x < 0:
            cur += 1
            if cur > worst:
                worst = cur
        else:
            cur = 0
    return worst


def longest_win_streak(pnls: List[float]) -> int:
    best = 0
    cur = 0
    for x in pnls:
        if x > 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def top_k_profit_share(pnls: List[float], k: int) -> float:
    wins = sorted([x for x in pnls if x > 0], reverse=True)
    total_profit = sum(wins)
    if total_profit <= 0:
        return math.nan
    return sum(wins[:k]) / total_profit


def bottom_k_loss_share(pnls: List[float], k: int) -> float:
    losses = sorted([x for x in pnls if x < 0])  # most negative first
    gross_loss = abs(sum(losses))
    if gross_loss <= 0:
        return math.nan
    return abs(sum(losses[:k])) / gross_loss


def profit_factor(pnls: List[float]) -> float:
    gross_profit = sum(x for x in pnls if x > 0)
    gross_loss = abs(sum(x for x in pnls if x < 0))
    if gross_loss == 0:
        return math.nan
    return gross_profit / gross_loss


def summarize_path(pnls: List[float]) -> Dict[str, float]:
    n = len(pnls)
    if n == 0:
        return {
            "n_trades": 0,
            "sum_pnl_pts": math.nan,
            "avg_pnl_pts": math.nan,
            "median_pnl_pts": math.nan,
            "std_pnl_pts": math.nan,
            "win_rate": math.nan,
            "profit_factor_proxy": math.nan,
            "max_drawdown_pts": math.nan,
            "terminal_equity_pts": math.nan,
            "worst_loss_streak": math.nan,
            "best_win_streak": math.nan,
            "top1_profit_share": math.nan,
            "top3_profit_share": math.nan,
            "top5_profit_share": math.nan,
            "worst1_loss_share": math.nan,
            "worst3_loss_share": math.nan,
            "worst5_loss_share": math.nan,
            "pnl_p10_trade": math.nan,
            "pnl_p50_trade": math.nan,
            "pnl_p90_trade": math.nan,
        }

    equity_curve = []
    running = 0.0
    for x in pnls:
        running += x
        equity_curve.append(running)

    return {
        "n_trades": n,
        "sum_pnl_pts": sum(pnls),
        "avg_pnl_pts": safe_mean(pnls),
        "median_pnl_pts": safe_quantile(pnls, 0.50),
        "std_pnl_pts": safe_std(pnls),
        "win_rate": sum(x > 0 for x in pnls) / n,
        "profit_factor_proxy": profit_factor(pnls),
        "max_drawdown_pts": max_drawdown_from_equity(equity_curve),
        "terminal_equity_pts": equity_curve[-1],
        "worst_loss_streak": longest_loss_streak(pnls),
        "best_win_streak": longest_win_streak(pnls),
        "top1_profit_share": top_k_profit_share(pnls, 1),
        "top3_profit_share": top_k_profit_share(pnls, 3),
        "top5_profit_share": top_k_profit_share(pnls, 5),
        "worst1_loss_share": bottom_k_loss_share(pnls, 1),
        "worst3_loss_share": bottom_k_loss_share(pnls, 3),
        "worst5_loss_share": bottom_k_loss_share(pnls, 5),
        "pnl_p10_trade": safe_quantile(pnls, 0.10),
        "pnl_p50_trade": safe_quantile(pnls, 0.50),
        "pnl_p90_trade": safe_quantile(pnls, 0.90),
    }


def summarize_across_paths(path_df: pl.DataFrame) -> Dict[str, float]:
    rows = path_df.to_dicts()

    def col(name: str) -> List[float]:
        out = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    terminal = col("terminal_equity_pts")
    drawdowns = col("max_drawdown_pts")
    win_rates = col("win_rate")
    pfs = col("profit_factor_proxy")
    top1 = col("top1_profit_share")
    top3 = col("top3_profit_share")
    worst1 = col("worst1_loss_share")
    worst3 = col("worst3_loss_share")

    prob_positive = sum(x > 0 for x in terminal) / len(terminal) if terminal else math.nan
    prob_drawdown_gt_20 = sum(x > 20 for x in drawdowns) / len(drawdowns) if drawdowns else math.nan
    prob_drawdown_gt_40 = sum(x > 40 for x in drawdowns) / len(drawdowns) if drawdowns else math.nan
    prob_drawdown_gt_60 = sum(x > 60 for x in drawdowns) / len(drawdowns) if drawdowns else math.nan

    return {
        "n_paths": int(path_df.height),
        "mean_terminal_equity_pts": safe_mean(terminal),
        "std_terminal_equity_pts": safe_std(terminal),
        "p10_terminal_equity_pts": safe_quantile(terminal, 0.10),
        "p25_terminal_equity_pts": safe_quantile(terminal, 0.25),
        "p50_terminal_equity_pts": safe_quantile(terminal, 0.50),
        "p75_terminal_equity_pts": safe_quantile(terminal, 0.75),
        "p90_terminal_equity_pts": safe_quantile(terminal, 0.90),
        "mean_max_drawdown_pts": safe_mean(drawdowns),
        "std_max_drawdown_pts": safe_std(drawdowns),
        "p50_max_drawdown_pts": safe_quantile(drawdowns, 0.50),
        "p90_max_drawdown_pts": safe_quantile(drawdowns, 0.90),
        "p95_max_drawdown_pts": safe_quantile(drawdowns, 0.95),
        "mean_win_rate": safe_mean(win_rates),
        "std_win_rate": safe_std(win_rates),
        "mean_profit_factor_proxy": safe_mean(pfs),
        "std_profit_factor_proxy": safe_std(pfs),
        "prob_positive_terminal_equity": prob_positive,
        "prob_drawdown_gt_20pts": prob_drawdown_gt_20,
        "prob_drawdown_gt_40pts": prob_drawdown_gt_40,
        "prob_drawdown_gt_60pts": prob_drawdown_gt_60,
        "mean_top1_profit_share": safe_mean(top1),
        "mean_top3_profit_share": safe_mean(top3),
        "mean_worst1_loss_share": safe_mean(worst1),
        "mean_worst3_loss_share": safe_mean(worst3),
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
        "gate_name",
        "target_pts",
        "stop_pts",
        "time_exit_s",
        "pnl_pts",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\n--------------------------------------")
    print("Strategy Equity Monte Carlo")
    print("--------------------------------------")
    print(f"Rows in trade file = {df.height:,}")
    print(f"N shortlist        = {len(STRATEGY_SHORTLIST)}")
    print(f"N paths            = {N_PATHS}")
    print(f"BASE_SEED          = {BASE_SEED}")
    print(f"USE_BOOTSTRAP      = {USE_BOOTSTRAP}")

    # --------------------------------------------------------
    # Filter strategy subset
    # --------------------------------------------------------
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

    trade_df = pl.concat(filtered_frames, how="vertical") if filtered_frames else pl.DataFrame()
    if trade_df.height == 0:
        raise ValueError("No trades matched the shortlist.")

    trade_df.write_csv(OUT_TRADE_SUBSET)

    # --------------------------------------------------------
    # Path-level Monte Carlo
    # --------------------------------------------------------
    path_records: List[Dict[str, Any]] = []

    for s in STRATEGY_SHORTLIST:
        sdf = trade_df.filter(pl.col("strategy_name") == s["strategy_name"]).sort("fold_idx")
        pnl_raw = [float(x) * CONTRACT_MULTIPLIER for x in sdf["pnl_pts"].to_list()]

        if len(pnl_raw) == 0:
            continue

        for path_idx in range(N_PATHS):
            rng = random.Random(BASE_SEED + 100000 * path_idx + hash(s["strategy_name"]) % 10000)

            if USE_BOOTSTRAP:
                path_pnls = [pnl_raw[rng.randrange(len(pnl_raw))] for _ in range(len(pnl_raw))]
            else:
                path_pnls = pnl_raw[:]
                rng.shuffle(path_pnls)

            stats = summarize_path(path_pnls)

            path_records.append({
                "strategy_name": s["strategy_name"],
                "gate_name": s["gate_name"],
                "target_pts": s["target_pts"],
                "stop_pts": s["stop_pts"],
                "time_exit_s": s["time_exit_s"],
                "path_idx": path_idx,
                **stats,
            })

    if not path_records:
        raise ValueError("No Monte Carlo path records produced.")

    path_df = pl.from_dicts(path_records)
    path_df.write_csv(OUT_PATH_LEVEL)

    # --------------------------------------------------------
    # Summary by strategy
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for s in STRATEGY_SHORTLIST:
        ps = path_df.filter(pl.col("strategy_name") == s["strategy_name"])
        agg = summarize_across_paths(ps)

        # One-line path-risk score:
        robust_score = (
            0.80 * (0.0 if math.isnan(agg["mean_terminal_equity_pts"]) else agg["mean_terminal_equity_pts"])
            + 0.50 * (0.0 if math.isnan(agg["prob_positive_terminal_equity"]) else agg["prob_positive_terminal_equity"])
            + 0.08 * (0.0 if math.isnan(agg["mean_profit_factor_proxy"]) else min(agg["mean_profit_factor_proxy"], 3.0))
            - 0.05 * (0.0 if math.isnan(agg["mean_max_drawdown_pts"]) else agg["mean_max_drawdown_pts"])
            - 0.20 * (0.0 if math.isnan(agg["prob_drawdown_gt_40pts"]) else agg["prob_drawdown_gt_40pts"])
        )

        stability_score = (
            (0.0 if math.isnan(agg["mean_terminal_equity_pts"]) else agg["mean_terminal_equity_pts"])
            - 0.20 * (0.0 if math.isnan(agg["std_terminal_equity_pts"]) else agg["std_terminal_equity_pts"])
            - 0.08 * (0.0 if math.isnan(agg["std_max_drawdown_pts"]) else agg["std_max_drawdown_pts"])
            + 0.25 * (0.0 if math.isnan(agg["prob_positive_terminal_equity"]) else agg["prob_positive_terminal_equity"])
        )

        summary_records.append({
            "strategy_name": s["strategy_name"],
            "gate_name": s["gate_name"],
            "target_pts": s["target_pts"],
            "stop_pts": s["stop_pts"],
            "time_exit_s": s["time_exit_s"],
            "use_bootstrap": USE_BOOTSTRAP,
            **agg,
            "robust_score": robust_score,
            "stability_score": stability_score,
        })

    summary_df = (
        pl.from_dicts(summary_records)
        .sort(["robust_score", "stability_score"], descending=[True, True])
    )
    summary_df.write_csv(OUT_SUMMARY)

    print("\nDone.")
    print(f"Wrote: {OUT_TRADE_SUBSET}")
    print(f"Wrote: {OUT_PATH_LEVEL}")
    print(f"Wrote: {OUT_SUMMARY}")

    print("\nTop equity Monte Carlo summaries:")
    print(summary_df)


if __name__ == "__main__":
    main()
# randomized_day_oos_validator.py
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\randomized_day_oos_validator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DAY_ORDER = OUTPUT_DIR / "randomized_day_order.csv"
OUT_TRADE_SUBSET = OUTPUT_DIR / "randomized_day_trade_subset.csv"
OUT_FOLD_RESULTS = OUTPUT_DIR / "randomized_day_fold_results.csv"
OUT_SUMMARY = OUTPUT_DIR / "randomized_day_summary.csv"

# ============================================================
# Core settings
# ============================================================
N_SEEDS = 20
BASE_SEED = 42
N_FOLDS = 4
MIN_TEST_TRADES = 25

# ============================================================
# Strategy shortlist
# ============================================================
STRATEGY_SHORTLIST: List[Dict[str, Any]] = [
    {
        "strategy_name": "G2_t5_s8_t600",
        "gate_name": "G2_VOTE2OF3_curv_cancel_spread",
        "target_pts": 5.0,
        "stop_pts": 8.0,
        "time_exit_s": 600,
    },
    {
        "strategy_name": "G2_t4_s8_t600",
        "gate_name": "G2_VOTE2OF3_curv_cancel_spread",
        "target_pts": 4.0,
        "stop_pts": 8.0,
        "time_exit_s": 600,
    },
    {
        "strategy_name": "G3_t3_s8_t120",
        "gate_name": "G3_ALL2_curv_cancelbid",
        "target_pts": 3.0,
        "stop_pts": 8.0,
        "time_exit_s": 120,
    },
]

# ============================================================
# Helpers
# ============================================================
def pct(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den


def safe_std(df: pl.DataFrame, col: str) -> float:
    if df.height <= 1:
        return 0.0
    s = df[col].std()
    return 0.0 if s is None else float(s)


def safe_mean(df: pl.DataFrame, col: str) -> float:
    if df.height == 0:
        return math.nan
    x = df[col].mean()
    return math.nan if x is None else float(x)


def safe_quantile(s: pl.Series, q: float) -> float:
    if len(s) == 0:
        return math.nan
    x = s.quantile(q)
    return math.nan if x is None else float(x)


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
        "win_rate": float((df["pnl_pts"] > 0).mean()),
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


def fold_boundaries(n: int, n_folds: int) -> List[tuple[int, int]]:
    out = []
    for i in range(n_folds):
        lo = int(round(i * n / n_folds))
        hi = int(round((i + 1) * n / n_folds))
        out.append((lo, hi))
    return out


def attach_trade_day(df: pl.DataFrame) -> pl.DataFrame:
    cols = set(df.columns)

    if "date_ymd" in cols:
        return df.with_columns(
            pl.col("date_ymd").cast(pl.Utf8).alias("trade_day")
        )

    # fallback timestamp columns
    ts_candidates = [
        "entry_ts",
        "ts",
        "event_ts",
        "entry_time",
        "open_ts",
    ]
    for c in ts_candidates:
        if c in cols:
            return df.with_columns(
                pl.col(c).str.slice(0, 10).alias("trade_day")
                if df.schema[c] == pl.Utf8
                else pl.col(c).dt.strftime("%Y-%m-%d").alias("trade_day")
            )

    raise ValueError(
        "Could not derive trade_day. Need one of: date_ymd, entry_ts, ts, event_ts, entry_time, open_ts."
    )


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

    df = attach_trade_day(df)

    print("\n--------------------------------------")
    print("Randomized Day OOS Validator")
    print("--------------------------------------")
    print(f"Rows in trade file = {df.height:,}")
    print(f"N shortlist        = {len(STRATEGY_SHORTLIST)}")
    print(f"BASE_SEED          = {BASE_SEED}")
    print(f"N_SEEDS            = {N_SEEDS}")
    print(f"N_FOLDS            = {N_FOLDS}")

    # --------------------------------------------------------
    # Filter shortlisted strategies once
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

    unique_days = sorted(set(trade_df["trade_day"].to_list()))
    if len(unique_days) < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} unique days, found {len(unique_days)}.")

    all_seed_summaries: List[pl.DataFrame] = []
    all_seed_fold_results: List[pl.DataFrame] = []
    all_seed_day_orders: List[pl.DataFrame] = []

    # ========================================================
    # Multi-seed loop
    # ========================================================
    for seed in range(BASE_SEED, BASE_SEED + N_SEEDS):
        print(f"\nRunning seed {seed}")

        rng = random.Random(seed)
        shuffled_days = unique_days[:]
        rng.shuffle(shuffled_days)

        day_order_df = pl.DataFrame({
            "seed": [seed] * len(shuffled_days),
            "shuffled_day_rank": list(range(len(shuffled_days))),
            "trade_day": shuffled_days,
        })
        all_seed_day_orders.append(day_order_df)

        fold_ranges = fold_boundaries(len(shuffled_days), N_FOLDS)
        fold_records: List[Dict[str, Any]] = []

        for fold_idx, (test_lo, test_hi) in enumerate(fold_ranges, start=1):
            test_days = set(shuffled_days[test_lo:test_hi])
            train_days = set(shuffled_days[:test_lo])

            if len(test_days) == 0 or len(train_days) == 0:
                continue

            for s in STRATEGY_SHORTLIST:
                sdf = trade_df.filter(pl.col("strategy_name") == s["strategy_name"])

                train_df = sdf.filter(pl.col("trade_day").is_in(sorted(train_days)))
                test_df = sdf.filter(pl.col("trade_day").is_in(sorted(test_days)))

                train_stats = summarize_trade_block(train_df)
                test_stats = summarize_trade_block(test_df)

                fold_records.append({
                    "seed": seed,
                    "strategy_name": s["strategy_name"],
                    "gate_name": s["gate_name"],
                    "target_pts": s["target_pts"],
                    "stop_pts": s["stop_pts"],
                    "time_exit_s": s["time_exit_s"],
                    "fold_idx": fold_idx,
                    "n_train_days": len(train_days),
                    "n_test_days": len(test_days),

                    "train_n_trades": train_stats["n_trades"],
                    "train_win_rate": train_stats["win_rate"],
                    "train_avg_pnl_pts": train_stats["avg_pnl_pts"],
                    "train_avg_mfe_pts": train_stats["avg_mfe_pts"],
                    "train_avg_mae_pts": train_stats["avg_mae_pts"],
                    "train_avg_end_ret_pts": train_stats["avg_end_ret_pts"],
                    "train_target_exit_rate": train_stats["target_exit_rate"],
                    "train_stop_exit_rate": train_stats["stop_exit_rate"],
                    "train_time_exit_rate": train_stats["time_exit_rate"],
                    "train_profit_factor_proxy": train_stats["profit_factor_proxy"],

                    "test_n_trades": test_stats["n_trades"],
                    "test_win_rate": test_stats["win_rate"],
                    "test_avg_pnl_pts": test_stats["avg_pnl_pts"],
                    "test_avg_mfe_pts": test_stats["avg_mfe_pts"],
                    "test_avg_mae_pts": test_stats["avg_mae_pts"],
                    "test_avg_end_ret_pts": test_stats["avg_end_ret_pts"],
                    "test_target_exit_rate": test_stats["target_exit_rate"],
                    "test_stop_exit_rate": test_stats["stop_exit_rate"],
                    "test_time_exit_rate": test_stats["time_exit_rate"],
                    "test_profit_factor_proxy": test_stats["profit_factor_proxy"],

                    "win_rate_degradation_test_minus_train":
                        (test_stats["win_rate"] - train_stats["win_rate"])
                        if (not math.isnan(test_stats["win_rate"]) and not math.isnan(train_stats["win_rate"]))
                        else math.nan,

                    "avg_pnl_degradation_test_minus_train":
                        (test_stats["avg_pnl_pts"] - train_stats["avg_pnl_pts"])
                        if (not math.isnan(test_stats["avg_pnl_pts"]) and not math.isnan(train_stats["avg_pnl_pts"]))
                        else math.nan,
                })

        if not fold_records:
            continue

        fold_df = pl.from_dicts(fold_records)
        all_seed_fold_results.append(fold_df)

        summary_records: List[Dict[str, Any]] = []

        for s in STRATEGY_SHORTLIST:
            fs = fold_df.filter(pl.col("strategy_name") == s["strategy_name"])

            valid = fs.filter(pl.col("test_n_trades") >= MIN_TEST_TRADES)
            coverage_status = "ok" if valid.height > 0 else "too_sparse"
            use_df = valid if valid.height > 0 else fs

            mean_test_trades = safe_mean(use_df, "test_n_trades")
            mean_test_win_rate = safe_mean(use_df, "test_win_rate")
            std_test_win_rate = safe_std(use_df, "test_win_rate")
            mean_test_avg_pnl = safe_mean(use_df, "test_avg_pnl_pts")
            std_test_avg_pnl = safe_std(use_df, "test_avg_pnl_pts")
            mean_test_avg_mfe = safe_mean(use_df, "test_avg_mfe_pts")
            mean_test_avg_mae = safe_mean(use_df, "test_avg_mae_pts")
            mean_test_avg_end = safe_mean(use_df, "test_avg_end_ret_pts")
            mean_test_target_exit_rate = safe_mean(use_df, "test_target_exit_rate")
            mean_test_stop_exit_rate = safe_mean(use_df, "test_stop_exit_rate")
            mean_test_time_exit_rate = safe_mean(use_df, "test_time_exit_rate")
            mean_test_pf = safe_mean(use_df, "test_profit_factor_proxy")
            mean_win_deg = safe_mean(use_df, "win_rate_degradation_test_minus_train")
            mean_pnl_deg = safe_mean(use_df, "avg_pnl_degradation_test_minus_train")

            robust_score = (
                1.10 * (0.0 if math.isnan(mean_test_avg_pnl) else mean_test_avg_pnl)
                + 0.40 * (0.0 if math.isnan(mean_test_win_rate) else mean_test_win_rate)
                + 0.04 * (0.0 if math.isnan(mean_test_avg_end) else mean_test_avg_end)
                + 0.06 * (0.0 if math.isnan(mean_test_pf) else min(mean_test_pf, 3.0))
                - 0.03 * (0.0 if math.isnan(mean_test_avg_mae) else mean_test_avg_mae)
                - 0.20 * (0.0 if math.isnan(mean_test_time_exit_rate) else mean_test_time_exit_rate)
                - 0.25 * max(0.0, -(0.0 if math.isnan(mean_win_deg) else mean_win_deg))
                - 0.20 * max(0.0, -(0.0 if math.isnan(mean_pnl_deg) else mean_pnl_deg))
            )

            stability_score = (
                (0.0 if math.isnan(mean_test_avg_pnl) else mean_test_avg_pnl)
                + 0.30 * (0.0 if math.isnan(mean_test_win_rate) else mean_test_win_rate)
                - 0.25 * (0.0 if math.isnan(std_test_win_rate) else std_test_win_rate)
                - 0.10 * (0.0 if math.isnan(std_test_avg_pnl) else std_test_avg_pnl)
            )

            max_usable_folds = N_FOLDS - 1
            n_valid_test_folds = int(valid.height)
            coverage_rank = max_usable_folds - n_valid_test_folds

            summary_records.append({
                "seed": seed,
                "strategy_name": s["strategy_name"],
                "gate_name": s["gate_name"],
                "target_pts": s["target_pts"],
                "stop_pts": s["stop_pts"],
                "time_exit_s": s["time_exit_s"],
                "n_valid_test_folds": n_valid_test_folds,
                "coverage_status": coverage_status,
                "coverage_rank": coverage_rank,
                "mean_test_trades": mean_test_trades,
                "mean_test_win_rate": mean_test_win_rate,
                "std_test_win_rate": std_test_win_rate,
                "mean_test_avg_pnl_pts": mean_test_avg_pnl,
                "std_test_avg_pnl_pts": std_test_avg_pnl,
                "mean_test_avg_mfe_pts": mean_test_avg_mfe,
                "mean_test_avg_mae_pts": mean_test_avg_mae,
                "mean_test_avg_end_ret_pts": mean_test_avg_end,
                "mean_test_target_exit_rate": mean_test_target_exit_rate,
                "mean_test_stop_exit_rate": mean_test_stop_exit_rate,
                "mean_test_time_exit_rate": mean_test_time_exit_rate,
                "mean_test_profit_factor_proxy": mean_test_pf,
                "mean_win_rate_degradation_test_minus_train": mean_win_deg,
                "mean_avg_pnl_degradation_test_minus_train": mean_pnl_deg,
                "robust_score": robust_score,
                "stability_score": stability_score,
            })

        seed_summary_df = pl.from_dicts(summary_records).sort(
            ["coverage_rank", "robust_score", "stability_score"],
            descending=[False, True, True],
        )
        all_seed_summaries.append(seed_summary_df)

    if not all_seed_summaries:
        raise ValueError("No seed summaries were produced.")

    all_results = pl.concat(all_seed_summaries, how="vertical")
    all_fold_results = pl.concat(all_seed_fold_results, how="vertical") if all_seed_fold_results else pl.DataFrame()
    all_day_orders = pl.concat(all_seed_day_orders, how="vertical") if all_seed_day_orders else pl.DataFrame()

    all_day_orders.write_csv(OUTPUT_DIR / "all_seed_day_orders.csv")
    all_fold_results.write_csv(OUTPUT_DIR / "all_seed_fold_results.csv")
    all_results.write_csv(OUTPUT_DIR / "all_seed_results.csv")

    avg_results = (
        all_results
        .group_by(["strategy_name", "gate_name", "target_pts", "stop_pts", "time_exit_s"])
        .agg([
            pl.mean("mean_test_avg_pnl_pts").alias("avg_pnl"),
            pl.std("mean_test_avg_pnl_pts").alias("pnl_stability"),
            pl.mean("mean_test_win_rate").alias("avg_win_rate"),
            pl.std("mean_test_win_rate").alias("win_rate_stability"),
            pl.mean("robust_score").alias("avg_robust_score"),
            pl.mean("stability_score").alias("avg_stability_score"),
            pl.mean("n_valid_test_folds").alias("avg_valid_test_folds"),
        ])
        .sort("avg_robust_score", descending=True)
    )

    avg_results.write_csv(OUTPUT_DIR / "multi_seed_average_summary.csv")

    print("\n======================================")
    print("MULTI-SEED AVERAGED RESULTS")
    print("======================================")
    print(avg_results)

    # --------------------------------------------------------
    # Filter shortlisted strategies
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
    # Shuffle unique days once, shared across all strategies
    # --------------------------------------------------------
    unique_days = sorted(set(trade_df["trade_day"].to_list()))
    if len(unique_days) < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} unique days, found {len(unique_days)}.")

    shuffled_days = unique_days[:]
    rng.shuffle(shuffled_days)

    day_order_df = pl.DataFrame({
        "shuffled_day_rank": list(range(len(shuffled_days))),
        "trade_day": shuffled_days,
    })
    day_order_df.write_csv(OUT_DAY_ORDER)

    # --------------------------------------------------------
    # Fold splits on shuffled day order
    # anchored by shuffled order
    # --------------------------------------------------------
    fold_ranges = fold_boundaries(len(shuffled_days), N_FOLDS)
    fold_records: List[Dict[str, Any]] = []

    for fold_idx, (test_lo, test_hi) in enumerate(fold_ranges, start=1):
        test_days = set(shuffled_days[test_lo:test_hi])
        train_days = set(shuffled_days[:test_lo])

        if len(test_days) == 0 or len(train_days) == 0:
            continue

        for s in STRATEGY_SHORTLIST:
            sdf = trade_df.filter(pl.col("strategy_name") == s["strategy_name"])

            train_df = sdf.filter(pl.col("trade_day").is_in(sorted(train_days)))
            test_df = sdf.filter(pl.col("trade_day").is_in(sorted(test_days)))

            train_stats = summarize_trade_block(train_df)
            test_stats = summarize_trade_block(test_df)

            fold_records.append({
                "strategy_name": s["strategy_name"],
                "gate_name": s["gate_name"],
                "target_pts": s["target_pts"],
                "stop_pts": s["stop_pts"],
                "time_exit_s": s["time_exit_s"],
                "fold_idx": fold_idx,
                "n_train_days": len(train_days),
                "n_test_days": len(test_days),

                "train_n_trades": train_stats["n_trades"],
                "train_win_rate": train_stats["win_rate"],
                "train_avg_pnl_pts": train_stats["avg_pnl_pts"],
                "train_avg_mfe_pts": train_stats["avg_mfe_pts"],
                "train_avg_mae_pts": train_stats["avg_mae_pts"],
                "train_avg_end_ret_pts": train_stats["avg_end_ret_pts"],
                "train_target_exit_rate": train_stats["target_exit_rate"],
                "train_stop_exit_rate": train_stats["stop_exit_rate"],
                "train_time_exit_rate": train_stats["time_exit_rate"],
                "train_profit_factor_proxy": train_stats["profit_factor_proxy"],

                "test_n_trades": test_stats["n_trades"],
                "test_win_rate": test_stats["win_rate"],
                "test_avg_pnl_pts": test_stats["avg_pnl_pts"],
                "test_avg_mfe_pts": test_stats["avg_mfe_pts"],
                "test_avg_mae_pts": test_stats["avg_mae_pts"],
                "test_avg_end_ret_pts": test_stats["avg_end_ret_pts"],
                "test_target_exit_rate": test_stats["target_exit_rate"],
                "test_stop_exit_rate": test_stats["stop_exit_rate"],
                "test_time_exit_rate": test_stats["time_exit_rate"],
                "test_profit_factor_proxy": test_stats["profit_factor_proxy"],

                "win_rate_degradation_test_minus_train":
                    (test_stats["win_rate"] - train_stats["win_rate"])
                    if (not math.isnan(test_stats["win_rate"]) and not math.isnan(train_stats["win_rate"]))
                    else math.nan,

                "avg_pnl_degradation_test_minus_train":
                    (test_stats["avg_pnl_pts"] - train_stats["avg_pnl_pts"])
                    if (not math.isnan(test_stats["avg_pnl_pts"]) and not math.isnan(train_stats["avg_pnl_pts"]))
                    else math.nan,
            })

    if not fold_records:
        raise ValueError("No fold records produced.")

    fold_df = pl.from_dicts(fold_records)
    fold_df.write_csv(OUT_FOLD_RESULTS)

    # --------------------------------------------------------
    # Summary by strategy
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for s in STRATEGY_SHORTLIST:
        fs = fold_df.filter(pl.col("strategy_name") == s["strategy_name"])

        valid = fs.filter(pl.col("test_n_trades") >= MIN_TEST_TRADES)
        coverage_status = "ok" if valid.height > 0 else "too_sparse"
        use_df = valid if valid.height > 0 else fs

        mean_test_trades = safe_mean(use_df, "test_n_trades")
        mean_test_win_rate = safe_mean(use_df, "test_win_rate")
        std_test_win_rate = safe_std(use_df, "test_win_rate")
        mean_test_avg_pnl = safe_mean(use_df, "test_avg_pnl_pts")
        std_test_avg_pnl = safe_std(use_df, "test_avg_pnl_pts")
        mean_test_avg_mfe = safe_mean(use_df, "test_avg_mfe_pts")
        mean_test_avg_mae = safe_mean(use_df, "test_avg_mae_pts")
        mean_test_avg_end = safe_mean(use_df, "test_avg_end_ret_pts")
        mean_test_target_exit_rate = safe_mean(use_df, "test_target_exit_rate")
        mean_test_stop_exit_rate = safe_mean(use_df, "test_stop_exit_rate")
        mean_test_time_exit_rate = safe_mean(use_df, "test_time_exit_rate")
        mean_test_pf = safe_mean(use_df, "test_profit_factor_proxy")
        mean_win_deg = safe_mean(use_df, "win_rate_degradation_test_minus_train")
        mean_pnl_deg = safe_mean(use_df, "avg_pnl_degradation_test_minus_train")

        robust_score = (
            1.10 * (0.0 if math.isnan(mean_test_avg_pnl) else mean_test_avg_pnl)
            + 0.40 * (0.0 if math.isnan(mean_test_win_rate) else mean_test_win_rate)
            + 0.04 * (0.0 if math.isnan(mean_test_avg_end) else mean_test_avg_end)
            + 0.06 * (0.0 if math.isnan(mean_test_pf) else min(mean_test_pf, 3.0))
            - 0.03 * (0.0 if math.isnan(mean_test_avg_mae) else mean_test_avg_mae)
            - 0.20 * (0.0 if math.isnan(mean_test_time_exit_rate) else mean_test_time_exit_rate)
            - 0.25 * max(0.0, -(0.0 if math.isnan(mean_win_deg) else mean_win_deg))
            - 0.20 * max(0.0, -(0.0 if math.isnan(mean_pnl_deg) else mean_pnl_deg))
        )

        stability_score = (
            (0.0 if math.isnan(mean_test_avg_pnl) else mean_test_avg_pnl)
            + 0.30 * (0.0 if math.isnan(mean_test_win_rate) else mean_test_win_rate)
            - 0.25 * (0.0 if math.isnan(std_test_win_rate) else std_test_win_rate)
            - 0.10 * (0.0 if math.isnan(std_test_avg_pnl) else std_test_avg_pnl)
        )

        coverage_rank = N_FOLDS - int(use_df.height)

        summary_records.append({
            "strategy_name": s["strategy_name"],
            "gate_name": s["gate_name"],
            "target_pts": s["target_pts"],
            "stop_pts": s["stop_pts"],
            "time_exit_s": s["time_exit_s"],
            "n_valid_test_folds": int(use_df.height),
            "coverage_status": coverage_status,
            "coverage_rank": coverage_rank,
            "mean_test_trades": mean_test_trades,
            "mean_test_win_rate": mean_test_win_rate,
            "std_test_win_rate": std_test_win_rate,
            "mean_test_avg_pnl_pts": mean_test_avg_pnl,
            "std_test_avg_pnl_pts": std_test_avg_pnl,
            "mean_test_avg_mfe_pts": mean_test_avg_mfe,
            "mean_test_avg_mae_pts": mean_test_avg_mae,
            "mean_test_avg_end_ret_pts": mean_test_avg_end,
            "mean_test_target_exit_rate": mean_test_target_exit_rate,
            "mean_test_stop_exit_rate": mean_test_stop_exit_rate,
            "mean_test_time_exit_rate": mean_test_time_exit_rate,
            "mean_test_profit_factor_proxy": mean_test_pf,
            "mean_win_rate_degradation_test_minus_train": mean_win_deg,
            "mean_avg_pnl_degradation_test_minus_train": mean_pnl_deg,
            "robust_score": robust_score,
            "stability_score": stability_score,
        })

    summary_df = (
        pl.from_dicts(summary_records)
        .sort(
            ["coverage_rank", "robust_score", "stability_score"],
            descending=[False, True, True],
        )
    )
    summary_df.write_csv(OUT_SUMMARY)

    summary_df = summary_df.with_columns(
        pl.lit(seed).alias("seed")
    )

    all_seed_summaries.append(summary_df)

    print("\nDone.")
    print(f"Wrote: {OUT_DAY_ORDER}")
    print(f"Wrote: {OUT_TRADE_SUBSET}")
    print(f"Wrote: {OUT_FOLD_RESULTS}")
    print(f"Wrote: {OUT_SUMMARY}")

    print("\nTop randomized-day summaries:")
    print(summary_df)

    print("\nTop randomized-day fold diagnostics:")
    print(
        fold_df
        .sort(["strategy_name", "fold_idx"])
        .head(30)
    )


if __name__ == "__main__":
    main()
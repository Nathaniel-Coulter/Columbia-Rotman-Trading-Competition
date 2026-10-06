# quick_peek_multi_seed_validator.py
from __future__ import annotations

from pathlib import Path
import polars as pl

BASE_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\randomized_day_oos_validator"
)

AVG_PATH = BASE_DIR / "multi_seed_average_summary.csv"
ALL_RESULTS_PATH = BASE_DIR / "all_seed_results.csv"

TOP_N = 20


def main() -> None:
    if not AVG_PATH.exists():
        raise FileNotFoundError(f"Missing file: {AVG_PATH}")
    if not ALL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {ALL_RESULTS_PATH}")

    avg_df = pl.read_csv(AVG_PATH)
    all_df = pl.read_csv(ALL_RESULTS_PATH)

    print("\n======================================")
    print("MULTI-SEED VALIDATOR PEEK")
    print("======================================")
    print(f"avg_df shape      = {avg_df.shape}")
    print(f"all_results shape = {all_df.shape}")

    print("\n=== avg_df columns ===")
    print(avg_df.columns)

    print("\n=== Top strategies by avg_robust_score ===")
    cols_top = [
        c for c in [
            "strategy_name",
            "gate_name",
            "target_pts",
            "stop_pts",
            "time_exit_s",
            "avg_pnl",
            "pnl_stability",
            "avg_win_rate",
            "win_rate_stability",
            "avg_robust_score",
            "avg_stability_score",
            "avg_valid_test_folds",
        ]
        if c in avg_df.columns
    ]
    print(
        avg_df
        .sort("avg_robust_score", descending=True)
        .select(cols_top)
        .head(TOP_N)
    )

    print("\n=== Top strategies by avg_stability_score ===")
    print(
        avg_df
        .sort("avg_stability_score", descending=True)
        .select(cols_top)
        .head(TOP_N)
    )

    print("\n=== Top strategies by pnl stability (lowest std first) ===")
    if "pnl_stability" in avg_df.columns:
        print(
            avg_df
            .sort("pnl_stability", descending=False)
            .select(cols_top)
            .head(TOP_N)
        )

    print("\n=== Top strategies by win-rate stability (lowest std first) ===")
    if "win_rate_stability" in avg_df.columns:
        print(
            avg_df
            .sort("win_rate_stability", descending=False)
            .select(cols_top)
            .head(TOP_N)
        )

    print("\n=== One-line robustness ranking metric ===")
    # A simple ranking heuristic:
    # higher avg_robust_score is better
    # lower pnl_stability and lower win_rate_stability are better
    if all(c in avg_df.columns for c in ["avg_robust_score", "pnl_stability", "win_rate_stability"]):
        ranked = avg_df.with_columns([
            (
                pl.col("avg_robust_score")
                - 0.35 * pl.col("pnl_stability").fill_null(0.0)
                - 0.20 * pl.col("win_rate_stability").fill_null(0.0)
            ).alias("robustness_rank_score")
        ])

        print(
            ranked
            .sort("robustness_rank_score", descending=True)
            .select([
                c for c in [
                    "strategy_name",
                    "gate_name",
                    "target_pts",
                    "stop_pts",
                    "time_exit_s",
                    "avg_pnl",
                    "pnl_stability",
                    "avg_win_rate",
                    "win_rate_stability",
                    "avg_robust_score",
                    "avg_stability_score",
                    "avg_valid_test_folds",
                    "robustness_rank_score",
                ] if c in ranked.columns
            ])
            .head(TOP_N)
        )

    print("\n=== Seed-level spread for each strategy ===")
    group_cols = [c for c in ["strategy_name", "gate_name", "target_pts", "stop_pts", "time_exit_s"] if c in all_df.columns]

    spread_cols = []
    if "mean_test_avg_pnl_pts" in all_df.columns:
        spread_cols += [
            pl.col("mean_test_avg_pnl_pts").mean().alias("seed_mean_pnl"),
            pl.col("mean_test_avg_pnl_pts").std().alias("seed_std_pnl"),
            pl.col("mean_test_avg_pnl_pts").min().alias("seed_min_pnl"),
            pl.col("mean_test_avg_pnl_pts").max().alias("seed_max_pnl"),
        ]
    if "mean_test_win_rate" in all_df.columns:
        spread_cols += [
            pl.col("mean_test_win_rate").mean().alias("seed_mean_win_rate"),
            pl.col("mean_test_win_rate").std().alias("seed_std_win_rate"),
            pl.col("mean_test_win_rate").min().alias("seed_min_win_rate"),
            pl.col("mean_test_win_rate").max().alias("seed_max_win_rate"),
        ]
    if "robust_score" in all_df.columns:
        spread_cols += [
            pl.col("robust_score").mean().alias("seed_mean_robust"),
            pl.col("robust_score").std().alias("seed_std_robust"),
            pl.col("robust_score").min().alias("seed_min_robust"),
            pl.col("robust_score").max().alias("seed_max_robust"),
        ]

    spread_df = (
        all_df
        .group_by(group_cols)
        .agg(spread_cols)
        .sort("seed_mean_robust", descending=True)
    )
    print(spread_df.head(TOP_N))

    print("\n=== Fragility screen ===")
    # Flags strategies that look good on average but vary too much across seeds
    frag_df = spread_df
    if "seed_mean_robust" in frag_df.columns and "seed_std_robust" in frag_df.columns:
        frag_df = frag_df.with_columns([
            (pl.col("seed_std_robust") / (pl.col("seed_mean_robust").abs() + 1e-9)).alias("robust_cv"),
        ])
        print(
            frag_df
            .sort(["robust_cv", "seed_mean_robust"], descending=[False, True])
            .head(TOP_N)
        )

    print("\n=== Worst-case seed behavior ===")
    worst_cols = [c for c in [
        "strategy_name",
        "gate_name",
        "target_pts",
        "stop_pts",
        "time_exit_s",
        "seed_min_pnl",
        "seed_max_pnl",
        "seed_min_robust",
        "seed_max_robust",
        "seed_std_pnl",
        "seed_std_robust",
    ] if c in spread_df.columns]
    print(
        spread_df
        .sort("seed_min_robust", descending=False)
        .select(worst_cols)
        .head(TOP_N)
    )


if __name__ == "__main__":
    main()
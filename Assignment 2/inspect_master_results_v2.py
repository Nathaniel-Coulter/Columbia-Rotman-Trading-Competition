# inspect_master_results_v2.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


MASTER_RESULTS_CSV = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\master_sweep_lgbm_v1\master_results.csv"
)

TOP_N = 25
TOP_DECILE_COUNT = 60


def safe_fill_numeric(df: pd.DataFrame, cols: list[str], fill_value=np.nan) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = fill_value
    return df


def main() -> None:
    if not MASTER_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {MASTER_RESULTS_CSV}")

    df = pd.read_csv(MASTER_RESULTS_CSV)

    print("\n=== BASIC SHAPE ===")
    print(f"Rows: {len(df)}")
    print(f"Cols: {len(df.columns)}")

    print("\n=== STATUS COUNTS ===")
    print(df["status"].value_counts(dropna=False))

    df_ok = df[df["status"] == "ok"].copy()

    print("\n=== OK VS NON-OK ===")
    print(f"status == 'ok': {len(df_ok)}")
    print(f"status != 'ok': {len(df) - len(df_ok)}")

    if len(df_ok) == 0:
        print("\nNo successful runs to inspect.")
        return

    numeric_cols = [
        "band_k",
        "horizon_s",
        "milestone_ticks",
        "threshold_points",
        "n_total",
        "n_train",
        "n_val",
        "n_test",
        "n_pos_train",
        "n_pos_val",
        "n_pos_test",
        "train_positive_rate",
        "val_positive_rate",
        "test_positive_rate",
        "val_roc_auc",
        "test_roc_auc",
        "val_pr_auc",
        "test_pr_auc",
        "val_log_loss",
        "test_log_loss",
        "val_brier",
        "test_brier",
        "best_iteration",
    ]
    df_ok = safe_fill_numeric(df_ok, numeric_cols)

    # ------------------------------------------------------------
    # Derived stability / quality columns
    # ------------------------------------------------------------
    eps = 1e-12

    df_ok["auc_gap_abs"] = (df_ok["val_roc_auc"] - df_ok["test_roc_auc"]).abs()
    df_ok["pr_gap_abs"] = (df_ok["val_pr_auc"] - df_ok["test_pr_auc"]).abs()
    df_ok["logloss_gap"] = df_ok["test_log_loss"] - df_ok["val_log_loss"]

    df_ok["min_pos_any_split"] = df_ok[
        ["n_pos_train", "n_pos_val", "n_pos_test"]
    ].min(axis=1)

    df_ok["min_rows_eval_split"] = df_ok[["n_val", "n_test"]].min(axis=1)

    # sample-size softness: compresses huge differences but rewards larger samples
    df_ok["sample_size_score"] = np.log1p(df_ok["n_total"].clip(lower=0))

    # stability score: penalize val-test gaps
    df_ok["stability_score"] = (
        1.0
        - 0.5 * df_ok["auc_gap_abs"].fillna(1.0)
        - 0.5 * df_ok["pr_gap_abs"].fillna(1.0)
    )

    # simple positive-support score
    df_ok["positive_support_score"] = np.log1p(df_ok["min_pos_any_split"].clip(lower=0))

    # ------------------------------------------------------------
    # Composite scores
    # ------------------------------------------------------------
    # Balanced model-quality score
    df_ok["score_balanced"] = (
        0.40 * df_ok["test_roc_auc"].fillna(0.0)
        + 0.40 * df_ok["test_pr_auc"].fillna(0.0)
        + 0.10 * df_ok["stability_score"].fillna(0.0)
        + 0.10 * (
            df_ok["sample_size_score"] / df_ok["sample_size_score"].max()
        ).fillna(0.0)
    )

    # More conservative score: rewards stability and enough positives a bit more
    df_ok["score_conservative"] = (
        0.35 * df_ok["test_roc_auc"].fillna(0.0)
        + 0.30 * df_ok["test_pr_auc"].fillna(0.0)
        + 0.20 * df_ok["stability_score"].fillna(0.0)
        + 0.15 * (
            df_ok["positive_support_score"] / df_ok["positive_support_score"].max()
        ).fillna(0.0)
    )

    # Event-quality score: useful when you care about stronger tradeable setups
    df_ok["score_trade_focus"] = (
        0.30 * df_ok["test_roc_auc"].fillna(0.0)
        + 0.35 * df_ok["test_pr_auc"].fillna(0.0)
        + 0.15 * df_ok["test_positive_rate"].fillna(0.0)
        + 0.10 * df_ok["stability_score"].fillna(0.0)
        + 0.10 * (
            df_ok["sample_size_score"] / df_ok["sample_size_score"].max()
        ).fillna(0.0)
    )

    # ------------------------------------------------------------
    # Pretty columns for ranking displays
    # ------------------------------------------------------------
    show_cols = [
        "side",
        "band_k",
        "horizon_s",
        "milestone_ticks",
        "threshold_points",
        "vol_filter_name",
        "n_total",
        "n_train",
        "n_val",
        "n_test",
        "n_pos_train",
        "n_pos_val",
        "n_pos_test",
        "val_roc_auc",
        "test_roc_auc",
        "val_pr_auc",
        "test_pr_auc",
        "test_log_loss",
        "test_brier",
        "auc_gap_abs",
        "pr_gap_abs",
        "best_iteration",
        "score_balanced",
        "score_conservative",
        "score_trade_focus",
        "top_shap_1",
        "top_shap_2",
        "top_shap_3",
        "top_gain_1",
        "top_gain_2",
        "top_gain_3",
    ]

    # ------------------------------------------------------------
    # Top tables
    # ------------------------------------------------------------
    print("\n=== TOP 25 OK RUNS BY score_balanced ===")
    print(
        df_ok.sort_values(
            ["score_balanced", "test_roc_auc", "test_pr_auc"],
            ascending=[False, False, False],
        )[show_cols]
        .head(TOP_N)
        .to_string(index=False)
    )

    print("\n=== TOP 25 OK RUNS BY score_conservative ===")
    print(
        df_ok.sort_values(
            ["score_conservative", "test_roc_auc", "test_pr_auc"],
            ascending=[False, False, False],
        )[show_cols]
        .head(TOP_N)
        .to_string(index=False)
    )

    print("\n=== TOP 25 OK RUNS BY score_trade_focus ===")
    print(
        df_ok.sort_values(
            ["score_trade_focus", "test_pr_auc", "test_roc_auc"],
            ascending=[False, False, False],
        )[show_cols]
        .head(TOP_N)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Top-decile-like slice using balanced score
    # ------------------------------------------------------------
    df_top = (
        df_ok.sort_values(
            ["score_balanced", "test_roc_auc", "test_pr_auc"],
            ascending=[False, False, False],
        )
        .head(TOP_DECILE_COUNT)
        .copy()
    )

    print(f"\n=== TOP {TOP_DECILE_COUNT} BY score_balanced PARAMETER CONCENTRATION ===")
    for c in ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]:
        print(f"\n-- {c} --")
        print(df_top[c].value_counts(dropna=False))

    # ------------------------------------------------------------
    # Average score by parameter
    # ------------------------------------------------------------
    print("\n=== AVERAGE score_balanced BY PARAMETER ===")
    for c in ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]:
        print(f"\n-- {c} --")
        print(
            df_ok.groupby(c)["score_balanced"]
            .agg(["count", "mean", "median", "max"])
            .sort_values("mean", ascending=False)
        )

    print("\n=== AVERAGE score_conservative BY PARAMETER ===")
    for c in ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]:
        print(f"\n-- {c} --")
        print(
            df_ok.groupby(c)["score_conservative"]
            .agg(["count", "mean", "median", "max"])
            .sort_values("mean", ascending=False)
        )

    # ------------------------------------------------------------
    # Best exact combinations by mean balanced score
    # ------------------------------------------------------------
    group_cols = ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]
    exact_combo = (
        df_ok.groupby(group_cols)["score_balanced"]
        .agg(["count", "mean", "median", "max"])
        .sort_values("mean", ascending=False)
        .head(20)
    )
    print("\n=== BEST EXACT COMBINATIONS BY MEAN score_balanced ===")
    print(exact_combo)

    # ------------------------------------------------------------
    # Top feature frequencies in best runs
    # ------------------------------------------------------------
    feature_cols = [
        "top_shap_1", "top_shap_2", "top_shap_3", "top_shap_4", "top_shap_5",
        "top_gain_1", "top_gain_2", "top_gain_3", "top_gain_4", "top_gain_5",
    ]
    feature_cols = [c for c in feature_cols if c in df_top.columns]

    top_feats = (
        pd.Series(df_top[feature_cols].values.ravel())
        .dropna()
        .astype(str)
        .value_counts()
        .head(30)
    )
    print(f"\n=== TOP FEATURE FREQUENCIES IN TOP {TOP_DECILE_COUNT} RUNS ===")
    print(top_feats)

    # ------------------------------------------------------------
    # Save ranked outputs
    # ------------------------------------------------------------
    out_dir = MASTER_RESULTS_CSV.parent

    df_ok.sort_values(
        ["score_balanced", "test_roc_auc", "test_pr_auc"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "master_results_ranked_by_score_balanced.csv", index=False)

    df_ok.sort_values(
        ["score_conservative", "test_roc_auc", "test_pr_auc"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "master_results_ranked_by_score_conservative.csv", index=False)

    df_ok.sort_values(
        ["score_trade_focus", "test_pr_auc", "test_roc_auc"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "master_results_ranked_by_score_trade_focus.csv", index=False)

    print("\nWrote ranked CSVs:")
    print(out_dir / "master_results_ranked_by_score_balanced.csv")
    print(out_dir / "master_results_ranked_by_score_conservative.csv")
    print(out_dir / "master_results_ranked_by_score_trade_focus.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
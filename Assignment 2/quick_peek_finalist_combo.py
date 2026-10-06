# quick_peek_finalist_combo.py
import polars as pl

SUMMARY_PATH = (
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs"
    r"\shock_final_entry_gate_search_v2\shock_final_entry_gate_summary.csv"
)

FOLDS_PATH = (
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs"
    r"\shock_final_entry_gate_search_v2\shock_final_entry_gate_fold_results.csv"
)

summary = pl.read_csv(SUMMARY_PATH)
folds = pl.read_csv(FOLDS_PATH)

print("\n=== SUMMARY: shape / columns ===")
print(summary.shape)
print(summary.columns)

print("\n=== SUMMARY: top rows (key columns) ===")
summary_cols = [
    c for c in [
        "combo_name",
        "logic",
        "confirmations",
        "n_valid_test_folds",
        "mean_test_n_events",
        "min_test_n_events",
        "max_test_n_events",
        "mean_test_hit_rate",
        "std_test_hit_rate",
        "mean_test_avg_mr_pts",
        "std_test_avg_mr_pts",
        "mean_test_avg_mae_pts",
        "std_test_avg_mae_pts",
        "mean_hit_rate_degradation_test_minus_train",
        "mean_avg_mr_pts_degradation_test_minus_train",
        "robust_score",
        "stability_score",
    ] if c in summary.columns
]
print(summary.select(summary_cols).head(30))

print("\n=== SUMMARY: stability_score distribution ===")
if "stability_score" in summary.columns:
    print(summary.select([
        pl.len().alias("n_rows"),
        pl.col("stability_score").null_count().alias("stability_nulls"),
        (pl.col("stability_score") == 0).sum().alias("stability_eq_0"),
        pl.col("stability_score").min().alias("stability_min"),
        pl.col("stability_score").max().alias("stability_max"),
    ]))

print("\n=== SUMMARY: n_valid_test_folds distribution ===")
if "n_valid_test_folds" in summary.columns:
    print(
        summary.group_by("n_valid_test_folds")
        .agg(pl.len().alias("n_combos"))
        .sort("n_valid_test_folds")
    )

print("\n=== SUMMARY: rows with stability_score == 0 ===")
if "stability_score" in summary.columns:
    print(
        summary
        .filter(pl.col("stability_score") == 0)
        .select(summary_cols)
        .head(50)
    )

print("\n=== FOLDS: shape / columns ===")
print(folds.shape)
print(folds.columns)

print("\n=== FOLDS: key columns sample ===")
fold_cols = [
    c for c in [
        "combo_name",
        "logic",
        "confirmations",
        "fold_idx",
        "train_n_events",
        "test_n_events",
        "is_valid_test_fold",
        "train_hit_rate",
        "test_hit_rate",
        "train_avg_mr_pts",
        "test_avg_mr_pts",
        "train_avg_mae_pts",
        "test_avg_mae_pts",
        "hit_rate_degradation_test_minus_train",
        "avg_mr_pts_degradation_test_minus_train",
    ] if c in folds.columns
]
print(
    folds
    .select(fold_cols)
    .sort([c for c in ["combo_name", "fold_idx"] if c in folds.columns])
    .head(60)
)

print("\n=== FOLDS: valid test fold counts by combo ===")
group_keys = [c for c in ["combo_name", "logic", "confirmations"] if c in folds.columns]
if group_keys and "is_valid_test_fold" in folds.columns:
    valid_counts = (
        folds.group_by(group_keys)
        .agg([
            pl.len().alias("n_folds"),
            pl.col("is_valid_test_fold").sum().alias("n_valid_test_folds"),
            pl.col("test_n_events").mean().alias("mean_test_n_events"),
            pl.col("test_n_events").min().alias("min_test_n_events"),
            pl.col("test_n_events").max().alias("max_test_n_events"),
        ])
        .sort(["n_valid_test_folds", "mean_test_n_events"], descending=[False, False])
    )
    print(valid_counts.head(50))

print("\n=== FOLDS: combos where all fold metrics are zero-ish ===")
zero_check_cols = [c for c in [
    "test_hit_rate",
    "test_avg_mr_pts",
    "test_avg_mae_pts",
    "hit_rate_degradation_test_minus_train",
    "avg_mr_pts_degradation_test_minus_train",
] if c in folds.columns]

if group_keys and zero_check_cols:
    zeroish = (
        folds.group_by(group_keys)
        .agg([
            *[(pl.col(c) == 0).sum().alias(f"{c}_eq0") for c in zero_check_cols],
            pl.len().alias("n_folds"),
        ])
    )
    print(zeroish.head(50))
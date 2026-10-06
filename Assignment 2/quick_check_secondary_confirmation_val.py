# quick_check_secondary_confirmation_validator_bugs.py
import polars as pl

df = pl.read_csv(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\secondary_confirmation_validator\secondary_confirmation_validator_fold_results.csv"
)

print(df.select([
    pl.len().alias("n_rows"),
    pl.col("test_n_events").min().alias("min_test_n"),
    pl.col("test_n_events").max().alias("max_test_n"),
    pl.col("is_valid_test_fold").sum().alias("n_valid_test_folds"),
]))

print(df.select([
    pl.col("test_hit_rate").null_count().alias("test_hit_rate_nulls"),
    pl.col("test_avg_mr_pts").null_count().alias("test_avg_mr_pts_nulls"),
    pl.col("test_avg_mae_pts").null_count().alias("test_avg_mae_pts_nulls"),
    pl.col("hit_rate_degradation_test_minus_train").null_count().alias("hit_deg_nulls"),
]))

print(df.sort(["secondary_feature", "fold_idx"]).select([
    "secondary_feature",
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
]))
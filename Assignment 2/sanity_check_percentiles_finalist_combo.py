# sanity_check_percentiles_finalist_combo.py
import polars as pl

df = pl.read_csv(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\finalist_confirmation_combo_gates\finalist_confirmation_event_level.csv"
)

print(df.select([
    pl.col("post_entry_mfe_pts").quantile(0.50).alias("mfe_p50"),
    pl.col("post_entry_mfe_pts").quantile(0.90).alias("mfe_p90"),
    pl.col("post_entry_mfe_pts").quantile(0.95).alias("mfe_p95"),
    pl.col("post_entry_mfe_pts").quantile(0.99).alias("mfe_p99"),

    pl.col("post_entry_mae_pts").quantile(0.50).alias("mae_p50"),
    pl.col("post_entry_mae_pts").quantile(0.90).alias("mae_p90"),
    pl.col("post_entry_mae_pts").quantile(0.95).alias("mae_p95"),
    pl.col("post_entry_mae_pts").quantile(0.99).alias("mae_p99"),

    pl.col("post_entry_end_ret_pts").quantile(0.50).alias("endret_p50"),
    pl.col("post_entry_end_ret_pts").quantile(0.90).alias("endret_p90"),
    pl.col("post_entry_end_ret_pts").quantile(0.95).alias("endret_p95"),
    pl.col("post_entry_end_ret_pts").quantile(0.99).alias("endret_p99"),
]))
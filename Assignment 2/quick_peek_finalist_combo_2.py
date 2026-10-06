# quick_peek_finalist_combo_2.py
import polars as pl

df = pl.read_csv(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\finalist_confirmation_combo_gates\finalist_confirmation_event_level.csv"
)

print(df.select([
    pl.col("post_entry_mfe_pts").min().alias("mfe_min"),
    pl.col("post_entry_mfe_pts").max().alias("mfe_max"),
    pl.col("post_entry_mae_pts").min().alias("mae_min"),
    pl.col("post_entry_mae_pts").max().alias("mae_max"),
    pl.col("post_entry_end_ret_pts").min().alias("endret_min"),
    pl.col("post_entry_end_ret_pts").max().alias("endret_max"),
]))
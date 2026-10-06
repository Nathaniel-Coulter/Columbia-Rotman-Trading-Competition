# v3_spot_check.py
import polars as pl

df = pl.read_parquet(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\shock_events\shock_event_features_v3.parquet"
)

print(df.select([
    pl.col("spread").drop_nulls().min().alias("spread_min"),
    pl.col("spread").drop_nulls().max().alias("spread_max"),
    pl.col("microprice_minus_mid").drop_nulls().min().alias("mp_minus_mid_min"),
    pl.col("microprice_minus_mid").drop_nulls().max().alias("mp_minus_mid_max"),
]))

print(df.select([
    pl.col("queue_imbalance").drop_nulls().min().alias("qi_min"),
    pl.col("queue_imbalance").drop_nulls().max().alias("qi_max"),
    pl.col("best_bid_queue_ratio").drop_nulls().min().alias("bbq_ratio_min"),
    pl.col("best_bid_queue_ratio").drop_nulls().max().alias("bbq_ratio_max"),
    pl.col("best_ask_queue_ratio").drop_nulls().min().alias("baq_ratio_min"),
    pl.col("best_ask_queue_ratio").drop_nulls().max().alias("baq_ratio_max"),
]))

print(df.select([
    pl.col("cancel_rate_bid").drop_nulls().min().alias("cancel_bid_min"),
    pl.col("cancel_rate_bid").drop_nulls().max().alias("cancel_bid_max"),
    pl.col("remove_rate_near_bid").drop_nulls().min().alias("remove_bid_min"),
    pl.col("remove_rate_near_bid").drop_nulls().max().alias("remove_bid_max"),
    pl.col("bid_liquidity_shock").drop_nulls().min().alias("shock_bid_min"),
    pl.col("bid_liquidity_shock").drop_nulls().max().alias("shock_bid_max"),
]))
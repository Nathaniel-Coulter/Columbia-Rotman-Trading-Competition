# v4_spot_check.py
import polars as pl

df = pl.read_parquet(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\shock_events\shock_event_features_v4.parquet"
)

print("\n=== Basic flow rates ===")
print(df.select([
    pl.col("trades_per_second").drop_nulls().min().alias("trades_per_second_min"),
    pl.col("trades_per_second").drop_nulls().max().alias("trades_per_second_max"),
    pl.col("volume_per_second").drop_nulls().min().alias("volume_per_second_min"),
    pl.col("volume_per_second").drop_nulls().max().alias("volume_per_second_max"),
    pl.col("avg_trade_size").drop_nulls().min().alias("avg_trade_size_min"),
    pl.col("avg_trade_size").drop_nulls().max().alias("avg_trade_size_max"),
]))

print("\n=== Trade-sign / imbalance sanity ===")
print(df.select([
    pl.col("trade_sign_autocorr_10s").drop_nulls().min().alias("ts_autocorr_10s_min"),
    pl.col("trade_sign_autocorr_10s").drop_nulls().max().alias("ts_autocorr_10s_max"),
    pl.col("trade_imbalance_5s").drop_nulls().min().alias("trade_imbalance_5s_min"),
    pl.col("trade_imbalance_5s").drop_nulls().max().alias("trade_imbalance_5s_max"),
    pl.col("trade_imbalance_30s").drop_nulls().min().alias("trade_imbalance_30s_min"),
    pl.col("trade_imbalance_30s").drop_nulls().max().alias("trade_imbalance_30s_max"),
    pl.col("trade_imbalance_60s").drop_nulls().min().alias("trade_imbalance_60s_min"),
    pl.col("trade_imbalance_60s").drop_nulls().max().alias("trade_imbalance_60s_max"),
]))

print("\n=== Aggressive flow / signed volume ===")
print(df.select([
    pl.col("aggressive_buy_volume").drop_nulls().min().alias("aggr_buy_vol_min"),
    pl.col("aggressive_buy_volume").drop_nulls().max().alias("aggr_buy_vol_max"),
    pl.col("aggressive_sell_volume").drop_nulls().min().alias("aggr_sell_vol_min"),
    pl.col("aggressive_sell_volume").drop_nulls().max().alias("aggr_sell_vol_max"),
    pl.col("aggressive_volume_ratio").drop_nulls().min().alias("aggr_vol_ratio_min"),
    pl.col("aggressive_volume_ratio").drop_nulls().max().alias("aggr_vol_ratio_max"),
    pl.col("signed_volume").drop_nulls().min().alias("signed_volume_min"),
    pl.col("signed_volume").drop_nulls().max().alias("signed_volume_max"),
]))

print("\n=== OFI / burst metrics ===")
print(df.select([
    pl.col("OFI").drop_nulls().min().alias("OFI_min"),
    pl.col("OFI").drop_nulls().max().alias("OFI_max"),
    pl.col("OFI_5s").drop_nulls().min().alias("OFI_5s_min"),
    pl.col("OFI_5s").drop_nulls().max().alias("OFI_5s_max"),
    pl.col("OFI_30s").drop_nulls().min().alias("OFI_30s_min"),
    pl.col("OFI_30s").drop_nulls().max().alias("OFI_30s_max"),
    pl.col("OFI_60s").drop_nulls().min().alias("OFI_60s_min"),
    pl.col("OFI_60s").drop_nulls().max().alias("OFI_60s_max"),
    pl.col("flow_burst_ratio").drop_nulls().min().alias("flow_burst_ratio_min"),
    pl.col("flow_burst_ratio").drop_nulls().max().alias("flow_burst_ratio_max"),
    pl.col("volume_burst_ratio").drop_nulls().min().alias("volume_burst_ratio_min"),
    pl.col("volume_burst_ratio").drop_nulls().max().alias("volume_burst_ratio_max"),
    pl.col("OFI_burst_ratio").drop_nulls().min().alias("OFI_burst_ratio_min"),
    pl.col("OFI_burst_ratio").drop_nulls().max().alias("OFI_burst_ratio_max"),
]))

print("\n=== CVD / queue dynamics ===")
print(df.select([
    pl.col("CVD").drop_nulls().min().alias("CVD_min"),
    pl.col("CVD").drop_nulls().max().alias("CVD_max"),
    pl.col("cvd_slope").drop_nulls().min().alias("cvd_slope_min"),
    pl.col("cvd_slope").drop_nulls().max().alias("cvd_slope_max"),
    pl.col("cvd_acceleration").drop_nulls().min().alias("cvd_accel_min"),
    pl.col("cvd_acceleration").drop_nulls().max().alias("cvd_accel_max"),
    pl.col("queue_imbalance_mean_5s").drop_nulls().min().alias("qi_mean_5s_min"),
    pl.col("queue_imbalance_mean_5s").drop_nulls().max().alias("qi_mean_5s_max"),
    pl.col("queue_imbalance_std_5s").drop_nulls().min().alias("qi_std_5s_min"),
    pl.col("queue_imbalance_std_5s").drop_nulls().max().alias("qi_std_5s_max"),
    pl.col("queue_imbalance_positive_ratio").drop_nulls().min().alias("qi_pos_ratio_min"),
    pl.col("queue_imbalance_positive_ratio").drop_nulls().max().alias("qi_pos_ratio_max"),
]))

print("\n=== Impact metrics (includes requested decay checks) ===")
print(df.select([
    pl.col("impact_efficiency_3s").drop_nulls().min().alias("impact_eff_3s_min"),
    pl.col("impact_efficiency_3s").drop_nulls().max().alias("impact_eff_3s_max"),
    pl.col("impact_efficiency_10s").drop_nulls().min().alias("impact_eff_10s_min"),
    pl.col("impact_efficiency_10s").drop_nulls().max().alias("impact_eff_10s_max"),
    pl.col("impact_efficiency_30s").drop_nulls().min().alias("impact_eff_30s_min"),
    pl.col("impact_efficiency_30s").drop_nulls().max().alias("impact_eff_30s_max"),
    pl.col("impact_decay_3s").drop_nulls().min().alias("impact_decay_3s_min"),
    pl.col("impact_decay_3s").drop_nulls().max().alias("impact_decay_3s_max"),
    pl.col("impact_decay_5s").drop_nulls().min().alias("impact_decay_5s_min"),
    pl.col("impact_decay_5s").drop_nulls().max().alias("impact_decay_5s_max"),
]))

print("\n=== Impact asymmetry / absorption ===")
print(df.select([
    pl.col("buy_impact_efficiency_10s").drop_nulls().min().alias("buy_impact_eff_10s_min"),
    pl.col("buy_impact_efficiency_10s").drop_nulls().max().alias("buy_impact_eff_10s_max"),
    pl.col("sell_impact_efficiency_10s").drop_nulls().min().alias("sell_impact_eff_10s_min"),
    pl.col("sell_impact_efficiency_10s").drop_nulls().max().alias("sell_impact_eff_10s_max"),
    pl.col("impact_asymmetry_10s").drop_nulls().min().alias("impact_asym_10s_min"),
    pl.col("impact_asymmetry_10s").drop_nulls().max().alias("impact_asym_10s_max"),
    pl.col("absorption_ratio").drop_nulls().min().alias("absorption_ratio_min"),
    pl.col("absorption_ratio").drop_nulls().max().alias("absorption_ratio_max"),
    pl.col("flow_impact").drop_nulls().min().alias("flow_impact_min"),
    pl.col("flow_impact").drop_nulls().max().alias("flow_impact_max"),
]))

print("\n=== Toxicity metrics ===")
print(df.select([
    pl.col("flow_toxicity").drop_nulls().min().alias("flow_toxicity_min"),
    pl.col("flow_toxicity").drop_nulls().max().alias("flow_toxicity_max"),
    pl.col("flow_toxicity_10s").drop_nulls().min().alias("flow_toxicity_10s_min"),
    pl.col("flow_toxicity_10s").drop_nulls().max().alias("flow_toxicity_10s_max"),
    pl.col("flow_toxicity_30s").drop_nulls().min().alias("flow_toxicity_30s_min"),
    pl.col("flow_toxicity_30s").drop_nulls().max().alias("flow_toxicity_30s_max"),
    pl.col("toxicity_ratio").drop_nulls().min().alias("toxicity_ratio_min"),
    pl.col("toxicity_ratio").drop_nulls().max().alias("toxicity_ratio_max"),
]))

print("\n=== Null counts for new v4 columns ===")
v4_cols = [
    "trades_per_second",
    "volume_per_second",
    "avg_trade_size",

    "trade_sign_autocorr_10s",
    "trade_sign_autocorr_30s",
    "trade_sign_autocorr_60s",

    "aggressive_buy_volume",
    "aggressive_sell_volume",
    "aggressive_volume_ratio",

    "OFI",
    "OFI_5s",
    "OFI_30s",
    "OFI_60s",

    "CVD",
    "cvd_slope",
    "cvd_acceleration",

    "flow_burst_ratio",
    "volume_burst_ratio",
    "OFI_burst_ratio",

    "signed_volume",

    "trade_imbalance",
    "trade_imbalance_5s",
    "trade_imbalance_30s",
    "trade_imbalance_60s",

    "queue_imbalance_mean_5s",
    "queue_imbalance_std_5s",
    "queue_imbalance_slope_5s",
    "queue_imbalance_positive_ratio",

    "impact_efficiency_3s",
    "impact_efficiency_10s",
    "impact_efficiency_30s",

    "impact_decay_3s",
    "impact_decay_5s",

    "buy_impact_efficiency_10s",
    "sell_impact_efficiency_10s",
    "impact_asymmetry_10s",

    "absorption_ratio",
    "flow_impact",

    "flow_toxicity",
    "flow_toxicity_10s",
    "flow_toxicity_30s",
    "toxicity_ratio",
]

print(df.select(
    [pl.len().alias("n_rows")] +
    [pl.col(c).null_count().alias(f"{c}_nulls") for c in v4_cols if c in df.columns]
))
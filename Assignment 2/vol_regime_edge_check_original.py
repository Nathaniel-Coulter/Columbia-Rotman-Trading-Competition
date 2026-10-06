# vol_regime_edge_check_original.py
import polars as pl

FEATURES_PATH = (
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

VOL_COL = "realized_volatility_30s"
TARGET_COL = "mr_pts_t180"   # change as needed
THRESHOLD = 4.0

df = pl.read_parquet(FEATURES_PATH)

df = df.filter(
    pl.col(VOL_COL).is_not_null() &
    pl.col(TARGET_COL).is_not_null()
)

n = df.height
if n == 0:
    raise ValueError("No rows left after filtering.")

df = df.with_columns(
    (pl.col(TARGET_COL) >= THRESHOLD).cast(pl.Int8).alias("reversion_flag")
)

baseline = df.select(
    pl.len().alias("n_events"),
    pl.col("reversion_flag").mean().alias("overall_reversion_rate")
)
print("\nOverall baseline:")
print(baseline)

df = df.with_columns(
    pl.col(VOL_COL).rank("dense").alias("vol_rank")
)

df = df.with_columns([
    (pl.col("vol_rank") / n).alias("vol_percentile"),
    ((pl.col("vol_rank") / n) * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .alias("vol_decile"),
])

result = (
    df.group_by("vol_decile")
    .agg([
        pl.len().alias("n_events"),
        pl.col("reversion_flag").mean().alias("reversion_rate"),
        pl.col(VOL_COL).min().alias(f"{VOL_COL}_min"),
        pl.col(VOL_COL).max().alias(f"{VOL_COL}_max"),
    ])
    .sort("vol_decile")
)

print("\nVolatility decile edge check:")
print(result)
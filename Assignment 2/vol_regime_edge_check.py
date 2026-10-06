# vol_regime_edge_check.py
import polars as pl

FEATURES_PATH = (
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

VOL_COL = "realized_volatility_30s"
TARGET_COL = "mr_pts_t180"   # change as needed
THRESHOLD = 4.0

# --------------------------------------------------
# OPTIONAL FILTERS
# --------------------------------------------------

# Example values:
# BAND_FILTER = None / 1 / 2 / 3
# SIDE_FILTER = None / "upper" / "lower"

BAND_FILTER = 3
SIDE_FILTER = "lower"

# --------------------------------------------------

df = pl.read_parquet(FEATURES_PATH)

# base filtering
df = df.filter(
    pl.col(VOL_COL).is_not_null() &
    pl.col(TARGET_COL).is_not_null()
)

# band filter
if BAND_FILTER is not None:
    df = df.filter(pl.col("band_k") == BAND_FILTER)

# side filter
if SIDE_FILTER is not None:
    df = df.filter(pl.col("side") == SIDE_FILTER)

n = df.height
if n == 0:
    raise ValueError("No rows left after filtering.")

print("\n--------------------------------------")
print("Volatility Regime Edge Diagnostic")
print("--------------------------------------")
print(f"Rows analyzed: {n}")

print(f"BAND_FILTER = {BAND_FILTER}")
print(f"SIDE_FILTER = {SIDE_FILTER}")
print(f"TARGET_COL  = {TARGET_COL}")
print(f"THRESHOLD   = {THRESHOLD}")

# --------------------------------------------------
# build reversion flag
# --------------------------------------------------

df = df.with_columns(
    (pl.col(TARGET_COL) >= THRESHOLD).cast(pl.Int8).alias("reversion_flag")
)

# --------------------------------------------------
# overall baseline
# --------------------------------------------------

baseline = df.select(
    pl.len().alias("n_events"),
    pl.col("reversion_flag").mean().alias("overall_reversion_rate")
)

print("\nOverall baseline:")
print(baseline)

# --------------------------------------------------
# volatility deciles
# --------------------------------------------------

df = df.with_columns(
    pl.col(VOL_COL).rank("dense").alias("vol_rank")
)

n = df.height

df = df.with_columns([
    (pl.col("vol_rank") / n).alias("vol_percentile"),
    ((pl.col("vol_rank") / n) * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .alias("vol_decile"),
])

# --------------------------------------------------
# compute results
# --------------------------------------------------

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
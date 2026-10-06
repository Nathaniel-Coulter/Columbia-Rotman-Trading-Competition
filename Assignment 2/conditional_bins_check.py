# conditional_bins_check.py
import polars as pl

FEATURES_PATH = (
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

# --------------------------------------------------
# subset settings
# --------------------------------------------------
TARGET_COL = "mr_pts_t300"
THRESHOLD = 4.0

BAND_FILTER = [1]          # e.g. [1] for band 1 lower subset
SIDE_FILTER = "lower"      # "upper" or "lower"
MIN_VOL_DECILE = 7
VOL_COL_FOR_FILTER = "realized_volatility_30s"

# choose one feature to inspect
FEATURE_TO_BIN = "wall_strength_opposite_5"

# number of equal-count bins
N_BINS = 10
# --------------------------------------------------

df = pl.read_parquet(FEATURES_PATH)

# base filtering
df = df.filter(
    pl.col(TARGET_COL).is_not_null() &
    pl.col(FEATURE_TO_BIN).is_not_null() &
    pl.col(VOL_COL_FOR_FILTER).is_not_null()
)

# target
df = df.with_columns(
    (pl.col(TARGET_COL) >= THRESHOLD).cast(pl.Int8).alias("target")
)

# subset filters
if BAND_FILTER is not None:
    df = df.filter(pl.col("band_k").is_in(BAND_FILTER))

if SIDE_FILTER is not None:
    df = df.filter(pl.col("side") == SIDE_FILTER)

# vol decile filter
if MIN_VOL_DECILE is not None:
    n_tmp = df.height
    df = df.with_columns(
        pl.col(VOL_COL_FOR_FILTER).rank("dense").alias("vol_rank_tmp")
    )
    df = df.with_columns(
        ((pl.col("vol_rank_tmp") / n_tmp) * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .alias("vol_decile_tmp")
    )
    df = df.filter(pl.col("vol_decile_tmp") >= MIN_VOL_DECILE)
    df = df.drop(["vol_rank_tmp", "vol_decile_tmp"])

if df.height == 0:
    raise ValueError("No rows left after filtering.")

print("\n--------------------------------------")
print("Conditional empirical bins check")
print("--------------------------------------")
print(f"Rows analyzed: {df.height}")
print(f"TARGET_COL      = {TARGET_COL}")
print(f"THRESHOLD       = {THRESHOLD}")
print(f"BAND_FILTER     = {BAND_FILTER}")
print(f"SIDE_FILTER     = {SIDE_FILTER}")
print(f"MIN_VOL_DECILE  = {MIN_VOL_DECILE}")
print(f"FEATURE_TO_BIN  = {FEATURE_TO_BIN}")
print(f"N_BINS          = {N_BINS}")

# rank feature and create equal-count bins
n = df.height
df = df.with_columns(
    pl.col(FEATURE_TO_BIN).rank("ordinal").alias("feature_rank")
)

df = df.with_columns(
    (((pl.col("feature_rank") - 1) / n) * N_BINS)
    .floor()
    .clip(0, N_BINS - 1)
    .cast(pl.Int64)
    .alias("feature_bin")
)

result = (
    df.group_by("feature_bin")
    .agg([
        pl.len().alias("n_events"),
        pl.col("target").mean().alias("hit_rate"),
        pl.col(TARGET_COL).mean().alias("avg_mr_pts"),
        pl.col(FEATURE_TO_BIN).min().alias(f"{FEATURE_TO_BIN}_min"),
        pl.col(FEATURE_TO_BIN).max().alias(f"{FEATURE_TO_BIN}_max"),
        pl.col(FEATURE_TO_BIN).median().alias(f"{FEATURE_TO_BIN}_median"),
    ])
    .sort("feature_bin")
)

print("\nBinned results:")
print(result)
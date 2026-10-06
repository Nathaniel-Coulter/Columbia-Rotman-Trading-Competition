# conditional_bins_check_2d.py
import polars as pl

FEATURES_PATH = (
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

# --------------------------------------------------
# subset settings
# --------------------------------------------------
TARGET_COL = "mr_pts_t300"
THRESHOLD = 4.0

BAND_FILTER = [2]          # e.g. [1] for band 1 lower subset
SIDE_FILTER = "upper"      # "upper" or "lower"
MIN_VOL_DECILE = 7
VOL_COL_FOR_FILTER = "realized_volatility_30s"

# choose two features to inspect
FEATURE_X = "toxicity_ratio"
FEATURE_Y = "impact_efficiency_10s"

# number of equal-count bins per feature
N_BINS_X = 5
N_BINS_Y = 5
# --------------------------------------------------

df = pl.read_parquet(FEATURES_PATH)

# base filtering
df = df.filter(
    pl.col(TARGET_COL).is_not_null() &
    pl.col(FEATURE_X).is_not_null() &
    pl.col(FEATURE_Y).is_not_null() &
    pl.col(VOL_COL_FOR_FILTER).is_not_null()
)

# binary target
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
    if n_tmp == 0:
        raise ValueError("No rows left before volatility filtering.")

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
print("Conditional empirical 2D bins check")
print("--------------------------------------")
print(f"Rows analyzed: {df.height}")
print(f"TARGET_COL      = {TARGET_COL}")
print(f"THRESHOLD       = {THRESHOLD}")
print(f"BAND_FILTER     = {BAND_FILTER}")
print(f"SIDE_FILTER     = {SIDE_FILTER}")
print(f"MIN_VOL_DECILE  = {MIN_VOL_DECILE}")
print(f"FEATURE_X       = {FEATURE_X}")
print(f"FEATURE_Y       = {FEATURE_Y}")
print(f"N_BINS_X        = {N_BINS_X}")
print(f"N_BINS_Y        = {N_BINS_Y}")

# --------------------------------------------------
# equal-count bins for X
# --------------------------------------------------
n_x = df.height
df = df.with_columns(
    pl.col(FEATURE_X).rank("ordinal").alias("x_rank")
)

df = df.with_columns(
    (((pl.col("x_rank") - 1) / n_x) * N_BINS_X)
    .floor()
    .clip(0, N_BINS_X - 1)
    .cast(pl.Int64)
    .alias("x_bin")
)

# --------------------------------------------------
# equal-count bins for Y
# --------------------------------------------------
n_y = df.height
df = df.with_columns(
    pl.col(FEATURE_Y).rank("ordinal").alias("y_rank")
)

df = df.with_columns(
    (((pl.col("y_rank") - 1) / n_y) * N_BINS_Y)
    .floor()
    .clip(0, N_BINS_Y - 1)
    .cast(pl.Int64)
    .alias("y_bin")
)

# --------------------------------------------------
# 2D grouped results
# --------------------------------------------------
result = (
    df.group_by(["x_bin", "y_bin"])
    .agg([
        pl.len().alias("n_events"),
        pl.col("target").mean().alias("hit_rate"),
        pl.col(TARGET_COL).mean().alias("avg_mr_pts"),

        pl.col(FEATURE_X).min().alias(f"{FEATURE_X}_min"),
        pl.col(FEATURE_X).max().alias(f"{FEATURE_X}_max"),
        pl.col(FEATURE_X).median().alias(f"{FEATURE_X}_median"),

        pl.col(FEATURE_Y).min().alias(f"{FEATURE_Y}_min"),
        pl.col(FEATURE_Y).max().alias(f"{FEATURE_Y}_max"),
        pl.col(FEATURE_Y).median().alias(f"{FEATURE_Y}_median"),
    ])
    .sort(["x_bin", "y_bin"])
)

print("\n2D binned results:")
print(result)

# --------------------------------------------------
# pivot tables for easier reading
# --------------------------------------------------
counts_pivot = (
    result
    .select(["x_bin", "y_bin", "n_events"])
    .pivot(index="x_bin", on="y_bin", values="n_events")
    .sort("x_bin")
)

hit_rate_pivot = (
    result
    .select(["x_bin", "y_bin", "hit_rate"])
    .pivot(index="x_bin", on="y_bin", values="hit_rate")
    .sort("x_bin")
)

avg_mr_pts_pivot = (
    result
    .select(["x_bin", "y_bin", "avg_mr_pts"])
    .pivot(index="x_bin", on="y_bin", values="avg_mr_pts")
    .sort("x_bin")
)

print("\nCounts pivot:")
print(counts_pivot)

print("\nHit-rate pivot:")
print(hit_rate_pivot)

print("\nAverage MR pts pivot:")
print(avg_mr_pts_pivot)
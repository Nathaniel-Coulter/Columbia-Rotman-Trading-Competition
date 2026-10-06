# immediate_bounce_by_time_ticks.py
import polars as pl

PATH = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\touch_outcomes.parquet"

HORIZONS = [1, 3, 5, 10, 15, 30, 60, 120, 300, 600, 900]
TICK = 0.25
TICK_THRESHOLDS = list(range(1, 41))  # 1..40 ticks (up to 10 points)

df = pl.read_parquet(PATH)

all_frames = []

for H in HORIZONS:
    col = f"mfe_pts_t{H}"
    if col not in df.columns:
        continue

    dH = df.filter(pl.col(col).is_not_null())

    exprs = [
        pl.len().alias("n_nonnull"),
        (pl.col(col) / TICK).mean().alias("mean_endret_ticks"),
        (pl.col(col) / TICK).median().alias("median_endret_ticks"),
    ]

    for tk in TICK_THRESHOLDS:
        exprs.append((pl.col(col) >= (tk * TICK)).mean().alias(f"p_ge_{tk}tick"))

    out = (
        dH
        .group_by(["band_k", "side"])
        .agg(exprs)
        .with_columns(pl.lit(H).cast(pl.Int32).alias("horizon_s"))
    )

    all_frames.append(out)

final = pl.concat(all_frames).sort(["band_k", "side", "horizon_s"])
print(final)
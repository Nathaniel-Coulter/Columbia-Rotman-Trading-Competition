import polars as pl

df = pl.read_parquet(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\touch_outcomes.parquet"
)

HORIZONS = [1,5,10,30,60,120,180,300,600,900]
POINT_THRESHOLDS = list(range(1, 11))

all_frames = []

for H in HORIZONS:
    col = f"mfe_pts_t{H}"
    if col not in df.columns:
        continue

    # base: only rows where the outcome exists
    base = df.filter(pl.col(col).is_not_null())

    exprs = [
        pl.len().alias("n_nonnull"),
        pl.col(col).mean().alias("mean_pts"),
        pl.col(col).median().alias("median_pts"),
    ]

    for pts in POINT_THRESHOLDS:
        exprs.append(
            (pl.col(col) >= pts).fill_null(False).mean().alias(f"p_ge_{pts}pt")
        )

    out = (
        base
        .group_by(["band_k", "side"])
        .agg(exprs)
        .with_columns([
            pl.lit(H).alias("horizon_s"),
        ])
    )

    all_frames.append(out)

final = pl.concat(all_frames).sort(["band_k", "side", "horizon_s"])
print(final)
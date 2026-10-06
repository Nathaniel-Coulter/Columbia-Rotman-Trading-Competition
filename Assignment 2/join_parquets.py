# join_parquets.py
from pathlib import Path
import polars as pl

V5_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v5.parquet"
)

MR_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\touch_reversion_targets.parquet"
)

OUT_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

df_v5 = pl.read_parquet(V5_PATH)
df_mr = pl.read_parquet(MR_PATH)

mr_cols = (
    ["event_id"]
    + [f"mr_pts_t{H}" for H in [1,3,5,10,15,30,60,120,180,300,600,900]]
    + [f"hit_vwap_t{H}" for H in [1,3,5,10,15,30,60,120,180,300,600,900]]
    + [f"mr_pts_n{N}" for N in [10,25,50,100,250,500,1000,2500]]
    + [f"hit_vwap_n{N}" for N in [10,25,50,100,250,500,1000,2500]]
)

df_mr_small = df_mr.select([c for c in mr_cols if c in df_mr.columns])

# Safety checks
if df_v5.select(pl.col("event_id").is_duplicated().any()).item():
    raise ValueError("Duplicate event_id found in v5 parquet")

if df_mr_small.select(pl.col("event_id").is_duplicated().any()).item():
    raise ValueError("Duplicate event_id found in MR parquet")

df_joined = df_v5.join(df_mr_small, on="event_id", how="left")

missing_matches = df_joined.select(pl.col("mr_pts_t1").is_null().sum()).item()
print(f"Rows with missing MR match: {missing_matches:,}")

# Optional sanity diagnostics
n_v5 = df_v5.height
n_joined = df_joined.height
mr_nulls_600 = (
    df_joined.select(pl.col("mr_pts_t600").is_null().sum()).item()
    if "mr_pts_t600" in df_joined.columns else None
)

print(f"v5 rows: {n_v5:,}")
print(f"joined rows: {n_joined:,}")
print(f"mr_pts_t600 nulls after join: {mr_nulls_600:,}" if mr_nulls_600 is not None else "mr_pts_t600 missing")

df_joined.write_parquet(OUT_PATH)
print(f"Wrote: {OUT_PATH}")
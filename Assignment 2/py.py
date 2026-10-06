import polars as pl
p = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\touch_outcomes.parquet"
df = pl.read_parquet(p)
print("rows", df.height, "cols", len(df.columns))
print("unique dates", df.select(pl.col("date_ymd").n_unique()).item())
print("min date", df.select(pl.col("date_ymd").min()).item())
print("max date", df.select(pl.col("date_ymd").max()).item())
print(df.group_by(["band_k","side"]).len().sort(["band_k","side"]))
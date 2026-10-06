# check_fold_coverage.py
import polars as pl

df = pl.read_csv(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_trade_level.csv",
    try_parse_dates=True,
)

print(
    df.group_by("fold_idx")
    .agg([
        pl.col("date_ymd").min().alias("min_date"),
        pl.col("date_ymd").max().alias("max_date"),
        pl.len().alias("n_trades"),
    ])
    .sort("fold_idx")
)
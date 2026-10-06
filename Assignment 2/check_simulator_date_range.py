# check_simulator_date_range.py

from pathlib import Path
import polars as pl

paths = [
r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_event_level.csv",
r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_trade_level.csv",
]

print("\n==============================")
print("SIMULATOR DATE RANGE CHECK")
print("==============================")

for p in paths:

    p = Path(p)
    df = pl.read_csv(p, try_parse_dates=True)

    print(f"\nFILE: {p.name}")
    print(f"Rows: {df.height}")

    if "date_ymd" in df.columns:
        print(
            "date range:",
            df["date_ymd"].min(),
            "->",
            df["date_ymd"].max()
        )
# validate_reversion_targets_vs_original.py
from pathlib import Path
import polars as pl

ORIG_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\touch_outcomes.parquet"
)

MR_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\touch_reversion_targets.parquet"
)

KEY_COLS = ["event_id", "date_ymd", "ts", "band_k", "side", "event_n", "entry_price", "vwap0", "sigma0", "distance0"]

def main():
    df_orig = pl.read_parquet(ORIG_PATH)
    df_mr = pl.read_parquet(MR_PATH)

    print("\n=== BASIC SHAPE ===")
    print(f"Original rows: {df_orig.height:,} | cols: {len(df_orig.columns)}")
    print(f"MR rows:       {df_mr.height:,} | cols: {len(df_mr.columns)}")

    print("\n=== UNIQUE EVENT_ID CHECK ===")
    orig_dupes = df_orig.select(pl.col("event_id").is_duplicated().sum()).item()
    mr_dupes = df_mr.select(pl.col("event_id").is_duplicated().sum()).item()
    print(f"Original duplicated event_id count: {orig_dupes:,}")
    print(f"MR duplicated event_id count:       {mr_dupes:,}")

    orig_unique = df_orig.select(pl.col("event_id").n_unique()).item()
    mr_unique = df_mr.select(pl.col("event_id").n_unique()).item()
    print(f"Original unique event_id count: {orig_unique:,}")
    print(f"MR unique event_id count:       {mr_unique:,}")

    print("\n=== COLUMN DIFFERENCES ===")
    orig_cols = set(df_orig.columns)
    mr_cols = set(df_mr.columns)

    only_orig = sorted(orig_cols - mr_cols)
    only_mr = sorted(mr_cols - orig_cols)

    print(f"Columns only in original ({len(only_orig)}):")
    print(only_orig[:50])
    if len(only_orig) > 50:
        print("...")

    print(f"\nColumns only in MR ({len(only_mr)}):")
    print(only_mr[:50])
    if len(only_mr) > 50:
        print("...")

    print("\n=== EVENT_ID SET DIFFERENCES ===")
    orig_ids = df_orig.select("event_id")
    mr_ids = df_mr.select("event_id")

    missing_in_mr = orig_ids.join(mr_ids, on="event_id", how="anti")
    extra_in_mr = mr_ids.join(orig_ids, on="event_id", how="anti")

    print(f"Event IDs in original but missing in MR: {missing_in_mr.height:,}")
    print(f"Event IDs in MR but not in original:     {extra_in_mr.height:,}")

    if missing_in_mr.height > 0:
        print("\nSample missing in MR:")
        print(missing_in_mr.head(10))

    if extra_in_mr.height > 0:
        print("\nSample extra in MR:")
        print(extra_in_mr.head(10))

    print("\n=== KEY FIELD MATCH CHECK (matched event_ids only) ===")
    df_orig_small = df_orig.select(KEY_COLS).rename({c: f"{c}_orig" for c in KEY_COLS if c != "event_id"})
    df_mr_small = df_mr.select(KEY_COLS).rename({c: f"{c}_mr" for c in KEY_COLS if c != "event_id"})

    joined = df_orig_small.join(df_mr_small, on="event_id", how="inner")

    print(f"Matched rows on event_id: {joined.height:,}")

    compare_pairs = [
        ("date_ymd_orig", "date_ymd_mr"),
        ("ts_orig", "ts_mr"),
        ("band_k_orig", "band_k_mr"),
        ("side_orig", "side_mr"),
        ("event_n_orig", "event_n_mr"),
        ("entry_price_orig", "entry_price_mr"),
        ("vwap0_orig", "vwap0_mr"),
        ("sigma0_orig", "sigma0_mr"),
        ("distance0_orig", "distance0_mr"),
    ]

    mismatch_exprs = []
    for a, b in compare_pairs:
        mismatch_exprs.append(
            (pl.col(a) != pl.col(b)).sum().alias(f"mismatch__{a}__vs__{b}")
        )

    mismatch_summary = joined.select(mismatch_exprs)
    print(mismatch_summary)

    print("\n=== SAMPLE MATCHED ROWS ===")
    print(joined.select([
        "event_id",
        "ts_orig", "ts_mr",
        "side_orig", "side_mr",
        "band_k_orig", "band_k_mr",
        "entry_price_orig", "entry_price_mr",
    ]).head(10))

if __name__ == "__main__":
    main()
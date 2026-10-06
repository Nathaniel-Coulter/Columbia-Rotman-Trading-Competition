# csv_to_parquet_es_t2.py
import os
import re
from pathlib import Path
import polars as pl

RAW_ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\raw\Future_ES_T2")
OUT_ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet")
LOG_DIR = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BAD_FILES_LOG = LOG_DIR / "bad_files.txt"

DAY_RE = re.compile(r"^(?P<day>\d{8})(?:\.\w+)?$")

# L2 columns (7)
L2_COLS = ["ts_raw", "level", "type", "price", "volume", "depth", "action"]
L2_SCHEMA = {
    "ts_raw": pl.Utf8,
    "level": pl.Int8,
    "type": pl.Int8,
    "price": pl.Float64,
    "volume": pl.Int64,
    "depth": pl.Int8,
    "action": pl.Int8,
}

# L1-shaped columns (5) inside some files
L1_COLS = ["ts_raw", "level", "type", "price", "volume"]
L1_SCHEMA = {
    "ts_raw": pl.Utf8,
    "level": pl.Int8,
    "type": pl.Int8,
    "price": pl.Float64,
    "volume": pl.Int64,
}

def parse_ts_expr(col: str = "ts_raw") -> pl.Expr:
    """
    MarketTick timestamp format:
      YYYYMMDDhhmmss + 6 digits microseconds
    """
    s = pl.col(col)
    return (
        pl.concat_str(
            [
                s.str.slice(0, 4), pl.lit("-"), s.str.slice(4, 2), pl.lit("-"), s.str.slice(6, 2),
                pl.lit(" "),
                s.str.slice(8, 2), pl.lit(":"), s.str.slice(10, 2), pl.lit(":"), s.str.slice(12, 2),
                pl.lit("."), s.str.slice(14, 6),
            ]
        )
        .str.strptime(pl.Datetime(time_unit="us"), format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        .alias("ts")
    )

def iter_daily_files(month_dir: Path):
    for p in month_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() == ".7z":
            continue
        if p.name.lower().startswith("data_structure"):
            continue
        m = DAY_RE.match(p.name)
        if not m:
            continue
        yield p, m.group("day")

def detect_ncols(path: Path) -> int:
    """
    Peek first non-empty line and count ';' separated fields.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # count fields by splitting
            return len(line.split(";"))
    return 0

def log_bad(path: Path, reason: str):
    with open(BAD_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(f"{path}\t{reason}\n")

def convert_day(path: Path, day: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "part-000.parquet"
    if out_file.exists():
        print(f"  skip {path.name} -> already converted")
        return

    ncols = detect_ncols(path)
    if ncols == 7:
        cols = L2_COLS
        schema = L2_SCHEMA
        lf = pl.scan_csv(
            path,
            has_header=False,
            separator=";",
            new_columns=cols,
            schema_overrides=schema,
            infer_schema_length=0,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
    elif ncols == 5:
        cols = L1_COLS
        schema = L1_SCHEMA
        lf = pl.scan_csv(
            path,
            has_header=False,
            separator=";",
            new_columns=cols,
            schema_overrides=schema,
            infer_schema_length=0,
            ignore_errors=True,
            truncate_ragged_lines=True,
        ).with_columns([
            pl.lit(None, dtype=pl.Int8).alias("depth"),
            pl.lit(None, dtype=pl.Int8).alias("action"),
        ])
    else:
        log_bad(path, f"unexpected column count: {ncols}")
        print(f"  ⚠ skip {path.name} (unexpected columns: {ncols})")
        return

    lf2 = (
        lf
        .with_columns([parse_ts_expr("ts_raw")])
        .drop("ts_raw")
        .with_columns([
            pl.lit(int(day[0:4])).alias("year"),
            pl.lit(int(day[4:6])).alias("month"),
            pl.lit(int(day[6:8])).alias("day"),
        ])
        .select(["ts", "year", "month", "day", "level", "type", "price", "volume", "depth", "action"])
    )

    lf2.sink_parquet(out_file)
    print(f"  ✅ {path.name} -> {out_file}")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    month_dirs = sorted([p for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.startswith("Future_ES_T2_")])
    if not month_dirs:
        raise RuntimeError(f"No month folders found under: {RAW_ROOT}")

    print(f"Found {len(month_dirs)} month folders.")

    for month_dir in month_dirs:
        print(f"\n== {month_dir.name} ==")
        for p, day in iter_daily_files(month_dir):
            out_dir = OUT_ROOT / f"year={day[0:4]}" / f"month={day[4:6]}" / f"day={day[6:8]}"
            convert_day(p, day, out_dir)

    print("\nDone.")
    print(f"Parquet root: {OUT_ROOT}")
    print(f"Bad files log (if any): {BAD_FILES_LOG}")

if __name__ == "__main__":
    main()
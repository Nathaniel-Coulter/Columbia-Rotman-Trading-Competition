# true_first_passage_analyzer.py
from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

# --------------------------------------------------
# paths
# --------------------------------------------------
EVENTS_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\true_first_passage_analyzer"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENT_LEVEL = OUT_DIR / "true_first_passage_event_level.csv"
OUT_RULE_SUMMARY = OUT_DIR / "true_first_passage_rule_summary.csv"

# --------------------------------------------------
# settings
# --------------------------------------------------
HORIZON_S = 600
LEVELS_PTS = [2.0, 4.0, 6.0, 8.0]

BAND_FILTER = [1]
SIDE_FILTER = "lower"

MAX_WORKERS = 6

# --------------------------------------------------
# optional rule set
# set USE_RULES = False if you want full subset only
# --------------------------------------------------
USE_RULES = True

RULES = {
    "SINGLE_A2": (
        (pl.col("depth_near_ask") >= 127) &
        (pl.col("depth_near_ask") <= 182) &
        (pl.col("cancel_rate_ask") >= 3.8) &
        (pl.col("cancel_rate_ask") <= 6.8)
    ),

    "SINGLE_C1": (
        (pl.col("realized_vol_1m") >= 0.000533821) &
        (pl.col("realized_vol_1m") <= 0.000692973) &
        (pl.col("depth_mid_ask") >= 149) &
        (pl.col("depth_mid_ask") <= 210)
    ),

    "ALL2_A2_C1": (
        (
            (pl.col("depth_near_ask") >= 127) &
            (pl.col("depth_near_ask") <= 182) &
            (pl.col("cancel_rate_ask") >= 3.8) &
            (pl.col("cancel_rate_ask") <= 6.8)
        ) &
        (
            (pl.col("realized_vol_1m") >= 0.000533821) &
            (pl.col("realized_vol_1m") <= 0.000692973) &
            (pl.col("depth_mid_ask") >= 149) &
            (pl.col("depth_mid_ask") <= 210)
        )
    ),

    "VOTE_B1_C1_D1": (
        (
            (
                (pl.col("depth_mid_ask") >= 151) &
                (pl.col("depth_mid_ask") <= 210) &
                (pl.col("queue_imbalance_std_5s") >= 0.140757) &
                (pl.col("queue_imbalance_std_5s") <= 0.193991)
            ).cast(pl.Int8)
            +
            (
                (pl.col("realized_vol_1m") >= 0.000533821) &
                (pl.col("realized_vol_1m") <= 0.000692973) &
                (pl.col("depth_mid_ask") >= 149) &
                (pl.col("depth_mid_ask") <= 210)
            ).cast(pl.Int8)
            +
            (
                (pl.col("cancel_rate_ask") >= 3.8) &
                (pl.col("cancel_rate_ask") <= 6.8) &
                (pl.col("bid_depth_curvature") >= -9.75) &
                (pl.col("bid_depth_curvature") <= -6.875)
            ).cast(pl.Int8)
        ) >= 2
    ),

    "VOTE4_A2_B1_C1_E1": (
        (
            (
                (pl.col("depth_near_ask") >= 127) &
                (pl.col("depth_near_ask") <= 182) &
                (pl.col("cancel_rate_ask") >= 3.8) &
                (pl.col("cancel_rate_ask") <= 6.8)
            ).cast(pl.Int8)
            +
            (
                (pl.col("depth_mid_ask") >= 151) &
                (pl.col("depth_mid_ask") <= 210) &
                (pl.col("queue_imbalance_std_5s") >= 0.140757) &
                (pl.col("queue_imbalance_std_5s") <= 0.193991)
            ).cast(pl.Int8)
            +
            (
                (pl.col("realized_vol_1m") >= 0.000533821) &
                (pl.col("realized_vol_1m") <= 0.000692973) &
                (pl.col("depth_mid_ask") >= 149) &
                (pl.col("depth_mid_ask") <= 210)
            ).cast(pl.Int8)
            +
            (
                (pl.col("cancel_rate_ask") >= 3.8) &
                (pl.col("cancel_rate_ask") <= 6.8) &
                (pl.col("cancel_ratio") >= 0.0592705) &
                (pl.col("cancel_ratio") <= 0.0799373)
            ).cast(pl.Int8)
        ) >= 2
    ),
}

# --------------------------------------------------
# helpers
# --------------------------------------------------
def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def iter_day_parquets(root: Path):
    for ydir in sorted(root.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            for ddir in sorted(mdir.glob("day=*")):
                p = ddir / "part-000.parquet"
                if not p.exists():
                    continue
                year = int(ydir.name.split("=")[1])
                month = int(mdir.name.split("=")[1])
                day = int(ddir.name.split("=")[1])
                yield year, month, day, p


def build_day_to_path(root: Path) -> Dict[str, Path]:
    return {
        f"{y:04d}-{m:02d}-{d:02d}": p
        for (y, m, d, p) in iter_day_parquets(root)
    }


def load_trade_day(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)

    required = {"ts", "type", "price", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing trade cols: {missing}")

    return (
        df
        .filter(
            (pl.col("type") == 2) &
            pl.col("price").is_not_null()
        )
        .select(["ts", "price"])
        .sort("ts")
    )


def first_hit_time_s(
    ts_us: List[int],
    signed_path_pts: List[float],
    start_idx: int,
    end_idx: int,
    threshold_pts: float,
    t0_us: int,
) -> Optional[float]:
    for i in range(start_idx, end_idx):
        if signed_path_pts[i] >= threshold_pts:
            return (ts_us[i] - t0_us) / 1_000_000.0
    return None


def process_day(
    day_ymd: str,
    full_path_str: Optional[str],
    df_day: pl.DataFrame,
) -> List[dict]:
    out_rows: List[dict] = []

    if full_path_str is None:
        return out_rows

    full_path = Path(full_path_str)
    if not full_path.exists():
        return out_rows

    df_tr = load_trade_day(full_path)
    if df_tr.height == 0:
        return out_rows

    tr_ts = df_tr["ts"].to_list()
    tr_px = [float(x) for x in df_tr["price"].to_list()]
    tr_us = [dt_to_us(x) for x in tr_ts]

    for r in df_day.to_dicts():
        t0: datetime = r["ts"]
        t0_us = dt_to_us(t0)
        entry_price = float(r["entry_price"])
        side = r["side"]
        rule_name = r["rule_name"]
        event_id = r["event_id"]

        # favorable direction:
        # lower-band touch -> favorable is upward
        # upper-band touch -> favorable is downward
        direction = 1.0 if side == "lower" else -1.0

        i0 = bisect_left(tr_us, t0_us)
        i1 = bisect_right(tr_us, t0_us + int(HORIZON_S * 1_000_000))

        if i0 >= i1:
            continue

        signed_fav = [(px - entry_price) * direction for px in tr_px]
        signed_adv = [(entry_price - px) * direction for px in tr_px]  # same convention: positive means adverse

        row = {
            "event_id": event_id,
            "date_ymd": day_ymd,
            "ts": t0,
            "rule_name": rule_name,
            "side": side,
            "band_k": r["band_k"],
            "entry_price": entry_price,
        }

        for lvl in LEVELS_PTS:
            t_fav = first_hit_time_s(tr_us, signed_fav, i0, i1, lvl, t0_us)
            t_adv = first_hit_time_s(tr_us, signed_adv, i0, i1, lvl, t0_us)

            row[f"t_fav_{int(lvl)}pt_s"] = t_fav
            row[f"t_adv_{int(lvl)}pt_s"] = t_adv

            row[f"fav_hit_{int(lvl)}pt"] = int(t_fav is not None)
            row[f"adv_hit_{int(lvl)}pt"] = int(t_adv is not None)

            row[f"fav_before_adv_{int(lvl)}pt"] = int(
                (t_fav is not None) and (t_adv is not None) and (t_fav < t_adv)
            )
            row[f"adv_before_fav_{int(lvl)}pt"] = int(
                (t_fav is not None) and (t_adv is not None) and (t_adv < t_fav)
            )
            row[f"fav_only_{int(lvl)}pt"] = int(
                (t_fav is not None) and (t_adv is None)
            )
            row[f"adv_only_{int(lvl)}pt"] = int(
                (t_fav is None) and (t_adv is not None)
            )
            row[f"both_hit_{int(lvl)}pt"] = int(
                (t_fav is not None) and (t_adv is not None)
            )
            row[f"neither_hit_{int(lvl)}pt"] = int(
                (t_fav is None) and (t_adv is None)
            )

        out_rows.append(row)

    return out_rows


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing events parquet: {EVENTS_PATH}")
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(f"Missing raw tick root: {FULL_TICK_ROOT}")

    df = pl.read_parquet(EVENTS_PATH)

    df = df.filter(
        pl.col("band_k").is_in(BAND_FILTER) &
        (pl.col("side") == SIDE_FILTER) &
        pl.col("entry_price").is_not_null()
    ).sort("ts")

    if USE_RULES:
        pieces = []
        for rule_name, expr in RULES.items():
            dfr = df.filter(expr).with_columns(pl.lit(rule_name).alias("rule_name"))
            pieces.append(dfr)
        df = pl.concat(pieces) if pieces else pl.DataFrame()

    if df.height == 0:
        raise ValueError("No rows left after rule/subset filtering.")

    print("\n--------------------------------------")
    print("True First-Passage Analyzer")
    print("--------------------------------------")
    print(f"Rows analyzed: {df.height:,}")
    print(f"HORIZON_S     = {HORIZON_S}")
    print(f"BAND_FILTER   = {BAND_FILTER}")
    print(f"SIDE_FILTER   = {SIDE_FILTER}")
    print(f"LEVELS_PTS    = {LEVELS_PTS}")
    print(f"USE_RULES     = {USE_RULES}")

    day_to_full = build_day_to_path(FULL_TICK_ROOT)

    day_jobs = []
    for day in sorted(df["date_ymd"].unique().to_list()):
        dfd = df.filter(pl.col("date_ymd") == day).sort("ts")
        full_path = day_to_full.get(day)
        day_jobs.append((day, None if full_path is None else str(full_path), dfd))

    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(process_day, day, full_path, dfd)
            for day, full_path, dfd in day_jobs
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.extend(fut.result())
            if i % 25 == 0:
                print(f"...processed {i} days")

    if not rows:
        raise ValueError("No event-level first-passage rows produced.")

    df_out = pl.from_dicts(rows).sort(["rule_name", "ts"])
    df_out.write_csv(OUT_EVENT_LEVEL)

    # summarize by rule and level
    summary_rows = []
    for rule_name in sorted(df_out["rule_name"].unique().to_list()):
        dfr = df_out.filter(pl.col("rule_name") == rule_name)

        for lvl in LEVELS_PTS:
            k = int(lvl)
            summary = dfr.select([
                pl.len().alias("n_events"),
                pl.col(f"fav_hit_{k}pt").mean().alias("fav_hit_rate"),
                pl.col(f"adv_hit_{k}pt").mean().alias("adv_hit_rate"),
                pl.col(f"fav_before_adv_{k}pt").mean().alias("fav_before_adv_rate"),
                pl.col(f"adv_before_fav_{k}pt").mean().alias("adv_before_fav_rate"),
                pl.col(f"fav_only_{k}pt").mean().alias("fav_only_rate"),
                pl.col(f"adv_only_{k}pt").mean().alias("adv_only_rate"),
                pl.col(f"both_hit_{k}pt").mean().alias("both_hit_rate"),
                pl.col(f"neither_hit_{k}pt").mean().alias("neither_hit_rate"),
                pl.col(f"t_fav_{k}pt_s").drop_nulls().mean().alias("mean_t_fav_s"),
                pl.col(f"t_adv_{k}pt_s").drop_nulls().mean().alias("mean_t_adv_s"),
            ]).to_dicts()[0]

            summary["rule_name"] = rule_name
            summary["level_pts"] = lvl
            summary["path_edge"] = (
                (summary["fav_before_adv_rate"] or 0.0) -
                (summary["adv_before_fav_rate"] or 0.0)
            )
            summary_rows.append(summary)

    df_summary = pl.from_dicts(summary_rows).sort(
        ["level_pts", "path_edge", "fav_before_adv_rate"],
        descending=[False, True, True]
    )
    df_summary.write_csv(OUT_RULE_SUMMARY)

    print("\nDone.")
    print(f"Wrote: {OUT_EVENT_LEVEL}")
    print(f"Wrote: {OUT_RULE_SUMMARY}")

    print("\nTop summary rows:")
    print(df_summary.head(20))


if __name__ == "__main__":
    main()
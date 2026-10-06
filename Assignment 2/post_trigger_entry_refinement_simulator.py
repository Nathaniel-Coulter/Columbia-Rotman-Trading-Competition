# post_trigger_entry_refinement_simulator.py
from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import math
import pandas as pd
import polars as pl


# ============================================================
# Paths
# ============================================================
FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\post_trigger_entry_refinement_simulator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "post_trigger_entry_event_level.csv"
SUMMARY_OUT = OUTPUT_DIR / "post_trigger_entry_summary.csv"


# ============================================================
# Research settings
# ============================================================
HORIZON_S = 600

BAND_FILTER = [1]
SIDE_FILTER = "upper"   # "lower" or "upper"

# Entry refinement:
# 0.0 = enter immediately at event price
# 1.0 = wait until path goes 1 point adverse, then enter
# 2.0 = wait until path goes 2 points adverse, then enter
ENTRY_ADVERSE_LEVELS_PTS = [0.0, 1.0, 2.0, 3.0, 4.0]

# Diagnostic post-entry targets (no forced exit; just did these happen after refined entry?)
FAVORABLE_TARGETS_PTS = [1.0, 2.0, 4.0, 6.0, 8.0]

# Diagnostic adverse thresholds after refined entry
ADVERSE_LEVELS_PTS = [2.0, 4.0, 6.0, 8.0, 10.0]

# Optional safety cap on how far into future to scan
# Usually same as horizon, but leave separate in case you want to shorten later
POST_ENTRY_MAX_SECONDS = HORIZON_S


# ============================================================
# Candidate rules / combos
# These are based on the strongest cells from your extractor.
# ============================================================
BASE_RULES: Dict[str, Dict[str, tuple[float, float]]] = {
    "A2": {
        "depth_near_ask": (4.0, 128.0),
        "cancel_rate_ask": (6.6, 427.0),
    },
    "B2": {
        "realized_vol_1m": (0.000682729, 0.00677785),
        "depth_mid_ask": (7.0, 150.0),
    },
    "C1": {
        "cancel_rate_ask": (6.6, 427.0),
        "cancel_ratio": (0.0809026, 0.495652),
    },
    "C2": {
        "cancel_rate_ask": (6.6, 427.0),
        "bid_depth_curvature": (-6.625, 11.5),
    },
    "D1": {
        "depth_mid_ask": (7.0, 151.0),
        "queue_imbalance_std_5s": (0.195659, 0.645851),
    },
    "D2": {
        "depth_near_ask": (4.0, 128.0),
        "queue_imbalance_std_5s": (0.195659, 0.645851),
    },
}

# Use whichever shortlist you want here
COMBO_RULES = [
    "SINGLE_C1",
    "SINGLE_C2",
    "SINGLE_B2",
    "SINGLE_A2",
    "ALL2_B2_C2",
    "ALL2_A2_C2",
    "ALL2_C1_C2",
    "UNION_A2_C2",
]


# ============================================================
# Helpers
# ============================================================
def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def iter_day_parquets(root: Path):
    for ydir in sorted(root.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            for ddir in sorted(mdir.glob("day=*")):
                p = ddir / "part-000.parquet"
                if p.exists():
                    y = int(ydir.name.split("=")[1])
                    m = int(mdir.name.split("=")[1])
                    d = int(ddir.name.split("=")[1])
                    yield f"{y:04d}-{m:02d}-{d:02d}", p


def build_day_to_path(root: Path) -> Dict[str, Path]:
    return {day: p for day, p in iter_day_parquets(root)}


def load_trade_day(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    required = {"ts", "type", "price", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required trade cols in {path}: {missing}")

    return (
        df.filter(
            (pl.col("type") == 2)
            & pl.col("price").is_not_null()
            & pl.col("volume").is_not_null()
        )
        .select(["ts", "price", "volume"])
        .sort("ts")
    )


def between_expr(col_name: str, lo: float, hi: float) -> pl.Expr:
    return (
        pl.col(col_name).is_not_null()
        & (pl.col(col_name) >= lo)
        & (pl.col(col_name) <= hi)
    )


def add_base_rule_flags(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for rule_name, conds in BASE_RULES.items():
        expr = None
        for feat, (lo, hi) in conds.items():
            e = between_expr(feat, lo, hi)
            expr = e if expr is None else (expr & e)
        exprs.append(expr.cast(pl.Int8).alias(f"flag_{rule_name}"))
    return df.with_columns(exprs)


def add_combo_rule_flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("flag_C1").alias("flag_SINGLE_C1"),
        pl.col("flag_C2").alias("flag_SINGLE_C2"),
        pl.col("flag_B2").alias("flag_SINGLE_B2"),
        pl.col("flag_A2").alias("flag_SINGLE_A2"),

        (pl.col("flag_B2") & pl.col("flag_C2")).cast(pl.Int8).alias("flag_ALL2_B2_C2"),
        (pl.col("flag_A2") & pl.col("flag_C2")).cast(pl.Int8).alias("flag_ALL2_A2_C2"),
        (pl.col("flag_C1") & pl.col("flag_C2")).cast(pl.Int8).alias("flag_ALL2_C1_C2"),

        ((pl.col("flag_A2") + pl.col("flag_C2")) >= 1).cast(pl.Int8).alias("flag_UNION_A2_C2"),
    ])


def first_index_where(xs: List[float], threshold: float, mode: str) -> Optional[int]:
    if mode == "ge":
        for i, x in enumerate(xs):
            if x >= threshold:
                return i
    elif mode == "le":
        for i, x in enumerate(xs):
            if x <= threshold:
                return i
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return None


# ============================================================
# Main simulation logic
# ============================================================
def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(FULL_TICK_ROOT)

    print("\n--------------------------------------")
    print("Post-Trigger Entry Refinement Simulator")
    print("--------------------------------------")
    print(f"HORIZON_S                = {HORIZON_S}")
    print(f"BAND_FILTER              = {BAND_FILTER}")
    print(f"SIDE_FILTER              = {SIDE_FILTER}")
    print(f"ENTRY_ADVERSE_LEVELS_PTS = {ENTRY_ADVERSE_LEVELS_PTS}")
    print(f"FAVORABLE_TARGETS_PTS    = {FAVORABLE_TARGETS_PTS}")
    print(f"ADVERSE_LEVELS_PTS       = {ADVERSE_LEVELS_PTS}")
    print(f"COMBO_RULES              = {COMBO_RULES}")

    # --- load features
    df = pl.read_parquet(FEATURES_PATH)

    # base subset
    df = df.filter(
        pl.col("entry_price").is_not_null()
        & pl.col("ts").is_not_null()
    )

    if BAND_FILTER is not None:
        df = df.filter(pl.col("band_k").is_in(BAND_FILTER))
    if SIDE_FILTER is not None:
        df = df.filter(pl.col("side") == SIDE_FILTER)

    df = add_base_rule_flags(df)
    df = add_combo_rule_flags(df)

    combo_flag_cols = [f"flag_{x}" for x in COMBO_RULES]
    keep_cols = [
        "event_id", "date_ymd", "ts", "entry_price", "side", "band_k",
        *combo_flag_cols
    ]
    df = df.select(keep_cols).sort(["date_ymd", "ts"])

    if df.height == 0:
        raise ValueError("No events left after filtering.")

    print(f"Rows analyzed: {df.height:,}")

    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    rows_out: List[Dict[str, Any]] = []

    event_days = df["date_ymd"].unique().sort().to_list()
    for i, day in enumerate(event_days, start=1):
        if i % 25 == 0:
            print(f"...processed {i} days")

        path = day_to_path.get(day)
        if path is None or not path.exists():
            continue

        df_day = df.filter(pl.col("date_ymd") == day).sort("ts")
        if df_day.height == 0:
            continue

        df_tr = load_trade_day(path)
        if df_tr.height == 0:
            continue

        tr_ts = df_tr["ts"].to_list()
        tr_us = [dt_to_us(x) for x in tr_ts]
        tr_px = [float(x) for x in df_tr["price"].to_list()]

        for r in df_day.to_dicts():
            t0: datetime = r["ts"]
            t0_us = dt_to_us(t0)
            event_px = float(r["entry_price"])
            side = r["side"]

            # direction: lower band = long reversion, upper band = short reversion
            direction = 1.0 if side == "lower" else -1.0

            # future trade path from event -> horizon
            start_i = bisect_left(tr_us, t0_us)
            end_us = t0_us + int(HORIZON_S * 1_000_000)
            end_i = bisect_right(tr_us, end_us)

            if start_i >= end_i:
                continue

            path_us = tr_us[start_i:end_i]
            path_px = tr_px[start_i:end_i]

            # signed move from event price
            signed_from_event = [(px - event_px) * direction for px in path_px]

            for combo_name in COMBO_RULES:
                if int(r[f"flag_{combo_name}"]) != 1:
                    continue

                for adverse_entry_pts in ENTRY_ADVERSE_LEVELS_PTS:
                    # immediate entry
                    if adverse_entry_pts <= 0.0:
                        entry_idx = 0
                        entry_time_us = path_us[0]
                        entry_px = event_px
                        entry_delay_s = (entry_time_us - t0_us) / 1_000_000.0
                    else:
                        # wait until price goes adverse by X points from event price
                        # adverse means signed move <= -adverse_entry_pts
                        j = first_index_where(
                            signed_from_event,
                            -adverse_entry_pts,
                            mode="le",
                        )
                        if j is None:
                            rows_out.append({
                                "event_id": r["event_id"],
                                "date_ymd": r["date_ymd"],
                                "ts": r["ts"],
                                "rule_name": combo_name,
                                "entry_adverse_pts": adverse_entry_pts,
                                "entry_triggered": 0,
                                "entry_delay_s": None,
                                "entry_px": None,
                                "entry_move_from_event_pts": None,
                                "post_entry_mfe_pts": None,
                                "post_entry_mae_pts": None,
                                "post_entry_end_ret_pts": None,
                            })
                            continue
                        entry_idx = j
                        entry_time_us = path_us[j]
                        entry_px = path_px[j]
                        entry_delay_s = (entry_time_us - t0_us) / 1_000_000.0

                    # remaining post-entry path
                    post_us = path_us[entry_idx:]
                    post_px = path_px[entry_idx:]
                    if not post_px:
                        continue

                    # optional post-entry cap by time (currently same as horizon)
                    post_end_us = entry_time_us + int(POST_ENTRY_MAX_SECONDS * 1_000_000)
                    cap_i = bisect_right(post_us, post_end_us)
                    post_us = post_us[:cap_i]
                    post_px = post_px[:cap_i]
                    if not post_px:
                        continue

                    signed_from_entry = [(px - entry_px) * direction for px in post_px]

                    post_entry_mfe_pts = max(signed_from_entry)
                    post_entry_mae_pts = max(0.0, -min(signed_from_entry))
                    post_entry_end_ret_pts = signed_from_entry[-1]
                    entry_move_from_event_pts = (entry_px - event_px) * direction

                    row = {
                        "event_id": r["event_id"],
                        "date_ymd": r["date_ymd"],
                        "ts": r["ts"],
                        "rule_name": combo_name,
                        "entry_adverse_pts": adverse_entry_pts,
                        "entry_triggered": 1,
                        "entry_delay_s": entry_delay_s,
                        "entry_px": entry_px,
                        "entry_move_from_event_pts": entry_move_from_event_pts,
                        "post_entry_mfe_pts": post_entry_mfe_pts,
                        "post_entry_mae_pts": post_entry_mae_pts,
                        "post_entry_end_ret_pts": post_entry_end_ret_pts,
                    }

                    for tgt in FAVORABLE_TARGETS_PTS:
                        row[f"hit_fav_{str(tgt).replace('.', '_')}pt"] = int(
                            post_entry_mfe_pts >= tgt
                        )

                    for adv in ADVERSE_LEVELS_PTS:
                        row[f"hit_adv_{str(adv).replace('.', '_')}pt"] = int(
                            post_entry_mae_pts >= adv
                        )

                    rows_out.append(row)

    if not rows_out:
        raise ValueError("No output rows produced.")

    pdf = pd.DataFrame(rows_out)
    pdf.to_csv(EVENT_LEVEL_OUT, index=False)

    # ========================================================
    # Summary
    # ========================================================
    entered = pdf[pdf["entry_triggered"] == 1].copy()

    summary_rows = []
    fav_cols = [f"hit_fav_{str(x).replace('.', '_')}pt" for x in FAVORABLE_TARGETS_PTS]
    adv_cols = [f"hit_adv_{str(x).replace('.', '_')}pt" for x in ADVERSE_LEVELS_PTS]

    for (rule_name, entry_adverse_pts), g_all in pdf.groupby(["rule_name", "entry_adverse_pts"], dropna=False):
        g = entered[
            (entered["rule_name"] == rule_name)
            & (entered["entry_adverse_pts"] == entry_adverse_pts)
        ].copy()

        n_total_candidates = len(g_all)
        n_entered = len(g)
        enter_rate = n_entered / n_total_candidates if n_total_candidates > 0 else math.nan

        if n_entered == 0:
            summary_rows.append({
                "rule_name": rule_name,
                "entry_adverse_pts": entry_adverse_pts,
                "n_total_candidates": n_total_candidates,
                "n_entered": 0,
                "enter_rate": enter_rate,
            })
            continue

        row = {
            "rule_name": rule_name,
            "entry_adverse_pts": entry_adverse_pts,
            "n_total_candidates": n_total_candidates,
            "n_entered": n_entered,
            "enter_rate": enter_rate,
            "mean_entry_delay_s": g["entry_delay_s"].mean(),
            "median_entry_delay_s": g["entry_delay_s"].median(),
            "mean_entry_move_from_event_pts": g["entry_move_from_event_pts"].mean(),
            "mean_post_entry_mfe_pts": g["post_entry_mfe_pts"].mean(),
            "median_post_entry_mfe_pts": g["post_entry_mfe_pts"].median(),
            "mean_post_entry_mae_pts": g["post_entry_mae_pts"].mean(),
            "median_post_entry_mae_pts": g["post_entry_mae_pts"].median(),
            "mean_post_entry_end_ret_pts": g["post_entry_end_ret_pts"].mean(),
            "median_post_entry_end_ret_pts": g["post_entry_end_ret_pts"].median(),
        }

        for c in fav_cols:
            row[c] = g[c].mean()

        for c in adv_cols:
            row[c] = g[c].mean()

        # crude score favoring high 4pt/6pt hit and low post-entry MAE
        fav4 = row.get("hit_fav_4_0pt", 0.0)
        fav6 = row.get("hit_fav_6_0pt", 0.0)
        adv6 = row.get("hit_adv_6_0pt", 0.0)
        row["refinement_score"] = (
            0.45 * fav4
            + 0.35 * fav6
            + 0.15 * max(0.0, row["mean_post_entry_mfe_pts"])
            - 0.20 * max(0.0, row["mean_post_entry_mae_pts"])
            - 0.15 * adv6
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["rule_name", "entry_adverse_pts"],
        ascending=[True, True]
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    display_cols = [
        "rule_name",
        "entry_adverse_pts",
        "n_entered",
        "enter_rate",
        "mean_entry_delay_s",
        "mean_post_entry_mfe_pts",
        "mean_post_entry_mae_pts",
        "mean_post_entry_end_ret_pts",
        "hit_fav_2_0pt",
        "hit_fav_4_0pt",
        "hit_fav_6_0pt",
        "hit_adv_4_0pt",
        "hit_adv_6_0pt",
        "refinement_score",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]

    print("\nTop refinement rows:")
    print(
        summary.sort_values(
            ["refinement_score", "n_entered"],
            ascending=[False, False]
        )[display_cols].head(25).to_string(index=False)
    )


if __name__ == "__main__":
    main()
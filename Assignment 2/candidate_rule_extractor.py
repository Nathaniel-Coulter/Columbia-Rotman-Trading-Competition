# candidate_rule_extractor.py
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\candidate_rule_extractor"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL_CELLS_CSV = OUTPUT_DIR / "candidate_rule_cells.csv"
OUT_TOP_RULES_CSV = OUTPUT_DIR / "candidate_top_rules.csv"


# --------------------------------------------------
# Fixed subset settings
# --------------------------------------------------
TARGET_COL = "mr_pts_t600"
THRESHOLD = 1.0

BAND_FILTER = [1]
SIDE_FILTER = "upper"

MIN_VOL_DECILE = None
VOL_COL_FOR_FILTER = "realized_volatility_30s"

# --------------------------------------------------
# Pair shortlist from your best results
# --------------------------------------------------
PAIR_LIST: List[Tuple[str, str]] = [
    ("depth_near_ask", "queue_imbalance_std_5s"),
    ("queue_imbalance_std_5s", "bid_depth_curvature"),
    ("depth_mid_ask", "queue_imbalance_std_5s"),
    ("depth_mid_ask", "cancel_rate_ask"),
    ("realized_vol_1m", "depth_mid_ask"),
    ("depth_near_ask", "cancel_rate_ask"),
    ("cancel_rate_ask", "cancel_ratio"),
    ("realized_volatility_30s", "depth_mid_ask"),
    ("cancel_rate_ask", "bid_depth_curvature"),
    ("queue_imbalance_std_5s", "cancel_ratio"),
]

# --------------------------------------------------
# bin settings
# --------------------------------------------------
N_BINS_X = 5
N_BINS_Y = 5

# --------------------------------------------------
# rule-quality filters
# tune these later if needed
# --------------------------------------------------
MIN_CELL_COUNT = 40
MIN_HIT_RATE = 0.85
MIN_AVG_MR_PTS = 6.0

# ranking
TOP_N_RULES = 50


def add_target(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col(TARGET_COL) >= THRESHOLD).cast(pl.Int8).alias("target")
    )


def apply_subset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col(TARGET_COL).is_not_null())

    if BAND_FILTER is not None:
        df = df.filter(pl.col("band_k").is_in(BAND_FILTER))

    if SIDE_FILTER is not None:
        df = df.filter(pl.col("side") == SIDE_FILTER)

    return df


def apply_vol_filter(df: pl.DataFrame) -> pl.DataFrame:
    if MIN_VOL_DECILE is None:
        return df

    df = df.filter(pl.col(VOL_COL_FOR_FILTER).is_not_null())
    if df.height == 0:
        return df

    n_tmp = df.height
    df = df.with_columns(
        pl.col(VOL_COL_FOR_FILTER).rank("dense").alias("vol_rank_tmp")
    )
    df = df.with_columns(
        ((pl.col("vol_rank_tmp") / n_tmp) * 10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int64)
        .alias("vol_decile_tmp")
    )
    df = df.filter(pl.col("vol_decile_tmp") >= MIN_VOL_DECILE)
    return df.drop(["vol_rank_tmp", "vol_decile_tmp"])


def make_equal_count_bin_expr(col_name: str, n_rows: int, n_bins: int, out_name: str) -> pl.Expr:
    return (
        (((pl.col(col_name).rank("ordinal") - 1) / n_rows) * n_bins)
        .floor()
        .clip(0, n_bins - 1)
        .cast(pl.Int64)
        .alias(out_name)
    )


def summarize_pair(df: pl.DataFrame, feature_x: str, feature_y: str) -> pd.DataFrame:
    tmp = df.filter(
        pl.col(feature_x).is_not_null() &
        pl.col(feature_y).is_not_null()
    ).select([feature_x, feature_y, TARGET_COL, "target"])

    n = tmp.height
    if n == 0:
        return pd.DataFrame()

    tmp = tmp.with_columns([
        make_equal_count_bin_expr(feature_x, n, N_BINS_X, "x_bin"),
        make_equal_count_bin_expr(feature_y, n, N_BINS_Y, "y_bin"),
    ])

    out = (
        tmp.group_by(["x_bin", "y_bin"])
        .agg([
            pl.len().alias("n_events"),
            pl.col("target").mean().alias("hit_rate"),
            pl.col(TARGET_COL).mean().alias("avg_mr_pts"),

            pl.col(feature_x).min().alias("x_min"),
            pl.col(feature_x).max().alias("x_max"),
            pl.col(feature_x).median().alias("x_median"),

            pl.col(feature_y).min().alias("y_min"),
            pl.col(feature_y).max().alias("y_max"),
            pl.col(feature_y).median().alias("y_median"),
        ])
        .sort(["x_bin", "y_bin"])
        .with_columns([
            pl.lit(feature_x).alias("feature_x"),
            pl.lit(feature_y).alias("feature_y"),
        ])
        .select([
            "feature_x",
            "feature_y",
            "x_bin",
            "y_bin",
            "n_events",
            "hit_rate",
            "avg_mr_pts",
            "x_min",
            "x_max",
            "x_median",
            "y_min",
            "y_max",
            "y_median",
        ])
    )

    return out.to_pandas()


def add_rule_scores(df_cells: pd.DataFrame) -> pd.DataFrame:
    df_cells = df_cells.copy()

    # normalize event count so larger cells get some reward
    count_norm = df_cells["n_events"] / df_cells["n_events"].max()

    # simple composite score
    df_cells["rule_score"] = (
        0.45 * df_cells["hit_rate"]
        + 0.35 * np.minimum(df_cells["avg_mr_pts"] / 10.0, 1.5)
        + 0.20 * count_norm
    )

    # stricter score
    df_cells["rule_score_conservative"] = (
        0.50 * df_cells["hit_rate"]
        + 0.20 * np.minimum(df_cells["avg_mr_pts"] / 10.0, 1.5)
        + 0.30 * count_norm
    )

    return df_cells


def format_rule_text(row: pd.Series) -> str:
    return (
        f"{row['feature_x']} in [{row['x_min']:.6g}, {row['x_max']:.6g}] "
        f"AND {row['feature_y']} in [{row['y_min']:.6g}, {row['y_max']:.6g}]"
    )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FEATURES_PATH}")

    required_cols = [
        "band_k", "side", TARGET_COL, VOL_COL_FOR_FILTER, "target"
    ] + sorted({c for pair in PAIR_LIST for c in pair})

    df = pl.read_parquet(FEATURES_PATH)
    df = apply_subset(df)
    df = add_target(df)
    df = apply_vol_filter(df)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.height == 0:
        raise ValueError("No rows left after filtering.")

    print("\n--------------------------------------")
    print("Candidate Rule Extractor")
    print("--------------------------------------")
    print(f"Rows analyzed: {df.height}")
    print(f"TARGET_COL       = {TARGET_COL}")
    print(f"THRESHOLD        = {THRESHOLD}")
    print(f"BAND_FILTER      = {BAND_FILTER}")
    print(f"SIDE_FILTER      = {SIDE_FILTER}")
    print(f"MIN_VOL_DECILE   = {MIN_VOL_DECILE}")
    print(f"N_PAIRS          = {len(PAIR_LIST)}")
    print(f"MIN_CELL_COUNT   = {MIN_CELL_COUNT}")
    print(f"MIN_HIT_RATE     = {MIN_HIT_RATE}")
    print(f"MIN_AVG_MR_PTS   = {MIN_AVG_MR_PTS}")

    all_cells: List[pd.DataFrame] = []

    for i, (fx, fy) in enumerate(PAIR_LIST, start=1):
        print(f"[{i}/{len(PAIR_LIST)}] {fx}  x  {fy}")
        out = summarize_pair(df, fx, fy)
        if not out.empty:
            all_cells.append(out)

    if not all_cells:
        raise ValueError("No 2D cell results were produced.")

    df_cells = pd.concat(all_cells, ignore_index=True)
    df_cells = add_rule_scores(df_cells)

    # create human-readable rule text
    df_cells["rule_text"] = df_cells.apply(format_rule_text, axis=1)

    # filter candidate rules
    df_rules = df_cells[
        (df_cells["n_events"] >= MIN_CELL_COUNT) &
        (df_cells["hit_rate"] >= MIN_HIT_RATE) &
        (df_cells["avg_mr_pts"] >= MIN_AVG_MR_PTS)
    ].copy()

    df_rules = df_rules.sort_values(
        ["rule_score", "hit_rate", "avg_mr_pts", "n_events"],
        ascending=[False, False, False, False],
        kind="stable",
    )

    df_cells.to_csv(OUT_ALL_CELLS_CSV, index=False)
    df_rules.to_csv(OUT_TOP_RULES_CSV, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_ALL_CELLS_CSV}")
    print(f"Wrote: {OUT_TOP_RULES_CSV}")

    print("\nTop candidate rules:")
    if df_rules.empty:
        print("No rules passed current thresholds.")
    else:
        show_cols = [
            "feature_x",
            "feature_y",
            "x_bin",
            "y_bin",
            "n_events",
            "hit_rate",
            "avg_mr_pts",
            "rule_score",
            "rule_score_conservative",
            "rule_text",
        ]
        print(df_rules[show_cols].head(TOP_N_RULES).to_string(index=False))


if __name__ == "__main__":
    main()
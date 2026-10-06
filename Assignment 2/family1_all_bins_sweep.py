# family1_all_bins_sweep.py
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\family1_all_bins_sweep"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_1D_CSV = OUTPUT_DIR / "all_1d_bins.csv"
OUT_2D_CSV = OUTPUT_DIR / "all_2d_bins.csv"
OUT_1D_SUMMARY_CSV = OUTPUT_DIR / "all_1d_feature_summary.csv"
OUT_2D_SUMMARY_CSV = OUTPUT_DIR / "all_2d_pair_summary.csv"


# --------------------------------------------------
# Fixed subset settings
# --------------------------------------------------
TARGET_COL = "mr_pts_t600"
THRESHOLD = 1.0

BAND_FILTER = [1]
SIDE_FILTER = "lower"

# optional regime filter
MIN_VOL_DECILE = None
VOL_COL_FOR_FILTER = "realized_volatility_30s"

# bin counts
N_BINS_1D = 10
N_BINS_2D_X = 5
N_BINS_2D_Y = 5

MIN_CELL_COUNT_2D = 15

FEATURE_LIST: List[str] = [
    "distance_from_overnight_high",
    "realized_volatility_30s",
    "realized_vol_1m",
    "depth_mid_ask",
    "depth_near_ask",
    "queue_imbalance_std_5s",
    "cancel_rate_ask",
    "cancel_ratio",
    "z_anchor_vwap",
    "distance_from_session_vwap_anchor",
    "bid_depth_curvature",
    "trade_imbalance_60s",
    "wall_persistence_2s",
    "convexity_imbalance",
    "liquidity_compression",
    "replenishment_rate_near_5s",
    "impact_asymmetry_10s",
    "flow_impact",
    "absorption_ratio",
    "microprice_pressure",
    "cvd_acceleration",
    "trade_sign_autocorr_60s",
]


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


def summarize_1d(df: pl.DataFrame, feature: str) -> pd.DataFrame:
    tmp = df.filter(pl.col(feature).is_not_null()).select([feature, TARGET_COL, "target"])

    n = tmp.height
    if n == 0:
        return pd.DataFrame()

    tmp = tmp.with_columns(
        make_equal_count_bin_expr(feature, n, N_BINS_1D, "feature_bin")
    )

    out = (
        tmp.group_by("feature_bin")
        .agg([
            pl.len().alias("n_events"),
            pl.col("target").mean().alias("hit_rate"),
            pl.col(TARGET_COL).mean().alias("avg_mr_pts"),
            pl.col(feature).min().alias("feature_min"),
            pl.col(feature).max().alias("feature_max"),
            pl.col(feature).median().alias("feature_median"),
        ])
        .sort("feature_bin")
        .with_columns(pl.lit(feature).alias("feature"))
        .select([
            "feature",
            "feature_bin",
            "n_events",
            "hit_rate",
            "avg_mr_pts",
            "feature_min",
            "feature_max",
            "feature_median",
        ])
    )
    return out.to_pandas()


def summarize_2d(df: pl.DataFrame, feature_x: str, feature_y: str) -> pd.DataFrame:
    tmp = df.filter(
        pl.col(feature_x).is_not_null() &
        pl.col(feature_y).is_not_null()
    ).select([feature_x, feature_y, TARGET_COL, "target"])

    n = tmp.height
    if n == 0:
        return pd.DataFrame()

    tmp = tmp.with_columns([
        make_equal_count_bin_expr(feature_x, n, N_BINS_2D_X, "x_bin"),
        make_equal_count_bin_expr(feature_y, n, N_BINS_2D_Y, "y_bin"),
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


def build_1d_feature_summary(df_1d: pd.DataFrame) -> pd.DataFrame:
    if df_1d.empty:
        return df_1d

    grp = df_1d.groupby("feature", dropna=False)

    summary = grp.apply(
        lambda g: pd.Series({
            "n_bins": len(g),
            "total_events": g["n_events"].sum(),
            "min_hit_rate": g["hit_rate"].min(),
            "max_hit_rate": g["hit_rate"].max(),
            "hit_rate_range": g["hit_rate"].max() - g["hit_rate"].min(),
            "min_avg_mr_pts": g["avg_mr_pts"].min(),
            "max_avg_mr_pts": g["avg_mr_pts"].max(),
            "avg_mr_pts_range": g["avg_mr_pts"].max() - g["avg_mr_pts"].min(),
            "best_bin_hit_rate": g.loc[g["hit_rate"].idxmax(), "feature_bin"],
            "best_bin_avg_mr_pts": g.loc[g["avg_mr_pts"].idxmax(), "feature_bin"],
            "best_hit_rate_events": g.loc[g["hit_rate"].idxmax(), "n_events"],
        })
    ).reset_index()

    summary = summary.sort_values(
        ["hit_rate_range", "avg_mr_pts_range", "max_hit_rate"],
        ascending=[False, False, False],
        kind="stable",
    )
    return summary


def build_2d_pair_summary(df_2d: pd.DataFrame) -> pd.DataFrame:
    if df_2d.empty:
        return df_2d

    grp = df_2d.groupby(["feature_x", "feature_y"], dropna=False)

    def agg_pair(g: pd.DataFrame) -> pd.Series:
        g_ok = g[g["n_events"] >= MIN_CELL_COUNT_2D].copy()
        if g_ok.empty:
            return pd.Series({
                "n_cells": len(g),
                "n_cells_ge_min": 0,
                "total_events": g["n_events"].sum(),
                "max_hit_rate": np.nan,
                "min_hit_rate": np.nan,
                "hit_rate_range": np.nan,
                "max_avg_mr_pts": np.nan,
                "min_avg_mr_pts": np.nan,
                "avg_mr_pts_range": np.nan,
                "best_x_bin_hit_rate": np.nan,
                "best_y_bin_hit_rate": np.nan,
                "best_x_bin_avg_mr": np.nan,
                "best_y_bin_avg_mr": np.nan,
                "best_hit_rate_events": np.nan,
            })

        best_hit_idx = g_ok["hit_rate"].idxmax()
        best_mr_idx = g_ok["avg_mr_pts"].idxmax()

        return pd.Series({
            "n_cells": len(g),
            "n_cells_ge_min": len(g_ok),
            "total_events": g["n_events"].sum(),
            "max_hit_rate": g_ok["hit_rate"].max(),
            "min_hit_rate": g_ok["hit_rate"].min(),
            "hit_rate_range": g_ok["hit_rate"].max() - g_ok["hit_rate"].min(),
            "max_avg_mr_pts": g_ok["avg_mr_pts"].max(),
            "min_avg_mr_pts": g_ok["avg_mr_pts"].min(),
            "avg_mr_pts_range": g_ok["avg_mr_pts"].max() - g_ok["avg_mr_pts"].min(),
            "best_x_bin_hit_rate": g_ok.loc[best_hit_idx, "x_bin"],
            "best_y_bin_hit_rate": g_ok.loc[best_hit_idx, "y_bin"],
            "best_x_bin_avg_mr": g_ok.loc[best_mr_idx, "x_bin"],
            "best_y_bin_avg_mr": g_ok.loc[best_mr_idx, "y_bin"],
            "best_hit_rate_events": g_ok.loc[best_hit_idx, "n_events"],
        })

    summary = grp.apply(agg_pair).reset_index()

    summary = summary.sort_values(
        ["hit_rate_range", "avg_mr_pts_range", "max_hit_rate"],
        ascending=[False, False, False],
        kind="stable",
    )
    return summary


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FEATURES_PATH}")

    required_cols = [
        "band_k", "side", TARGET_COL, VOL_COL_FOR_FILTER, "target"
    ] + FEATURE_LIST

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
    print("Family 1 all-feature 1D/2D bin sweep")
    print("--------------------------------------")
    print(f"Rows analyzed: {df.height}")
    print(f"TARGET_COL      = {TARGET_COL}")
    print(f"THRESHOLD       = {THRESHOLD}")
    print(f"BAND_FILTER     = {BAND_FILTER}")
    print(f"SIDE_FILTER     = {SIDE_FILTER}")
    print(f"MIN_VOL_DECILE  = {MIN_VOL_DECILE}")
    print(f"N_FEATURES      = {len(FEATURE_LIST)}")
    print(f"N_2D_PAIRS      = {len(list(combinations(FEATURE_LIST, 2)))}")

    # -----------------------------
    # 1D sweep
    # -----------------------------
    rows_1d: List[pd.DataFrame] = []
    for i, feat in enumerate(FEATURE_LIST, start=1):
        print(f"[1D {i}/{len(FEATURE_LIST)}] {feat}")
        out = summarize_1d(df, feat)
        if not out.empty:
            rows_1d.append(out)

    df_1d = pd.concat(rows_1d, ignore_index=True) if rows_1d else pd.DataFrame()
    df_1d_summary = build_1d_feature_summary(df_1d) if not df_1d.empty else pd.DataFrame()

    # -----------------------------
    # 2D sweep
    # -----------------------------
    pairs = list(combinations(FEATURE_LIST, 2))
    rows_2d: List[pd.DataFrame] = []

    for i, (fx, fy) in enumerate(pairs, start=1):
        print(f"[2D {i}/{len(pairs)}] {fx}  x  {fy}")
        out = summarize_2d(df, fx, fy)
        if not out.empty:
            rows_2d.append(out)

    df_2d = pd.concat(rows_2d, ignore_index=True) if rows_2d else pd.DataFrame()
    df_2d_summary = build_2d_pair_summary(df_2d) if not df_2d.empty else pd.DataFrame()

    # -----------------------------
    # Save
    # -----------------------------
    if not df_1d.empty:
        df_1d.to_csv(OUT_1D_CSV, index=False)
    if not df_2d.empty:
        df_2d.to_csv(OUT_2D_CSV, index=False)
    if not df_1d_summary.empty:
        df_1d_summary.to_csv(OUT_1D_SUMMARY_CSV, index=False)
    if not df_2d_summary.empty:
        df_2d_summary.to_csv(OUT_2D_SUMMARY_CSV, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_1D_CSV}")
    print(f"Wrote: {OUT_2D_CSV}")
    print(f"Wrote: {OUT_1D_SUMMARY_CSV}")
    print(f"Wrote: {OUT_2D_SUMMARY_CSV}")

    if not df_1d_summary.empty:
        print("\nTop 15 1D features by hit_rate_range:")
        print(df_1d_summary.head(15).to_string(index=False))

    if not df_2d_summary.empty:
        print("\nTop 15 2D pairs by hit_rate_range:")
        print(df_2d_summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
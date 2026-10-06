# walkforward_combo_validator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\walkforward_combo_validator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS_CSV = OUTPUT_DIR / "walkforward_combo_results.csv"
OUT_SUMMARY_CSV = OUTPUT_DIR / "walkforward_combo_summary.csv"


# --------------------------------------------------
# research subset
# --------------------------------------------------
TARGET_COL = "mr_pts_t600"
THRESHOLD = 1.0

BAND_FILTER = [1]
SIDE_FILTER = "upper"


# --------------------------------------------------
# chronological multi-fold settings
# folds are rolling contiguous blocks:
# [train | val | test]
# --------------------------------------------------
N_FOLDS = 5

TRAIN_FRAC_WITHIN_FOLD = 0.60
VAL_FRAC_WITHIN_FOLD = 0.20
TEST_FRAC_WITHIN_FOLD = 0.20


# --------------------------------------------------
# combo candidates
# upper +1.0 finalists from overlap / union analysis
# --------------------------------------------------
COMBOS: List[Dict[str, Any]] = [
    {
        "combo_name": "VOTE_A2_C2_D1",
        "logic": "AT_LEAST_2",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
            "D1_depth_mid_ask__queue_imbalance_std_5s",
        ],
    },
    {
        "combo_name": "VOTE_A2_B2_C2",
        "logic": "AT_LEAST_2",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "B2_realized_vol_1m__depth_mid_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
        ],
    },
    {
        "combo_name": "VOTE_A2_C2_D2",
        "logic": "AT_LEAST_2",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
            "D2_depth_near_ask__queue_imbalance_std_5s",
        ],
    },
    {
        "combo_name": "ALL2_A2_C2",
        "logic": "ALL",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
        ],
    },
    {
        "combo_name": "ALL2_C1_C2",
        "logic": "ALL",
        "rules": [
            "C1_cancel_rate_ask__cancel_ratio",
            "C2_cancel_rate_ask__bid_depth_curvature",
        ],
    },
    {
        "combo_name": "ALL2_B2_C2",
        "logic": "ALL",
        "rules": [
            "B2_realized_vol_1m__depth_mid_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
        ],
    },
    {
        "combo_name": "ALL3_A2_C2_D1",
        "logic": "ALL",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
            "D1_depth_mid_ask__queue_imbalance_std_5s",
        ],
    },
    {
        "combo_name": "SINGLE_C2",
        "logic": "SINGLE",
        "rules": [
            "C2_cancel_rate_ask__bid_depth_curvature",
        ],
    },
    {
        "combo_name": "SINGLE_A2",
        "logic": "SINGLE",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
        ],
    },
    {
        "combo_name": "SINGLE_C1",
        "logic": "SINGLE",
        "rules": [
            "C1_cancel_rate_ask__cancel_ratio",
        ],
    },
    {
        "combo_name": "SINGLE_B2",
        "logic": "SINGLE",
        "rules": [
            "B2_realized_vol_1m__depth_mid_ask",
        ],
    },
    {
        "combo_name": "UNION_A2_C2",
        "logic": "ANY",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "C2_cancel_rate_ask__bid_depth_curvature",
        ],
    },
]


# --------------------------------------------------
# atomic rules
# --------------------------------------------------
RULE_DEFS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "A2_depth_near_ask__cancel_rate_ask": {
        "depth_near_ask": (4.0, 128.0),
        "cancel_rate_ask": (6.6, 427.0),
    },
    "B2_realized_vol_1m__depth_mid_ask": {
        "realized_vol_1m": (0.000682729, 0.00677785),
        "depth_mid_ask": (7.0, 150.0),
    },
    "C1_cancel_rate_ask__cancel_ratio": {
        "cancel_rate_ask": (6.6, 427.0),
        "cancel_ratio": (0.0809026, 0.495652),
    },
    "C2_cancel_rate_ask__bid_depth_curvature": {
        "cancel_rate_ask": (6.6, 427.0),
        "bid_depth_curvature": (-6.625, 11.5),
    },
    "D1_depth_mid_ask__queue_imbalance_std_5s": {
        "depth_mid_ask": (7.0, 151.0),
        "queue_imbalance_std_5s": (0.195659, 0.645851),
    },
    "D2_depth_near_ask__queue_imbalance_std_5s": {
        "depth_near_ask": (4.0, 128.0),
        "queue_imbalance_std_5s": (0.195659, 0.645851),
    },
}


def ensure_columns(df: pl.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


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


def build_rule_mask(pdf: pd.DataFrame, conds: Dict[str, Tuple[float, float]]) -> pd.Series:
    mask = pd.Series(True, index=pdf.index)
    for col, (lo, hi) in conds.items():
        mask = mask & pdf[col].notna() & (pdf[col] >= lo) & (pdf[col] <= hi)
    return mask


def combine_masks(mask_dict: Dict[str, pd.Series], combo_logic: str, combo_rules: List[str]) -> pd.Series:
    if combo_logic == "SINGLE":
        return mask_dict[combo_rules[0]].copy()

    mat = pd.concat([mask_dict[r].astype(int) for r in combo_rules], axis=1)
    vote_count = mat.sum(axis=1)

    if combo_logic == "ANY":
        return vote_count >= 1
    if combo_logic == "ALL":
        return vote_count == len(combo_rules)
    if combo_logic == "AT_LEAST_2":
        return vote_count >= 2

    raise ValueError(f"Unknown combo logic: {combo_logic}")


def eval_subset(df_part: pd.DataFrame, mask: pd.Series) -> Dict[str, Any]:
    sub = df_part.loc[mask]
    n = len(sub)
    if n == 0:
        return {
            "n_events": 0,
            "hit_rate": np.nan,
            "avg_mr_pts": np.nan,
        }
    return {
        "n_events": int(n),
        "hit_rate": float(sub["target"].mean()),
        "avg_mr_pts": float(sub[TARGET_COL].mean()),
    }


def build_fold_ranges(n: int, n_folds: int) -> List[Tuple[int, int]]:
    # split full sample into equal contiguous outer blocks
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_folds)]


def split_inner_block(start: int, end: int) -> Dict[str, slice]:
    m = end - start
    train_end = start + int(m * TRAIN_FRAC_WITHIN_FOLD)
    val_end = start + int(m * (TRAIN_FRAC_WITHIN_FOLD + VAL_FRAC_WITHIN_FOLD))
    return {
        "train": slice(start, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, end),
    }


def safe_mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if pd.notna(x)]
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    needed_rule_cols = sorted({c for conds in RULE_DEFS.values() for c in conds.keys()})
    needed_cols = ["event_id", "date_ymd", "ts", "band_k", "side", TARGET_COL] + needed_rule_cols

    df = pl.read_parquet(FEATURES_PATH)
    ensure_columns(df, needed_cols)

    df = apply_subset(df)
    df = add_target(df)
    df = df.sort("ts")

    if df.height < 500:
        raise ValueError(f"Too few rows after filtering: {df.height}")

    pdf = df.to_pandas().reset_index(drop=True)

    print("\n--------------------------------------")
    print("Multi-Fold Walk-Forward Combo Validator")
    print("--------------------------------------")
    print(f"Rows analyzed: {len(pdf):,}")
    print(f"TARGET_COL   = {TARGET_COL}")
    print(f"THRESHOLD    = {THRESHOLD}")
    print(f"BAND_FILTER  = {BAND_FILTER}")
    print(f"SIDE_FILTER  = {SIDE_FILTER}")
    print(f"N_FOLDS      = {N_FOLDS}")
    print(f"N_COMBOS     = {len(COMBOS)}")

    # atomic rule masks over full dataset
    atomic_masks: Dict[str, pd.Series] = {
        rule_name: build_rule_mask(pdf, conds).astype(bool)
        for rule_name, conds in RULE_DEFS.items()
    }

    # combo masks over full dataset
    combo_masks: Dict[str, pd.Series] = {}
    for combo in COMBOS:
        combo_masks[combo["combo_name"]] = combine_masks(
            atomic_masks,
            combo["logic"],
            combo["rules"],
        ).astype(bool)

    fold_ranges = build_fold_ranges(len(pdf), N_FOLDS)

    fold_rows = []
    for fold_id, (outer_start, outer_end) in enumerate(fold_ranges, start=1):
        inner = split_inner_block(outer_start, outer_end)

        train_df = pdf.iloc[inner["train"]].copy()
        val_df = pdf.iloc[inner["val"]].copy()
        test_df = pdf.iloc[inner["test"]].copy()

        print(f"\nFold {fold_id}/{N_FOLDS} | rows {outer_start}:{outer_end}")

        for combo in COMBOS:
            combo_name = combo["combo_name"]
            mask_full = combo_masks[combo_name]

            train_mask = mask_full.iloc[inner["train"]]
            val_mask = mask_full.iloc[inner["val"]]
            test_mask = mask_full.iloc[inner["test"]]

            train_stats = eval_subset(train_df, train_mask)
            val_stats = eval_subset(val_df, val_mask)
            test_stats = eval_subset(test_df, test_mask)

            fold_rows.append({
                "fold_id": fold_id,
                "combo_name": combo_name,
                "logic": combo["logic"],
                "rule_names": " | ".join(combo["rules"]),

                "train_n_events": train_stats["n_events"],
                "train_hit_rate": train_stats["hit_rate"],
                "train_avg_mr_pts": train_stats["avg_mr_pts"],

                "val_n_events": val_stats["n_events"],
                "val_hit_rate": val_stats["hit_rate"],
                "val_avg_mr_pts": val_stats["avg_mr_pts"],

                "test_n_events": test_stats["n_events"],
                "test_hit_rate": test_stats["hit_rate"],
                "test_avg_mr_pts": test_stats["avg_mr_pts"],

                "hit_rate_degradation_test_minus_train":
                    (test_stats["hit_rate"] - train_stats["hit_rate"])
                    if pd.notna(test_stats["hit_rate"]) and pd.notna(train_stats["hit_rate"])
                    else np.nan,

                "avg_mr_pts_degradation_test_minus_train":
                    (test_stats["avg_mr_pts"] - train_stats["avg_mr_pts"])
                    if pd.notna(test_stats["avg_mr_pts"]) and pd.notna(train_stats["avg_mr_pts"])
                    else np.nan,
            })

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(OUT_RESULTS_CSV, index=False)

    summary_rows = []
    for combo_name, g in df_folds.groupby("combo_name", sort=False):
        g = g.sort_values("fold_id")

        test_hit_mean = safe_mean(g["test_hit_rate"].tolist())
        test_hit_std = safe_mean([np.std(g["test_hit_rate"].dropna(), ddof=0)]) if g["test_hit_rate"].notna().any() else np.nan
        test_mr_mean = safe_mean(g["test_avg_mr_pts"].tolist())
        test_mr_std = safe_mean([np.std(g["test_avg_mr_pts"].dropna(), ddof=0)]) if g["test_avg_mr_pts"].notna().any() else np.nan

        train_hit_mean = safe_mean(g["train_hit_rate"].tolist())
        val_hit_mean = safe_mean(g["val_hit_rate"].tolist())

        mean_test_events = safe_mean(g["test_n_events"].tolist())
        min_test_events = int(g["test_n_events"].min())
        max_test_events = int(g["test_n_events"].max())

        hit_deg_mean = safe_mean(g["hit_rate_degradation_test_minus_train"].tolist())
        mr_deg_mean = safe_mean(g["avg_mr_pts_degradation_test_minus_train"].tolist())

        positive_test_folds = int((g["test_n_events"] > 0).sum())
        strong_test_folds = int((g["test_hit_rate"] >= 0.90).sum())

        robust_score = (
            (0.40 * (0.0 if pd.isna(test_hit_mean) else test_hit_mean)) +
            (0.20 * (0.0 if pd.isna(test_mr_mean) else min(test_mr_mean, 15.0) / 15.0)) +
            (0.15 * (0.0 if pd.isna(val_hit_mean) else val_hit_mean)) +
            (0.15 * min(mean_test_events, 100.0) / 100.0) +
            (0.10 * (1.0 - min(abs(hit_deg_mean), 1.0) if pd.notna(hit_deg_mean) else 0.0))
        )

        stability_score = (
            (0.50 * (0.0 if pd.isna(test_hit_mean) else test_hit_mean)) +
            (0.20 * (1.0 - min(test_hit_std, 0.25) / 0.25 if pd.notna(test_hit_std) else 0.0)) +
            (0.15 * strong_test_folds / N_FOLDS) +
            (0.15 * positive_test_folds / N_FOLDS)
        )

        summary_rows.append({
            "combo_name": combo_name,
            "logic": g["logic"].iloc[0],
            "rule_names": g["rule_names"].iloc[0],

            "n_folds": int(len(g)),
            "positive_test_folds": positive_test_folds,
            "strong_test_folds_hit_ge_0_90": strong_test_folds,

            "mean_train_hit_rate": train_hit_mean,
            "mean_val_hit_rate": val_hit_mean,
            "mean_test_hit_rate": test_hit_mean,
            "std_test_hit_rate": test_hit_std,

            "mean_test_avg_mr_pts": test_mr_mean,
            "std_test_avg_mr_pts": test_mr_std,

            "mean_test_events": mean_test_events,
            "min_test_events": min_test_events,
            "max_test_events": max_test_events,

            "mean_hit_rate_degradation_test_minus_train": hit_deg_mean,
            "mean_avg_mr_pts_degradation_test_minus_train": mr_deg_mean,

            "robust_score": robust_score,
            "stability_score": stability_score,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["robust_score", "stability_score", "mean_test_hit_rate", "mean_test_avg_mr_pts"],
        ascending=[False, False, False, False],
        kind="stable",
    )

    df_summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_RESULTS_CSV}")
    print(f"Wrote: {OUT_SUMMARY_CSV}")

    show_cols = [
        "combo_name",
        "logic",
        "positive_test_folds",
        "strong_test_folds_hit_ge_0_90",
        "mean_test_events",
        "mean_test_hit_rate",
        "std_test_hit_rate",
        "mean_test_avg_mr_pts",
        "std_test_avg_mr_pts",
        "mean_hit_rate_degradation_test_minus_train",
        "robust_score",
        "stability_score",
    ]

    print("\nTop combo summaries:")
    print(df_summary[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
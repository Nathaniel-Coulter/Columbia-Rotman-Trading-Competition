# walkforward_rule_validator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\walkforward_rule_validator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS_CSV = OUTPUT_DIR / "walkforward_rule_results.csv"


# --------------------------------------------------
# Fixed subset settings
# --------------------------------------------------
TARGET_COL = "mr_pts_t600"
THRESHOLD = 1.0

BAND_FILTER = [1]
SIDE_FILTER = "upper"

TRAIN_FRAC = 0.65
VAL_FRAC = 0.20
TEST_FRAC = 0.15


# --------------------------------------------------
# Candidate rules to validate
# --------------------------------------------------
RULES: List[Dict[str, Any]] = [
    {
        "rule_name": "A1_depth_mid_ask__cancel_rate_ask",
        "conds": {
            "depth_mid_ask": (7.0, 151.0),
            "cancel_rate_ask": (6.6, 427.0),
        },
    },
    {
        "rule_name": "A2_depth_near_ask__cancel_rate_ask",
        "conds": {
            "depth_near_ask": (4.0, 128.0),
            "cancel_rate_ask": (6.6, 427.0),
        },
    },
    {
        "rule_name": "B1_realized_volatility_30s__depth_mid_ask",
        "conds": {
            "realized_volatility_30s": (0.000514191, 0.00594037),
            "depth_mid_ask": (7.0, 150.0),
        },
    },
    {
        "rule_name": "B2_realized_vol_1m__depth_mid_ask",
        "conds": {
            "realized_vol_1m": (0.000682729, 0.00677785),
            "depth_mid_ask": (7.0, 150.0),
        },
    },
    {
        "rule_name": "C1_cancel_rate_ask__cancel_ratio",
        "conds": {
            "cancel_rate_ask": (6.6, 427.0),
            "cancel_ratio": (0.0809026, 0.495652),
        },
    },
    {
        "rule_name": "C2_cancel_rate_ask__bid_depth_curvature",
        "conds": {
            "cancel_rate_ask": (6.6, 427.0),
            "bid_depth_curvature": (-6.625, 11.5),
        },
    },
    {
        "rule_name": "D1_depth_mid_ask__queue_imbalance_std_5s",
        "conds": {
            "depth_mid_ask": (7.0, 151.0),
            "queue_imbalance_std_5s": (0.195659, 0.645851),
        },
    },
    {
        "rule_name": "D2_depth_near_ask__queue_imbalance_std_5s",
        "conds": {
            "depth_near_ask": (4.0, 128.0),
            "queue_imbalance_std_5s": (0.195659, 0.645851),
        },
    },
]


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


def time_split_indices(n: int):
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


def evaluate_split(pdf: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    n_events = len(pdf)
    if n_events == 0:
        return {
            "split": split_name,
            "n_events": 0,
            "hit_rate": None,
            "avg_mr_pts": None,
        }

    return {
        "split": split_name,
        "n_events": int(n_events),
        "hit_rate": float(pdf["target"].mean()),
        "avg_mr_pts": float(pdf[TARGET_COL].mean()),
    }


def build_rule_mask(pdf: pd.DataFrame, conds: Dict[str, tuple[float, float]]) -> pd.Series:
    mask = pd.Series(True, index=pdf.index)
    for col, (lo, hi) in conds.items():
        mask = mask & pdf[col].notna() & (pdf[col] >= lo) & (pdf[col] <= hi)
    return mask


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    needed_cols = ["ts", "band_k", "side", TARGET_COL] + sorted(
        {col for rule in RULES for col in rule["conds"].keys()}
    )

    df = pl.read_parquet(FEATURES_PATH)
    ensure_columns(df, needed_cols)

    df = apply_subset(df)
    df = add_target(df)
    df = df.sort("ts")

    if df.height == 0:
        raise ValueError("No rows left after filtering.")

    pdf = df.to_pandas()
    split_idx = time_split_indices(len(pdf))

    results: List[Dict[str, Any]] = []

    print("\n--------------------------------------")
    print("Walk-forward Rule Validator")
    print("--------------------------------------")
    print(f"Rows analyzed: {len(pdf):,}")
    print(f"TARGET_COL   = {TARGET_COL}")
    print(f"THRESHOLD    = {THRESHOLD}")
    print(f"BAND_FILTER  = {BAND_FILTER}")
    print(f"SIDE_FILTER  = {SIDE_FILTER}")
    print(f"N_RULES      = {len(RULES)}")

    for i, rule in enumerate(RULES, start=1):
        rule_name = rule["rule_name"]
        conds = rule["conds"]

        print(f"[{i}/{len(RULES)}] {rule_name}")

        mask = build_rule_mask(pdf, conds)
        pdf_rule = pdf.loc[mask].copy()

        train_df = pdf_rule.iloc[split_idx["train"]] if len(pdf_rule) > 0 else pdf_rule
        val_df = pdf_rule.iloc[split_idx["val"]] if len(pdf_rule) > 0 else pdf_rule
        test_df = pdf_rule.iloc[split_idx["test"]] if len(pdf_rule) > 0 else pdf_rule

        # IMPORTANT:
        # We want time splits based on original sequence, not re-splitting filtered rows.
        # So instead do mask by original split.
        train_df = pdf.iloc[split_idx["train"]].loc[mask.iloc[split_idx["train"]]]
        val_df = pdf.iloc[split_idx["val"]].loc[mask.iloc[split_idx["val"]]]
        test_df = pdf.iloc[split_idx["test"]].loc[mask.iloc[split_idx["test"]]]

        train_stats = evaluate_split(train_df, "train")
        val_stats = evaluate_split(val_df, "val")
        test_stats = evaluate_split(test_df, "test")

        train_hit = train_stats["hit_rate"]
        test_hit = test_stats["hit_rate"]
        train_mr = train_stats["avg_mr_pts"]
        test_mr = test_stats["avg_mr_pts"]

        hit_rate_degradation = None
        avg_mr_pts_degradation = None

        if train_hit is not None and test_hit is not None:
            hit_rate_degradation = test_hit - train_hit

        if train_mr is not None and test_mr is not None:
            avg_mr_pts_degradation = test_mr - train_mr

        row = {
            "rule_name": rule_name,
            "rule_text": " AND ".join(
                f"{col} in [{lo}, {hi}]"
                for col, (lo, hi) in conds.items()
            ),

            "train_n_events": train_stats["n_events"],
            "train_hit_rate": train_stats["hit_rate"],
            "train_avg_mr_pts": train_stats["avg_mr_pts"],

            "val_n_events": val_stats["n_events"],
            "val_hit_rate": val_stats["hit_rate"],
            "val_avg_mr_pts": val_stats["avg_mr_pts"],

            "test_n_events": test_stats["n_events"],
            "test_hit_rate": test_stats["hit_rate"],
            "test_avg_mr_pts": test_stats["avg_mr_pts"],

            "hit_rate_degradation_test_minus_train": hit_rate_degradation,
            "avg_mr_pts_degradation_test_minus_train": avg_mr_pts_degradation,
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    # Simple ranking
    results_df["robust_score"] = (
        results_df["test_hit_rate"].fillna(0.0) * 0.45
        + (results_df["test_avg_mr_pts"].fillna(0.0).clip(upper=15.0) / 15.0) * 0.25
        + (results_df["test_n_events"].fillna(0).clip(upper=400) / 400.0) * 0.20
        + (1.0 - results_df["hit_rate_degradation_test_minus_train"].fillna(-1.0).abs().clip(upper=1.0)) * 0.10
    )

    results_df = results_df.sort_values(
        ["robust_score", "test_hit_rate", "test_avg_mr_pts", "test_n_events"],
        ascending=[False, False, False, False],
        kind="stable",
    )

    results_df.to_csv(OUT_RESULTS_CSV, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_RESULTS_CSV}")

    show_cols = [
        "rule_name",
        "train_n_events", "train_hit_rate", "train_avg_mr_pts",
        "val_n_events", "val_hit_rate", "val_avg_mr_pts",
        "test_n_events", "test_hit_rate", "test_avg_mr_pts",
        "hit_rate_degradation_test_minus_train",
        "avg_mr_pts_degradation_test_minus_train",
        "robust_score",
    ]
    print("\nTop walk-forward rules:")
    print(results_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
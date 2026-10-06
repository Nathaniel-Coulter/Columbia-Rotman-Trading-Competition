# rule_union_simulator.py
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\rule_union_simulator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS_CSV = OUTPUT_DIR / "rule_union_results.csv"


# --------------------------------------------------
# Same subset
# --------------------------------------------------
TARGET_COL = "mr_pts_t600"
THRESHOLD = 1.0

BAND_FILTER = [1]
SIDE_FILTER = "upper"

TRAIN_FRAC = 0.65
VAL_FRAC = 0.20
TEST_FRAC = 0.15


# --------------------------------------------------
# Candidate rules
# I removed the most obvious near-duplicates so the basket
# is diversified across rule families.
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


def time_split_indices(n: int) -> Dict[str, slice]:
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


def build_rule_mask(pdf: pd.DataFrame, conds: Dict[str, Tuple[float, float]]) -> pd.Series:
    mask = pd.Series(True, index=pdf.index)
    for col, (lo, hi) in conds.items():
        mask = mask & pdf[col].notna() & (pdf[col] >= lo) & (pdf[col] <= hi)
    return mask


def eval_mask(df_part: pd.DataFrame, mask: pd.Series) -> Dict[str, Any]:
    sub = df_part.loc[mask]
    n = len(sub)
    if n == 0:
        return {
            "n_events": 0,
            "hit_rate": None,
            "avg_mr_pts": None,
        }
    return {
        "n_events": int(n),
        "hit_rate": float(sub["target"].mean()),
        "avg_mr_pts": float(sub[TARGET_COL].mean()),
    }


def join_rule_names(rule_names: List[str]) -> str:
    return " | ".join(rule_names)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    needed_cols = ["event_id", "date_ymd", "ts", "band_k", "side", TARGET_COL] + sorted(
        {col for rule in RULES for col in rule["conds"].keys()}
    )

    df = pl.read_parquet(FEATURES_PATH)
    ensure_columns(df, needed_cols)

    df = apply_subset(df)
    df = add_target(df)
    df = df.sort("ts")

    if df.height == 0:
        raise ValueError("No rows left after filtering.")

    pdf = df.to_pandas().reset_index(drop=True)
    split_idx = time_split_indices(len(pdf))

    train_df = pdf.iloc[split_idx["train"]].copy()
    val_df = pdf.iloc[split_idx["val"]].copy()
    test_df = pdf.iloc[split_idx["test"]].copy()

    print("\n--------------------------------------")
    print("Rule Union Simulator")
    print("--------------------------------------")
    print(f"Rows analyzed: {len(pdf):,}")
    print(f"TARGET_COL   = {TARGET_COL}")
    print(f"THRESHOLD    = {THRESHOLD}")
    print(f"BAND_FILTER  = {BAND_FILTER}")
    print(f"SIDE_FILTER  = {SIDE_FILTER}")
    print(f"N_RULES      = {len(RULES)}")

    # per-rule event flags
    rule_flags = {}
    for rule in RULES:
        rule_flags[rule["rule_name"]] = build_rule_mask(pdf, rule["conds"]).astype(bool)

    # split flags
    train_flags = {k: v.iloc[split_idx["train"]] for k, v in rule_flags.items()}
    val_flags = {k: v.iloc[split_idx["val"]] for k, v in rule_flags.items()}
    test_flags = {k: v.iloc[split_idx["test"]] for k, v in rule_flags.items()}

    results = []

    rule_names = [r["rule_name"] for r in RULES]

    # --------------------------------------------------
    # 1) Single rules
    # --------------------------------------------------
    for rn in rule_names:
        train_stats = eval_mask(train_df, train_flags[rn])
        val_stats = eval_mask(val_df, val_flags[rn])
        test_stats = eval_mask(test_df, test_flags[rn])

        results.append({
            "combo_type": "single",
            "combo_size": 1,
            "combo_name": rn,
            "logic": "single",
            "rule_names": rn,

            "train_n_events": train_stats["n_events"],
            "train_hit_rate": train_stats["hit_rate"],
            "train_avg_mr_pts": train_stats["avg_mr_pts"],

            "val_n_events": val_stats["n_events"],
            "val_hit_rate": val_stats["hit_rate"],
            "val_avg_mr_pts": val_stats["avg_mr_pts"],

            "test_n_events": test_stats["n_events"],
            "test_hit_rate": test_stats["hit_rate"],
            "test_avg_mr_pts": test_stats["avg_mr_pts"],
        })

    # --------------------------------------------------
    # 2) ANY of 2, 3, 4 rules
    # 3) ALL of 2, 3 rules
    # 4) AT LEAST 2 of 3 / 4 rules
    # --------------------------------------------------
    for k in [2, 3, 4]:
        for combo in combinations(rule_names, k):
            combo = list(combo)

            # build count of triggered rules
            train_sum = sum(train_flags[r].astype(int) for r in combo)
            val_sum = sum(val_flags[r].astype(int) for r in combo)
            test_sum = sum(test_flags[r].astype(int) for r in combo)

            # ANY
            train_any = train_sum >= 1
            val_any = val_sum >= 1
            test_any = test_sum >= 1

            train_stats = eval_mask(train_df, train_any)
            val_stats = eval_mask(val_df, val_any)
            test_stats = eval_mask(test_df, test_any)

            results.append({
                "combo_type": "union_any",
                "combo_size": k,
                "combo_name": join_rule_names(combo),
                "logic": f"ANY_{k}",
                "rule_names": join_rule_names(combo),

                "train_n_events": train_stats["n_events"],
                "train_hit_rate": train_stats["hit_rate"],
                "train_avg_mr_pts": train_stats["avg_mr_pts"],

                "val_n_events": val_stats["n_events"],
                "val_hit_rate": val_stats["hit_rate"],
                "val_avg_mr_pts": val_stats["avg_mr_pts"],

                "test_n_events": test_stats["n_events"],
                "test_hit_rate": test_stats["hit_rate"],
                "test_avg_mr_pts": test_stats["avg_mr_pts"],
            })

            # ALL
            if k <= 3:
                train_all = train_sum == k
                val_all = val_sum == k
                test_all = test_sum == k

                train_stats = eval_mask(train_df, train_all)
                val_stats = eval_mask(val_df, val_all)
                test_stats = eval_mask(test_df, test_all)

                results.append({
                    "combo_type": "intersection_all",
                    "combo_size": k,
                    "combo_name": join_rule_names(combo),
                    "logic": f"ALL_{k}",
                    "rule_names": join_rule_names(combo),

                    "train_n_events": train_stats["n_events"],
                    "train_hit_rate": train_stats["hit_rate"],
                    "train_avg_mr_pts": train_stats["avg_mr_pts"],

                    "val_n_events": val_stats["n_events"],
                    "val_hit_rate": val_stats["hit_rate"],
                    "val_avg_mr_pts": val_stats["avg_mr_pts"],

                    "test_n_events": test_stats["n_events"],
                    "test_hit_rate": test_stats["hit_rate"],
                    "test_avg_mr_pts": test_stats["avg_mr_pts"],
                })

            # AT LEAST 2
            if k >= 3:
                train_ge2 = train_sum >= 2
                val_ge2 = val_sum >= 2
                test_ge2 = test_sum >= 2

                train_stats = eval_mask(train_df, train_ge2)
                val_stats = eval_mask(val_df, val_ge2)
                test_stats = eval_mask(test_df, test_ge2)

                results.append({
                    "combo_type": "vote_ge_2",
                    "combo_size": k,
                    "combo_name": join_rule_names(combo),
                    "logic": f"AT_LEAST_2_OF_{k}",
                    "rule_names": join_rule_names(combo),

                    "train_n_events": train_stats["n_events"],
                    "train_hit_rate": train_stats["hit_rate"],
                    "train_avg_mr_pts": train_stats["avg_mr_pts"],

                    "val_n_events": val_stats["n_events"],
                    "val_hit_rate": val_stats["hit_rate"],
                    "val_avg_mr_pts": val_stats["avg_mr_pts"],

                    "test_n_events": test_stats["n_events"],
                    "test_hit_rate": test_stats["hit_rate"],
                    "test_avg_mr_pts": test_stats["avg_mr_pts"],
                })

    out = pd.DataFrame(results)

    # degradation
    out["hit_rate_degradation_test_minus_train"] = out["test_hit_rate"] - out["train_hit_rate"]
    out["avg_mr_pts_degradation_test_minus_train"] = out["test_avg_mr_pts"] - out["train_avg_mr_pts"]

    # simple filters for practical use
    out["min_split_events"] = out[["train_n_events", "val_n_events", "test_n_events"]].min(axis=1)
    out["mean_split_events"] = out[["train_n_events", "val_n_events", "test_n_events"]].mean(axis=1)

    # ranking scores
    out["score_balanced"] = (
        out["test_hit_rate"].fillna(0.0) * 0.45
        + (out["test_avg_mr_pts"].fillna(0.0).clip(upper=15.0) / 15.0) * 0.25
        + (out["test_n_events"].fillna(0).clip(upper=400) / 400.0) * 0.20
        + (1.0 - out["hit_rate_degradation_test_minus_train"].fillna(-1.0).abs().clip(upper=1.0)) * 0.10
    )

    out["score_conservative"] = (
        out["test_hit_rate"].fillna(0.0) * 0.40
        + (out["val_hit_rate"].fillna(0.0)) * 0.20
        + (out["test_avg_mr_pts"].fillna(0.0).clip(upper=15.0) / 15.0) * 0.15
        + (out["min_split_events"].fillna(0).clip(upper=100) / 100.0) * 0.15
        + (1.0 - out["hit_rate_degradation_test_minus_train"].fillna(-1.0).abs().clip(upper=1.0)) * 0.10
    )

    out["score_trade_focus"] = (
        out["test_hit_rate"].fillna(0.0) * 0.35
        + (out["test_avg_mr_pts"].fillna(0.0).clip(upper=15.0) / 15.0) * 0.30
        + (out["test_n_events"].fillna(0).clip(upper=300) / 300.0) * 0.15
        + (out["val_hit_rate"].fillna(0.0)) * 0.10
        + (1.0 - out["avg_mr_pts_degradation_test_minus_train"].fillna(-15.0).abs().clip(upper=15.0) / 15.0) * 0.10
    )

    out = out.sort_values(
        ["score_balanced", "test_hit_rate", "test_avg_mr_pts", "test_n_events"],
        ascending=[False, False, False, False],
        kind="stable",
    )

    out.to_csv(OUT_RESULTS_CSV, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_RESULTS_CSV}")

    cols = [
        "combo_type",
        "combo_size",
        "logic",
        "rule_names",
        "train_n_events",
        "train_hit_rate",
        "val_n_events",
        "val_hit_rate",
        "test_n_events",
        "test_hit_rate",
        "test_avg_mr_pts",
        "hit_rate_degradation_test_minus_train",
        "score_balanced",
        "score_conservative",
        "score_trade_focus",
    ]

    print("\nTop 25 by score_balanced:")
    print(out[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
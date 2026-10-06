# rule_overlap_analyzer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\rule_overlap_analyzer"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENT_FLAGS_CSV = OUTPUT_DIR / "rule_event_flags.csv"
OUT_PAIRWISE_OVERLAP_CSV = OUTPUT_DIR / "rule_pairwise_overlap.csv"
OUT_RULE_SUMMARY_CSV = OUTPUT_DIR / "rule_summary.csv"


# --------------------------------------------------
# Same subset as walk-forward run
# --------------------------------------------------
TARGET_COL = "mr_pts_t600"
THRESHOLD = 1.0

BAND_FILTER = [1]
SIDE_FILTER = "upper"


# --------------------------------------------------
# Same candidate rules
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


def build_rule_mask(pdf: pd.DataFrame, conds: Dict[str, tuple[float, float]]) -> pd.Series:
    mask = pd.Series(True, index=pdf.index)
    for col, (lo, hi) in conds.items():
        mask = mask & pdf[col].notna() & (pdf[col] >= lo) & (pdf[col] <= hi)
    return mask


def jaccard(a: int, b: int, inter: int) -> float | None:
    union = a + b - inter
    if union <= 0:
        return None
    return inter / union


def overlap_ratio(inter: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return inter / denom


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

    print("\n--------------------------------------")
    print("Rule Overlap Analyzer")
    print("--------------------------------------")
    print(f"Rows analyzed: {len(pdf):,}")
    print(f"TARGET_COL   = {TARGET_COL}")
    print(f"THRESHOLD    = {THRESHOLD}")
    print(f"BAND_FILTER  = {BAND_FILTER}")
    print(f"SIDE_FILTER  = {SIDE_FILTER}")
    print(f"N_RULES      = {len(RULES)}")

    # Per-event flags
    flags_df = pdf[["event_id", "date_ymd", "ts", "target", TARGET_COL]].copy()

    for rule in RULES:
        rule_name = rule["rule_name"]
        flags_df[rule_name] = build_rule_mask(pdf, rule["conds"]).astype(int)

    flags_df.to_csv(OUT_EVENT_FLAGS_CSV, index=False)

    # Rule summary
    rule_summary_rows = []
    for rule in RULES:
        r = rule["rule_name"]
        mask = flags_df[r] == 1
        n_events = int(mask.sum())
        hit_rate = float(flags_df.loc[mask, "target"].mean()) if n_events > 0 else None
        avg_mr_pts = float(flags_df.loc[mask, TARGET_COL].mean()) if n_events > 0 else None
        rule_summary_rows.append({
            "rule_name": r,
            "n_events": n_events,
            "hit_rate": hit_rate,
            "avg_mr_pts": avg_mr_pts,
        })

    rule_summary_df = pd.DataFrame(rule_summary_rows).sort_values(
        ["n_events", "hit_rate"], ascending=[False, False], kind="stable"
    )
    rule_summary_df.to_csv(OUT_RULE_SUMMARY_CSV, index=False)

    # Pairwise overlap
    pair_rows = []
    rule_names = [r["rule_name"] for r in RULES]

    for i, r1 in enumerate(rule_names):
        m1 = flags_df[r1].to_numpy(dtype=int)
        n1 = int(m1.sum())

        for j, r2 in enumerate(rule_names):
            if j <= i:
                continue

            m2 = flags_df[r2].to_numpy(dtype=int)
            n2 = int(m2.sum())

            inter = int(((m1 == 1) & (m2 == 1)).sum())
            only_r1 = int(((m1 == 1) & (m2 == 0)).sum())
            only_r2 = int(((m1 == 0) & (m2 == 1)).sum())
            union = n1 + n2 - inter

            pair_rows.append({
                "rule_1": r1,
                "rule_2": r2,
                "n_rule_1": n1,
                "n_rule_2": n2,
                "intersection_n": inter,
                "union_n": union,
                "only_rule_1_n": only_r1,
                "only_rule_2_n": only_r2,
                "jaccard_overlap": jaccard(n1, n2, inter),
                "intersection_over_rule_1": overlap_ratio(inter, n1),
                "intersection_over_rule_2": overlap_ratio(inter, n2),
                "min_rule_coverage_overlap": overlap_ratio(inter, min(n1, n2)),
                "max_rule_coverage_overlap": overlap_ratio(inter, max(n1, n2)),
            })

    pair_df = pd.DataFrame(pair_rows).sort_values(
        ["jaccard_overlap", "intersection_n"],
        ascending=[False, False],
        kind="stable",
    )
    pair_df.to_csv(OUT_PAIRWISE_OVERLAP_CSV, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_EVENT_FLAGS_CSV}")
    print(f"Wrote: {OUT_PAIRWISE_OVERLAP_CSV}")
    print(f"Wrote: {OUT_RULE_SUMMARY_CSV}")

    print("\nRule summary:")
    print(rule_summary_df.to_string(index=False))

    print("\nTop pairwise overlaps by Jaccard:")
    show_cols = [
        "rule_1",
        "rule_2",
        "n_rule_1",
        "n_rule_2",
        "intersection_n",
        "union_n",
        "jaccard_overlap",
        "intersection_over_rule_1",
        "intersection_over_rule_2",
        "min_rule_coverage_overlap",
    ]
    print(pair_df[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
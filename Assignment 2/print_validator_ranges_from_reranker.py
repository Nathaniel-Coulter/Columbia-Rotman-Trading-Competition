# print_validator_ranges_from_reranker.py
from __future__ import annotations

from pathlib import Path
import polars as pl

CSV_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\entry_quality_reranker\entry_quality_reranker_summary.csv"
)

# optional filters
RULE_NAME_FILTER = None         # e.g. "SINGLE_C1" or None for all
ONLY_POSITIVE_SCORE = False     # True = only rows with strict_score > 0
SORT_BY = ["rule_name", "secondary_feature", "strict_score"]


def make_label(feature_bin: int, n_bins_for_feature: int) -> str:
    if n_bins_for_feature <= 1:
        return "only_bin"
    if feature_bin == 0:
        return "low_bin"
    if feature_bin == n_bins_for_feature - 1:
        return "high_bin"
    return f"mid_bin_{feature_bin}"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pl.read_csv(CSV_PATH)

    required = {
        "rule_name",
        "secondary_feature",
        "feature_bin",
        "feature_min",
        "feature_max",
        "feature_median",
        "strict_score",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if RULE_NAME_FILTER is not None:
        df = df.filter(pl.col("rule_name") == RULE_NAME_FILTER)

    if ONLY_POSITIVE_SCORE:
        df = df.filter(pl.col("strict_score") > 0)

    if df.height == 0:
        print("No rows matched filters.")
        return

    # number of bins observed per (rule_name, secondary_feature)
    bin_counts = (
        df.group_by(["rule_name", "secondary_feature"])
        .agg(pl.col("feature_bin").n_unique().alias("n_bins_for_feature"))
    )

    df = df.join(bin_counts, on=["rule_name", "secondary_feature"], how="left")

    # label based on feature_bin position within that feature
    rows = []
    for r in df.sort(SORT_BY).to_dicts():
        feature_bin = int(r["feature_bin"])
        n_bins_for_feature = int(r["n_bins_for_feature"])
        label = make_label(feature_bin, n_bins_for_feature)

        rows.append({
            "rule_name": r["rule_name"],
            "secondary_feature": r["secondary_feature"],
            "feature_bin": feature_bin,
            "label": label,
            "lo": float(r["feature_min"]),
            "hi": float(r["feature_max"]),
            "median": float(r["feature_median"]) if r["feature_median"] is not None else None,
            "strict_score": float(r["strict_score"]),
        })

    out = pl.from_dicts(rows).sort(
        ["rule_name", "secondary_feature", "feature_bin"]
    )

    print("\n=== Range table for validator ===")
    print(
        out.select([
            "rule_name",
            "secondary_feature",
            "feature_bin",
            "label",
            "lo",
            "hi",
            "median",
            "strict_score",
        ])
    )

    print("\n=== Paste-ready validator blocks ===")
    current_rule = None
    for r in out.to_dicts():
        rule_name = r["rule_name"]
        if rule_name != current_rule:
            current_rule = rule_name
            print(f"\n# {rule_name}")

        print(
            f'"{r["secondary_feature"]}": [("{r["label"]}", {r["lo"]}, {r["hi"]})],'
        )


if __name__ == "__main__":
    main()
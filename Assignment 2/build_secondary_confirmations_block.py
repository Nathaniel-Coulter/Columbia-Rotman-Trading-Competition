# build_secondary_confirmations_from_reranker.py
from __future__ import annotations

from pathlib import Path
import polars as pl

CSV_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\entry_quality_reranker\entry_quality_reranker_summary.csv"
)

# ------------------------------------------------------------
# Filters / settings
# ------------------------------------------------------------
RULE_NAME_FILTER = "UNION_A2_C2"      # set to None to include all rule_names
MIN_STRICT_SCORE = None             # e.g. 0.0, or None for no filter
MIN_N_ENTERED = 25                   # e.g. 25 if you want a floor
TOP_K_PER_RULE = None               # e.g. 10, or None for all best-per-feature rows

# Optional:
# If True, only keep bins with positive strict score
ONLY_POSITIVE_STRICT = False

# If True, require feature_min / feature_max to be non-null
REQUIRE_BOUNDS = True


def make_label(feature_bin: int, max_bin_for_feature: int) -> str:
    if feature_bin == 0:
        return "low_bin"
    if feature_bin == max_bin_for_feature:
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
        "n_entered",
        "feature_min",
        "feature_max",
        "strict_score",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if RULE_NAME_FILTER is not None:
        df = df.filter(pl.col("rule_name") == RULE_NAME_FILTER)

    if MIN_STRICT_SCORE is not None:
        df = df.filter(pl.col("strict_score") >= MIN_STRICT_SCORE)

    if ONLY_POSITIVE_STRICT:
        df = df.filter(pl.col("strict_score") > 0)

    if MIN_N_ENTERED is not None:
        df = df.filter(pl.col("n_entered") >= MIN_N_ENTERED)

    if REQUIRE_BOUNDS:
        df = df.filter(
            pl.col("feature_min").is_not_null() &
            pl.col("feature_max").is_not_null()
        )

    if df.height == 0:
        print("No rows left after filters.")
        return

    # max observed bin index per (rule_name, secondary_feature)
    max_bin_df = (
        df.group_by(["rule_name", "secondary_feature"])
        .agg(pl.col("feature_bin").max().alias("max_bin_for_feature"))
    )
    df = df.join(max_bin_df, on=["rule_name", "secondary_feature"], how="left")

    # best row per (rule_name, secondary_feature)
    # tie-breakers:
    #   1) strict_score desc
    #   2) n_entered desc
    #   3) lower feature_bin first
    best_df = (
        df.sort(
            ["rule_name", "secondary_feature", "strict_score", "n_entered", "feature_bin"],
            descending=[False, False, True, True, False],
        )
        .group_by(["rule_name", "secondary_feature"], maintain_order=True)
        .first()
        .sort(["rule_name", "strict_score", "n_entered"], descending=[False, True, True])
    )

    if TOP_K_PER_RULE is not None:
        best_df = (
            best_df
            .group_by("rule_name", maintain_order=True)
            .head(TOP_K_PER_RULE)
        )

    # nice table
    display_df = best_df.select([
        "rule_name",
        "secondary_feature",
        "feature_bin",
        "max_bin_for_feature",
        "n_entered",
        "strict_score",
        "feature_min",
        "feature_max",
    ])

    print("\n=== Best bin per feature ===")
    print(display_df)

    print("\n=== Paste-ready SECONDARY_CONFIRMATIONS block ===")
    current_rule = None
    for r in best_df.to_dicts():
        rule_name = r["rule_name"]
        feature = r["secondary_feature"]
        feature_bin = int(r["feature_bin"])
        max_bin_for_feature = int(r["max_bin_for_feature"])
        lo = float(r["feature_min"])
        hi = float(r["feature_max"])
        label = make_label(feature_bin, max_bin_for_feature)

        if rule_name != current_rule:
            current_rule = rule_name
            print(f"\n# {rule_name}")

        print(f'"{feature}": [')
        print(f'    ("{label}", {repr(lo)}, {repr(hi)}),')
        print("],")


if __name__ == "__main__":
    main()
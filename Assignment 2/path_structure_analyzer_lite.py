# path_structure_analyzer_lite.py
import polars as pl
from pathlib import Path

FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\path_structure_analyzer_lite"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "path_structure_lite_summary.csv"

# --------------------------------------------------
# core study settings
# --------------------------------------------------
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "lower"

# symmetric first-passage levels to test
LEVELS_PTS = [2.0, 4.0, 6.0, 8.0]

# --------------------------------------------------
# candidate rules
# same family you validated already
# --------------------------------------------------
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
# helper
# --------------------------------------------------
def choose_time_col(level_pts: float) -> str | None:
    """
    If your parquet contains time_to_vwap_s only, we can only report that.
    If later you add richer first-passage timing columns, wire them here.
    """
    # currently your parquet appears to only have time_to_vwap_s
    # so we keep a placeholder interface
    return "time_to_vwap_s"


# --------------------------------------------------
# load and subset
# --------------------------------------------------
df = pl.read_parquet(FEATURES_PATH)

mfe_col = f"mr_pts_t{HORIZON_S}"
mae_col = f"mae_pts_t{HORIZON_S}"

required = ["band_k", "side", mfe_col, mae_col]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.filter(
    pl.col("band_k").is_in(BAND_FILTER) &
    (pl.col("side") == SIDE_FILTER) &
    pl.col(mfe_col).is_not_null() &
    pl.col(mae_col).is_not_null()
).sort("ts")

if df.height == 0:
    raise ValueError("No rows left after base filtering.")

print("\n--------------------------------------")
print("Path Structure Analyzer")
print("--------------------------------------")
print(f"Rows analyzed: {df.height:,}")
print(f"HORIZON_S     = {HORIZON_S}")
print(f"BAND_FILTER   = {BAND_FILTER}")
print(f"SIDE_FILTER   = {SIDE_FILTER}")
print(f"LEVELS_PTS    = {LEVELS_PTS}")
print(f"N_RULES       = {len(RULES)}")

rows_out = []

for rule_name, rule_expr in RULES.items():
    dfr = df.filter(rule_expr)

    if dfr.height == 0:
        continue

    base_row = {
        "rule_name": rule_name,
        "n_events": dfr.height,
        "mean_mr_pts": dfr.select(pl.col(mfe_col).mean()).item(),
        "mean_mae_pts": dfr.select(pl.col(mae_col).mean()).item(),
    }

    # optional rough VWAP-touch timing if available
    if "time_to_vwap_s" in dfr.columns:
        base_row["mean_time_to_vwap_s"] = dfr.select(
            pl.col("time_to_vwap_s").drop_nulls().mean()
        ).item()
        base_row["median_time_to_vwap_s"] = dfr.select(
            pl.col("time_to_vwap_s").drop_nulls().median()
        ).item()
    else:
        base_row["mean_time_to_vwap_s"] = None
        base_row["median_time_to_vwap_s"] = None

    for lvl in LEVELS_PTS:
        # with only horizon-level MFE/MAE available, this is a structural dominance test:
        # did favorable excursion reach +lvl?
        # did adverse excursion reach -lvl?
        #
        # This does NOT yet know which happened first intrapath.
        # It is still useful for viability screening.
        #
        # Later, if you add first-passage path columns, we can replace these with true order stats.
        dfr_lvl = dfr.with_columns([
            (pl.col(mfe_col) >= lvl).cast(pl.Int8).alias("mr_hit"),
            (pl.col(mae_col) >= lvl).cast(pl.Int8).alias("mae_hit"),
        ])

        n = dfr_lvl.height
        mr_hit_rate = dfr_lvl.select(pl.col("mr_hit").mean()).item()
        mae_hit_rate = dfr_lvl.select(pl.col("mae_hit").mean()).item()

        both_hit_rate = dfr_lvl.select(
            ((pl.col("mr_hit") == 1) & (pl.col("mae_hit") == 1)).cast(pl.Int8).mean()
        ).item()

        mr_only_rate = dfr_lvl.select(
            ((pl.col("mr_hit") == 1) & (pl.col("mae_hit") == 0)).cast(pl.Int8).mean()
        ).item()

        mae_only_rate = dfr_lvl.select(
            ((pl.col("mr_hit") == 0) & (pl.col("mae_hit") == 1)).cast(pl.Int8).mean()
        ).item()

        neither_rate = dfr_lvl.select(
            ((pl.col("mr_hit") == 0) & (pl.col("mae_hit") == 0)).cast(pl.Int8).mean()
        ).item()

        dominance = None
        if mr_hit_rate is not None and mae_hit_rate is not None:
            dominance = mr_hit_rate - mae_hit_rate

        row = dict(base_row)
        row.update({
            "level_pts": lvl,
            "n_level_events": n,
            "mr_hit_rate": mr_hit_rate,
            "mae_hit_rate": mae_hit_rate,
            "mr_minus_mae_hit_rate": dominance,
            "mr_only_rate": mr_only_rate,
            "mae_only_rate": mae_only_rate,
            "both_hit_rate": both_hit_rate,
            "neither_hit_rate": neither_rate,
        })
        rows_out.append(row)

out = pl.DataFrame(rows_out).sort(
    ["level_pts", "mr_minus_mae_hit_rate", "mr_hit_rate"],
    descending=[False, True, True]
)

out.write_csv(OUT_CSV)

print("\nDone.")
print(f"Wrote: {OUT_CSV}")

print("\nTop path-structure rows:")
print(out.head(20))
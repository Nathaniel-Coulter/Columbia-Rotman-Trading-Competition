# secondary_confirmation_entry_sweep.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\secondary_confirmation_entry_sweep"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "secondary_confirmation_event_level.csv"
SUMMARY_OUT = OUTPUT_DIR / "secondary_confirmation_summary.csv"


# ============================================================
# Research settings
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "lower"

# parent rules from your earlier work
COMBO_RULES = [
    "SINGLE_A2",
    "SINGLE_C1",
    "ALL2_A2_C1",
    "VOTE_B1_C1_D1",
    "VOTE4_A2_B1_C1_E1",
]

# post-trigger adverse move before "refined" entry
ENTRY_ADVERSE_PTS = 0

# post-entry evaluation levels
FAVORABLE_TARGETS_PTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ADVERSE_LEVELS_PTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# feature binning
N_BINS = 5
MIN_BIN_EVENTS = 25

# choose features here freely
SECONDARY_FEATURES: List[str] = [
    "OBI_l1",
    "OBI_l3",
    "OBI_l5",
    "OBI_l10",
    "weighted_OBI_l3",
    "weighted_OBI_l5",
    "weighted_OBI_l10",
    "depth_near_bid",
    "depth_near_ask",
    "depth_mid_bid",
    "depth_mid_ask",
    "depth_far_bid",
    "depth_far_ask",
    "liquidity_slope_bid",
    "liquidity_slope_ask",
    "bid_convexity",
    "ask_convexity",
    "convexity_ratio_bid",
    "convexity_ratio_ask",
    "convexity_imbalance",
    "liquidity_compression",
    "bid_depth_curvature",
    "ask_depth_curvature",
    "depth_curvature_pressure",
    "bid_curvature_l5",
    "ask_curvature_l5",
    "curvature_pressure_l5",
    "wall_strength_opposite_3",
    "wall_strength_opposite_5",
    "wall_strength_opposite_3_2s_avg",
    "wall_strength_opposite_3_5s_avg",
    "wall_strength_opposite_5_2s_avg",
    "wall_strength_opposite_5_5s_avg",
    "wall_persistence_2s",
    "microprice_deviation",
    "microprice_pressure",
    "spread_change_rate",
    "spread_volatility",
    "spread_expansion_ratio",
    "cancel_rate_bid",
    "cancel_rate_ask",
    "cancel_ratio",
    "add_rate_bid",
    "add_rate_ask",
    "remove_rate_near_bid",
    "remove_rate_near_ask",
    "update_rate_bid",
    "update_rate_ask",
    "bid_liquidity_shock",
    "ask_liquidity_shock",
    "net_liquidity_shock",
    "replenishment_rate_near_5s",
    "best_bid_queue_norm",
    "best_ask_queue_norm",
    "queue_length_total_norm",
    "bid_queue_change_rate",
    "ask_queue_change_rate",
    "bid_queue_depletion_rate",
    "ask_queue_depletion_rate",
    "queue_imbalance",
    "microprice",
    "microprice_minus_mid",
    "spread",
    "spread_mid_ratio",
    "best_bid_queue_ratio",
    "best_ask_queue_ratio",
    "queue_length_ratio",
    "trades_per_second",
    "volume_per_second",
    "avg_trade_size",
    "trade_sign_autocorr_10s",
    "trade_sign_autocorr_30s",
    "trade_sign_autocorr_60s",
    "aggressive_buy_volume",
    "aggressive_sell_volume",
    "aggressive_volume_ratio",
    "OFI",
    "OFI_5s",
    "OFI_30s",
    "OFI_60s",
    "CVD",
    "cvd_slope",
    "cvd_acceleration",
    "flow_burst_ratio",
    "volume_burst_ratio",
    "OFI_burst_ratio",
    "signed_volume",
    "trade_imbalance",
    "trade_imbalance_5s",
    "trade_imbalance_30s",
    "trade_imbalance_60s",
    "queue_imbalance_mean_5s",
    "queue_imbalance_std_5s",
    "queue_imbalance_slope_5s",
    "queue_imbalance_positive_ratio",
    "impact_efficiency_3s",
    "impact_efficiency_10s",
    "impact_efficiency_30s",
    "impact_decay_3s",
    "impact_decay_5s",
    "buy_impact_efficiency_10s",
    "sell_impact_efficiency_10s",
    "impact_asymmetry_10s",
    "absorption_ratio",
    "flow_impact",
    "flow_toxicity",
    "flow_toxicity_10s",
    "flow_toxicity_30s",
    "toxicity_ratio",
]

# ============================================================
# Rule definitions
# These should match your earlier selected rules as closely as possible
# ============================================================
def rule_single_a2(r: dict) -> bool:
    return (
        r.get("depth_near_ask") is not None
        and r.get("cancel_rate_ask") is not None
        and float(r["depth_near_ask"]) >= 3.0
        and float(r["depth_near_ask"]) <= 127.0
        and float(r["cancel_rate_ask"]) >= 6.8
        and float(r["cancel_rate_ask"]) <= 222.2
    )


def rule_single_c1(r: dict) -> bool:
    return (
        r.get("realized_vol_1m") is not None
        and r.get("depth_mid_ask") is not None
        and float(r["realized_vol_1m"]) >= 0.000694397
        and float(r["realized_vol_1m"]) <= 0.00493388
        and float(r["depth_mid_ask"]) >= 5.0
        and float(r["depth_mid_ask"]) <= 149.0
    )


def rule_b1(r: dict) -> bool:
    return (
        r.get("depth_mid_ask") is not None
        and r.get("queue_imbalance_std_5s") is not None
        and float(r["depth_mid_ask"]) >= 5.0
        and float(r["depth_mid_ask"]) <= 150.0
        and float(r["queue_imbalance_std_5s"]) >= 0.194964
        and float(r["queue_imbalance_std_5s"]) <= 0.528415
    )


def rule_d1(r: dict) -> bool:
    return (
        r.get("cancel_rate_ask") is not None
        and r.get("bid_depth_curvature") is not None
        and float(r["cancel_rate_ask"]) >= 6.8
        and float(r["cancel_rate_ask"]) <= 222.2
        and float(r["bid_depth_curvature"]) >= -6.875
        and float(r["bid_depth_curvature"]) <= 25.375
    )


def rule_e1(r: dict) -> bool:
    return (
        r.get("cancel_rate_ask") is not None
        and r.get("cancel_ratio") is not None
        and float(r["cancel_rate_ask"]) >= 6.8
        and float(r["cancel_rate_ask"]) <= 222.2
        and float(r["cancel_ratio"]) >= 0.0805134
        and float(r["cancel_ratio"]) <= 0.319363
    )


def combo_rule_hit(rule_name: str, r: dict) -> bool:
    a2 = rule_single_a2(r)
    c1 = rule_single_c1(r)
    b1 = rule_b1(r)
    d1 = rule_d1(r)
    e1 = rule_e1(r)

    if rule_name == "SINGLE_A2":
        return a2
    if rule_name == "SINGLE_C1":
        return c1
    if rule_name == "ALL2_A2_C1":
        return a2 and c1
    if rule_name == "VOTE_B1_C1_D1":
        return sum([b1, c1, d1]) >= 2
    if rule_name == "VOTE4_A2_B1_C1_E1":
        return sum([a2, b1, c1, e1]) >= 2

    raise ValueError(f"Unknown rule name: {rule_name}")


# ============================================================
# Helpers
# ============================================================
def iter_day_parquets(root: Path):
    for ydir in sorted(root.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            for ddir in sorted(mdir.glob("day=*")):
                p = ddir / "part-000.parquet"
                if not p.exists():
                    continue
                year = int(ydir.name.split("=")[1])
                month = int(mdir.name.split("=")[1])
                day = int(ddir.name.split("=")[1])
                yield year, month, day, p


def build_day_to_path(root: Path) -> Dict[str, Path]:
    out = {}
    for y, m, d, p in iter_day_parquets(root):
        out[f"{y:04d}-{m:02d}-{d:02d}"] = p
    return out


def load_trade_day(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)

    required = {"ts", "type", "price"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required trade columns: {missing}")

    return (
        df.filter(
            (pl.col("type") == 2) &
            pl.col("price").is_not_null()
        )
        .select(["ts", "price"])
        .sort("ts")
    )


def compute_path_stats_from_entry(
    entry_px: float,
    future_prices: List[float],
) -> Dict[str, float]:
    if not future_prices:
        return {
            "post_entry_mfe_pts": 0.0,
            "post_entry_mae_pts": 0.0,
            "post_entry_end_ret_pts": 0.0,
        }

    rets = [float(px) - entry_px for px in future_prices]
    return {
        "post_entry_mfe_pts": max(rets),
        "post_entry_mae_pts": max([-x for x in rets] + [0.0]),
        "post_entry_end_ret_pts": rets[-1],
    }


def first_time_reach_adverse(
    entry_px0: float,
    future_ts: List,
    future_prices: List[float],
    adverse_pts: float,
) -> Optional[Tuple]:
    for ts, px in zip(future_ts, future_prices):
        if (entry_px0 - float(px)) >= adverse_pts:
            return ts, float(px)
    return None


def add_hit_columns(row: dict, mfe: float, mae: float) -> None:
    for x in FAVORABLE_TARGETS_PTS:
        row[f"hit_fav_{x:.1f}pt"] = int(mfe >= x)
    for x in ADVERSE_LEVELS_PTS:
        row[f"hit_adv_{x:.1f}pt"] = int(mae >= x)


def safe_mean(xs: List[float]) -> Optional[float]:
    return None if not xs else sum(xs) / len(xs)


def assign_feature_bins(df: pl.DataFrame, col: str, n_bins: int) -> pl.DataFrame:
    n = df.height
    return (
        df.with_columns(
            pl.col(col).rank("ordinal").alias("_rank_tmp")
        )
        .with_columns(
            (
                ((pl.col("_rank_tmp") - 1) / n) * n_bins
            )
            .floor()
            .clip(0, n_bins - 1)
            .cast(pl.Int64)
            .alias("_feature_bin")
        )
        .drop("_rank_tmp")
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features parquet: {FEATURES_PATH}")
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(f"Missing full tick root: {FULL_TICK_ROOT}")

    print("\n--------------------------------------")
    print("Secondary Confirmation Entry Sweep")
    print("--------------------------------------")
    print(f"HORIZON_S                = {HORIZON_S}")
    print(f"BAND_FILTER              = {BAND_FILTER}")
    print(f"SIDE_FILTER              = {SIDE_FILTER}")
    print(f"ENTRY_ADVERSE_PTS        = {ENTRY_ADVERSE_PTS}")
    print(f"N_BINS                   = {N_BINS}")
    print(f"MIN_BIN_EVENTS           = {MIN_BIN_EVENTS}")
    print(f"N_PARENT_RULES           = {len(COMBO_RULES)}")
    print(f"N_SECONDARY_FEATURES     = {len(SECONDARY_FEATURES)}")

    df = pl.read_parquet(FEATURES_PATH).sort("ts")

    df = df.filter(
        pl.col("band_k").is_in(BAND_FILTER) &
        (pl.col("side") == SIDE_FILTER)
    )

    if df.height == 0:
        raise ValueError("No rows left after band/side filtering.")

    print(f"Rows analyzed: {df.height:,}")

    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    event_rows = []
    all_days = df["date_ymd"].unique().sort().to_list()

    for i_day, day in enumerate(all_days, start=1):
        df_day = df.filter(pl.col("date_ymd") == day).sort("ts")
        full_path = day_to_path.get(day)

        if full_path is None or not full_path.exists():
            continue

        tr = load_trade_day(full_path)
        if tr.height == 0:
            continue

        tr_ts = tr["ts"].to_list()
        tr_px = [float(x) for x in tr["price"].to_list()]

        events = df_day.to_dicts()

        j = 0
        n_tr = len(tr_ts)

        for ev in events:
            ev_ts = ev["ts"]
            while j + 1 < n_tr and tr_ts[j + 1] <= ev_ts:
                j += 1

            if j >= n_tr or tr_ts[j] > ev_ts:
                continue

            entry_px0 = tr_px[j]

            # base rule trigger
            hits = {rule_name: combo_rule_hit(rule_name, ev) for rule_name in COMBO_RULES}
            if not any(hits.values()):
                continue

            horizon_end_us = ev_ts.timestamp() + HORIZON_S
            fut_ts = []
            fut_px = []

            k = j
            while k < n_tr and tr_ts[k].timestamp() <= horizon_end_us:
                fut_ts.append(tr_ts[k])
                fut_px.append(tr_px[k])
                k += 1

            if len(fut_px) < 2:
                continue

            adverse_entry = first_time_reach_adverse(
                entry_px0=entry_px0,
                future_ts=fut_ts,
                future_prices=fut_px,
                adverse_pts=ENTRY_ADVERSE_PTS,
            )

            if adverse_entry is None:
                continue

            entry_ts, refined_entry_px = adverse_entry

            # future path after refined entry
            post_ts = []
            post_px = []
            for ts2, px2 in zip(fut_ts, fut_px):
                if ts2 >= entry_ts:
                    post_ts.append(ts2)
                    post_px.append(px2)

            stats = compute_path_stats_from_entry(
                entry_px=refined_entry_px,
                future_prices=post_px,
            )

            base_row = {
                "event_id": ev["event_id"],
                "date_ymd": ev["date_ymd"],
                "trigger_ts": ev_ts,
                "refined_entry_ts": entry_ts,
                "entry_delay_s": float((entry_ts - ev_ts).total_seconds()),
                "rule_hit_any": 1,
                "entry_price0": entry_px0,
                "refined_entry_px": refined_entry_px,
                "entry_discount_pts": entry_px0 - refined_entry_px,
                **stats,
            }
            add_hit_columns(base_row, stats["post_entry_mfe_pts"], stats["post_entry_mae_pts"])

            for rule_name, rule_hit in hits.items():
                if not rule_hit:
                    continue

                row = dict(base_row)
                row["rule_name"] = rule_name

                for feat in SECONDARY_FEATURES:
                    row[feat] = ev.get(feat)

                event_rows.append(row)

        if i_day % 25 == 0:
            print(f"...processed {i_day} days")

    if not event_rows:
        raise ValueError("No event rows generated.")

    event_df = pl.from_dicts(event_rows)

    # --------------------------------------------------------
    # Per-feature bin sweep inside each parent rule
    # --------------------------------------------------------
    summary_rows = []

    for rule_name in COMBO_RULES:
        df_rule = event_df.filter(pl.col("rule_name") == rule_name)

        for feat in SECONDARY_FEATURES:
            if feat not in df_rule.columns:
                continue

            df_feat = df_rule.filter(pl.col(feat).is_not_null())
            if df_feat.height < (N_BINS * MIN_BIN_EVENTS):
                continue

            df_binned = assign_feature_bins(df_feat, feat, N_BINS)

            grouped = (
                df_binned.group_by("_feature_bin")
                .agg([
                    pl.len().alias("n_entered"),
                    pl.col("entry_delay_s").mean().alias("mean_entry_delay_s"),
                    pl.col("post_entry_mfe_pts").mean().alias("mean_post_entry_mfe_pts"),
                    pl.col("post_entry_mae_pts").mean().alias("mean_post_entry_mae_pts"),
                    pl.col("post_entry_end_ret_pts").mean().alias("mean_post_entry_end_ret_pts"),
                    pl.col(feat).min().alias("feature_min"),
                    pl.col(feat).max().alias("feature_max"),
                    pl.col(feat).median().alias("feature_median"),
                    *[
                        pl.col(f"hit_fav_{x:.1f}pt").mean().alias(f"hit_fav_{x:.1f}pt")
                        for x in FAVORABLE_TARGETS_PTS
                    ],
                    *[
                        pl.col(f"hit_adv_{x:.1f}pt").mean().alias(f"hit_adv_{x:.1f}pt")
                        for x in ADVERSE_LEVELS_PTS
                    ],
                ])
                .sort("_feature_bin")
            )

            for r in grouped.to_dicts():
                n_entered = int(r["n_entered"])
                if n_entered < MIN_BIN_EVENTS:
                    continue

                hit_fav_4 = float(r.get("hit_fav_4.0pt", 0.0))
                hit_fav_6 = float(r.get("hit_fav_6.0pt", 0.0))
                hit_adv_4 = float(r.get("hit_adv_4.0pt", 0.0))
                hit_adv_6 = float(r.get("hit_adv_6.0pt", 0.0))
                mean_end = float(r["mean_post_entry_end_ret_pts"])
                mean_mfe = float(r["mean_post_entry_mfe_pts"])
                mean_mae = float(r["mean_post_entry_mae_pts"])

                refinement_score = (
                    0.30 * hit_fav_4
                    + 0.25 * hit_fav_6
                    - 0.20 * hit_adv_4
                    - 0.15 * hit_adv_6
                    + 0.05 * mean_end
                    + 0.03 * mean_mfe
                    - 0.02 * mean_mae
                )

                summary_rows.append({
                    "rule_name": rule_name,
                    "secondary_feature": feat,
                    "feature_bin": r["_feature_bin"],
                    "n_entered": n_entered,
                    "mean_entry_delay_s": r["mean_entry_delay_s"],
                    "mean_post_entry_mfe_pts": mean_mfe,
                    "mean_post_entry_mae_pts": mean_mae,
                    "mean_post_entry_end_ret_pts": mean_end,
                    "feature_min": r["feature_min"],
                    "feature_max": r["feature_max"],
                    "feature_median": r["feature_median"],
                    **{
                        f"hit_fav_{x:.1f}pt": r.get(f"hit_fav_{x:.1f}pt")
                        for x in FAVORABLE_TARGETS_PTS
                    },
                    **{
                        f"hit_adv_{x:.1f}pt": r.get(f"hit_adv_{x:.1f}pt")
                        for x in ADVERSE_LEVELS_PTS
                    },
                    "refinement_score": refinement_score,
                })

    if not summary_rows:
        raise ValueError("No summary rows generated. Try fewer features or lower MIN_BIN_EVENTS.")

    summary_df = (
        pl.from_dicts(summary_rows)
        .sort(["refinement_score", "n_entered"], descending=[True, True])
    )

    event_df.write_csv(EVENT_LEVEL_OUT)
    summary_df.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop secondary confirmation bins:")
    print(summary_df.head(30))


if __name__ == "__main__":
    main()
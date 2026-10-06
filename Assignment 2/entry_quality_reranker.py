# entry_quality_reranker.py
from __future__ import annotations

from pathlib import Path
from bisect import bisect_right
from typing import Dict, List, Tuple, Any

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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\entry_quality_reranker"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "entry_quality_reranker_event_level.csv"
SUMMARY_OUT = OUTPUT_DIR / "entry_quality_reranker_summary.csv"


# ============================================================
# Core configuration
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "upper"

# Parent rule we are refining
PARENT_RULE_NAME = "UNION_A2_C2"

# We are evaluating better entry timing after adverse excursion
ENTRY_ADVERSE_PTS = 1.0

# candidate binning setup
N_BINS = 2
MIN_BIN_EVENTS = 5

# score weights / thresholds
FAVORABLE_TARGET_PTS = 4.0
ADVERSE_PENALTY_LEVEL_PTS = 6.0

# this is where you can easily add / remove features
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

# if you want, add more here later:
# SECONDARY_FEATURES += [
#     "best_bid_queue_norm",
#     "bid_queue_depletion_rate",
# ]


# ============================================================
# Parent rule definitions
# Same families we already validated
# ============================================================
RULES: Dict[str, List[Tuple[str, str, float, float]]] = {
    "A2": [
        ("depth_near_ask", "between", 4.0, 128.0),
        ("cancel_rate_ask", "between", 6.6, 427.0),
    ],
    "B2": [
        ("realized_vol_1m", "between", 0.000682729, 0.00677785),
        ("depth_mid_ask", "between", 7.0, 150.0),
    ],
    "C1": [
        ("cancel_rate_ask", "between", 6.6, 427.0),
        ("cancel_ratio", "between", 0.0809026, 0.495652),
    ],
    "C2": [
        ("cancel_rate_ask", "between", 6.6, 427.0),
        ("bid_depth_curvature", "between", -6.625, 11.5),
    ],
    "D1": [
        ("depth_mid_ask", "between", 7.0, 151.0),
        ("queue_imbalance_std_5s", "between", 0.195659, 0.645851),
    ],
    "D2": [
        ("depth_near_ask", "between", 4.0, 128.0),
        ("queue_imbalance_std_5s", "between", 0.195659, 0.645851),
    ],
}


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
        raise ValueError(f"Missing required trade cols: {missing}")

    return (
        df.filter(
            (pl.col("type") == 2)
            & pl.col("price").is_not_null()
        )
        .select(["ts", "price"])
        .sort("ts")
    )


def passes_condition(v: Any, op: str, lo: float, hi: float | None = None) -> bool:
    if v is None:
        return False
    try:
        x = float(v)
    except Exception:
        return False

    if op == "between":
        assert hi is not None
        return lo <= x <= hi
    raise ValueError(f"Unsupported op: {op}")


def event_passes_rule(row: Dict[str, Any], rule_conds: List[Tuple[str, str, float, float]]) -> bool:
    for feat, op, lo, hi in rule_conds:
        if not passes_condition(row.get(feat), op, lo, hi):
            return False
    return True


def parent_rule_pass(row: Dict[str, Any]) -> bool:
    if PARENT_RULE_NAME == "SINGLE_A2":
        return event_passes_rule(row, RULES["A2"])
    elif PARENT_RULE_NAME == "SINGLE_B2":
        return event_passes_rule(row, RULES["B2"])
    elif PARENT_RULE_NAME == "SINGLE_C1":
        return event_passes_rule(row, RULES["C1"])
    elif PARENT_RULE_NAME == "SINGLE_C2":
        return event_passes_rule(row, RULES["C2"])
    elif PARENT_RULE_NAME == "ALL2_B2_C2":
        return event_passes_rule(row, RULES["B2"]) and event_passes_rule(row, RULES["C2"])
    elif PARENT_RULE_NAME == "ALL2_A2_C2":
        return event_passes_rule(row, RULES["A2"]) and event_passes_rule(row, RULES["C2"])
    elif PARENT_RULE_NAME == "ALL2_C1_C2":
        return event_passes_rule(row, RULES["C1"]) and event_passes_rule(row, RULES["C2"])
    elif PARENT_RULE_NAME == "UNION_A2_C2":
        return event_passes_rule(row, RULES["A2"]) or event_passes_rule(row, RULES["C2"])
    else:
        raise ValueError(f"Unsupported PARENT_RULE_NAME: {PARENT_RULE_NAME}")

def compute_bin_edges(values: List[float], n_bins: int) -> List[float]:
    xs = sorted(values)
    if len(xs) == 0:
        return []
    edges = []
    for k in range(n_bins + 1):
        idx = int(round(k * (len(xs) - 1) / n_bins))
        edges.append(xs[idx])
    return edges


def assign_bin(x: float, edges: List[float]) -> int:
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i < n_bins - 1:
            if lo <= x < hi:
                return i
        else:
            if lo <= x <= hi:
                return i
    return n_bins - 1


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(FULL_TICK_ROOT)

    df = pl.read_parquet(FEATURES_PATH).sort(["date_ymd", "ts"])

    df = df.filter(
        pl.col("band_k").is_in(BAND_FILTER)
        & (pl.col("side") == SIDE_FILTER)
    )

    rows = df.to_dicts()

    # parent-rule subset
    parent_rows = [r for r in rows if parent_rule_pass(r)]
    if not parent_rows:
        raise ValueError("No rows passed parent rule.")

    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    print("\n--------------------------------------")
    print("Entry Quality Reranker")
    print("--------------------------------------")
    print(f"HORIZON_S                 = {HORIZON_S}")
    print(f"BAND_FILTER               = {BAND_FILTER}")
    print(f"SIDE_FILTER               = {SIDE_FILTER}")
    print(f"PARENT_RULE_NAME          = {PARENT_RULE_NAME}")
    print(f"ENTRY_ADVERSE_PTS         = {ENTRY_ADVERSE_PTS}")
    print(f"N_BINS                    = {N_BINS}")
    print(f"MIN_BIN_EVENTS            = {MIN_BIN_EVENTS}")
    print(f"N_SECONDARY_FEATURES      = {len(SECONDARY_FEATURES)}")
    print(f"Rows analyzed             = {len(rows):,}")
    print(f"Parent rule rows          = {len(parent_rows):,}")

    # --------------------------------------------------------
    # First, compute entry-refined event outcomes from raw path
    # --------------------------------------------------------
    event_level_records: List[Dict[str, Any]] = []

    parent_rows_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for r in parent_rows:
        parent_rows_by_day.setdefault(r["date_ymd"], []).append(r)

    processed_days = 0
    for day_ymd, evs in parent_rows_by_day.items():
        full_path = day_to_path.get(day_ymd)
        if full_path is None or not full_path.exists():
            continue

        tr = load_trade_day(full_path)
        if tr.height == 0:
            continue

        ts_list = tr["ts"].to_list()
        px_list = [float(x) for x in tr["price"].to_list()]

        for r in evs:
            entry_price = r.get("entry_price")
            event_ts = r.get("ts")
            side = r.get("side")

            if entry_price is None or event_ts is None or side is None:
                continue

            entry_price = float(entry_price)
            direction = 1.0 if side == "lower" else -1.0

            j0 = bisect_right(ts_list, event_ts) - 1
            if j0 < 0:
                continue

            # find post-trigger delayed entry after adverse move
            entry_idx = None
            entry_ts = None
            entry_px = None

            horizon_end_idx = j0
            for j in range(j0, len(ts_list)):
                dt_s = (ts_list[j] - event_ts).total_seconds()
                if dt_s > HORIZON_S:
                    break
                horizon_end_idx = j

                px = px_list[j]
                signed_move_from_event = (px - entry_price) * direction
                adverse_move = -signed_move_from_event

                if adverse_move >= ENTRY_ADVERSE_PTS:
                    entry_idx = j
                    entry_ts = ts_list[j]
                    entry_px = px
                    break

            if entry_idx is None:
                continue

            future_px = px_list[entry_idx:horizon_end_idx + 1]
            if not future_px:
                continue

            signed_post_moves = [(p - entry_px) * direction for p in future_px]
            post_mfe = max(signed_post_moves)
            post_mae = max(0.0, -min(signed_post_moves))
            end_ret = signed_post_moves[-1]
            delay_s = (entry_ts - event_ts).total_seconds()

            rec = {
                "event_id": r["event_id"],
                "date_ymd": r["date_ymd"],
                "ts": r["ts"],
                "entry_price_trigger": entry_price,
                "entry_price_refined": entry_px,
                "entry_delay_s": delay_s,
                "post_entry_mfe_pts": post_mfe,
                "post_entry_mae_pts": post_mae,
                "post_entry_end_ret_pts": end_ret,
            }

            for feat in SECONDARY_FEATURES:
                rec[feat] = r.get(feat)

            event_level_records.append(rec)

        processed_days += 1
        if processed_days % 25 == 0:
            print(f"...processed {processed_days} days")

    if not event_level_records:
        raise ValueError("No event-level refined entries were produced.")

    event_df = pl.from_dicts(event_level_records)
    event_df.write_csv(EVENT_LEVEL_OUT)
    print(f"Refined-entry event rows = {event_df.height:,}")

    # --------------------------------------------------------
    # Now rerank candidate features by strict entry quality
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for feat in SECONDARY_FEATURES:
        feat_df = event_df.filter(pl.col(feat).is_not_null())
        if feat_df.height < N_BINS * MIN_BIN_EVENTS:
            continue

        vals = [float(x) for x in feat_df[feat].to_list()]
        edges = compute_bin_edges(vals, N_BINS)
        if len(edges) < 2:
            continue

        tmp_rows = feat_df.to_dicts()
        bin_records = []

        for row in tmp_rows:
            x = row.get(feat)
            if x is None:
                continue
            b = assign_bin(float(x), edges)
            row["feature_bin"] = b
            bin_records.append(row)

        tmp = pl.from_dicts(bin_records)

        grouped = (
            tmp.group_by("feature_bin")
            .agg([
                pl.len().alias("n_entered"),
                pl.col("entry_delay_s").mean().alias("mean_entry_delay_s"),
                pl.col("post_entry_mfe_pts").mean().alias("mean_post_entry_mfe_pts"),
                pl.col("post_entry_mae_pts").mean().alias("mean_post_entry_mae_pts"),
                pl.col("post_entry_end_ret_pts").mean().alias("mean_post_entry_end_ret_pts"),
                (pl.col("post_entry_mfe_pts") >= FAVORABLE_TARGET_PTS).mean().alias("hit_fav_target"),
                (pl.col("post_entry_mae_pts") >= ADVERSE_PENALTY_LEVEL_PTS).mean().alias("hit_adv_penalty"),
                pl.col(feat).min().alias("feature_min"),
                pl.col(feat).max().alias("feature_max"),
                pl.col(feat).median().alias("feature_median"),
            ])
            .sort("feature_bin")
        )

        for g in grouped.to_dicts():
            n_entered = int(g["n_entered"])
            if n_entered < MIN_BIN_EVENTS:
                continue

            hit_fav = float(g["hit_fav_target"])
            hit_adv = float(g["hit_adv_penalty"])
            mean_mfe = float(g["mean_post_entry_mfe_pts"])
            mean_mae = float(g["mean_post_entry_mae_pts"])
            mean_end = float(g["mean_post_entry_end_ret_pts"])
            delay = float(g["mean_entry_delay_s"])

            # Stricter reranker:
            # reward target hit strongly, reward positive end-ret,
            # penalize adverse penetration heavily, penalize long delays mildly
            score_strict = (
                1.40 * hit_fav
                + 0.08 * mean_end
                + 0.02 * mean_mfe
                - 0.95 * hit_adv
                - 0.015 * mean_mae
                - 0.0008 * delay
            )

            summary_records.append({
                "rule_name": PARENT_RULE_NAME,
                "secondary_feature": feat,
                "feature_bin": int(g["feature_bin"]),
                "n_entered": n_entered,
                "enter_rate_vs_parent": n_entered / event_df.height,
                "mean_entry_delay_s": delay,
                "mean_post_entry_mfe_pts": mean_mfe,
                "mean_post_entry_mae_pts": mean_mae,
                "mean_post_entry_end_ret_pts": mean_end,
                "hit_fav_target_pts": FAVORABLE_TARGET_PTS,
                "hit_fav_target_rate": hit_fav,
                "hit_adv_penalty_pts": ADVERSE_PENALTY_LEVEL_PTS,
                "hit_adv_penalty_rate": hit_adv,
                "feature_min": g["feature_min"],
                "feature_max": g["feature_max"],
                "feature_median": g["feature_median"],
                "strict_score": score_strict,
            })

    if not summary_records:
        raise ValueError("No summary rows produced. Lower MIN_BIN_EVENTS or change features.")

    summary_df = (
        pl.from_dicts(summary_records)
        .sort("strict_score", descending=True)
    )
    summary_df.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop entry-quality reranker bins:")
    print(summary_df.head(30))


if __name__ == "__main__":
    main()
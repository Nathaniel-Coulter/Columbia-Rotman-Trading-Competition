# confirmation_activation_overlap_matrix.py
from __future__ import annotations

from pathlib import Path
from bisect import bisect_right
from itertools import combinations
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\confirmation_activation_overlap_matrix"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "confirmation_activation_event_level.csv"
SUMMARY_OUT = OUTPUT_DIR / "confirmation_activation_summary.csv"
PAIRWISE_OUT = OUTPUT_DIR / "confirmation_activation_pairwise.csv"

# ============================================================
# Core configuration
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "upper"

PARENT_RULE_NAME = "UNION_A2_C2"
ENTRY_ADVERSE_PTS = 1.0

# If True, only analyze refined-entry events that actually got the delayed entry
# after adverse move. This is usually what you want here.
USE_REFINED_ENTRY_ONLY = True

# minimum event count for pairwise stats to be considered "usable"
MIN_PAIR_COUNT = 25

# ============================================================
# Parent rule definitions
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
# Final 6 Features
# format:
#   feature_name: [("label", lo, hi)]
# ============================================================
FINALIST_CONFIRMATIONS: Dict[str, List[Tuple[str, float, float]]] = {
    "best_ask_queue_norm": [
        ("high_bin", 0.9811400683599983, 16.909106355664214),
    ],
    "aggressive_buy_volume": [
        ("low_bin", 70.0, 507.0),
    ],
    "ask_liquidity_shock": [
        ("high_bin", -0.0028267546300197546, 5.6005765787305695),
    ],
    "volume_per_second": [
        ("low_bin", 12.4, 111.2),
    ],
    "wall_persistence_2s": [
        ("high_bin", 0.24626006904487918, 0.5486725663716814),
    ],
    "impact_efficiency_30s": [
        ("high_bin", 0.009170653907496013, 1.1428571428571428),
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
            (pl.col("type") == 2) &
            pl.col("price").is_not_null()
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


def confirmation_pass(row: Dict[str, Any], feat: str, lo: float, hi: float) -> bool:
    v = row.get(feat)
    if v is None:
        return False
    try:
        x = float(v)
    except Exception:
        return False
    return lo <= x <= hi


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    # 2x2 table:
    # a = both true
    # b = x true, y false
    # c = x false, y true
    # d = both false
    denom = math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if denom == 0:
        return 0.0
    return (a * d - b * c) / denom


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
        pl.col("band_k").is_in(BAND_FILTER) &
        (pl.col("side") == SIDE_FILTER)
    )

    rows = df.to_dicts()
    parent_rows = [r for r in rows if parent_rule_pass(r)]
    if not parent_rows:
        raise ValueError("No rows passed parent rule.")

    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    print("\n--------------------------------------")
    print("Confirmation Activation / Overlap Matrix")
    print("--------------------------------------")
    print(f"HORIZON_S               = {HORIZON_S}")
    print(f"BAND_FILTER             = {BAND_FILTER}")
    print(f"SIDE_FILTER             = {SIDE_FILTER}")
    print(f"PARENT_RULE_NAME        = {PARENT_RULE_NAME}")
    print(f"ENTRY_ADVERSE_PTS       = {ENTRY_ADVERSE_PTS}")
    print(f"USE_REFINED_ENTRY_ONLY  = {USE_REFINED_ENTRY_ONLY}")
    print(f"N_CONFIRMATIONS         = {len(FINALIST_CONFIRMATIONS)}")
    print(f"Rows analyzed           = {len(rows):,}")
    print(f"Parent rule rows        = {len(parent_rows):,}")

    # --------------------------------------------------------
    # Build event-level refined-entry dataset
    # --------------------------------------------------------
    parent_rows_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for r in parent_rows:
        parent_rows_by_day.setdefault(r["date_ymd"], []).append(r)

    event_level_records: List[Dict[str, Any]] = []
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

            entry_idx = None
            entry_ts = None
            entry_px = None

            # ------------------------------------------------
            # Stage 1: find refined entry after adverse move
            # ------------------------------------------------
            for j in range(j0, len(ts_list)):
                dt_s = (ts_list[j] - event_ts).total_seconds()
                if dt_s > HORIZON_S:
                    break

                px = px_list[j]
                signed_move_from_event = (px - entry_price) * direction
                adverse_move = -signed_move_from_event

                if adverse_move >= ENTRY_ADVERSE_PTS:
                    entry_idx = j
                    entry_ts = ts_list[j]
                    entry_px = px
                    break

            if USE_REFINED_ENTRY_ONLY and entry_idx is None:
                continue

            if entry_idx is None:
                entry_idx = j0
                entry_ts = ts_list[j0]
                entry_px = entry_price

            # ------------------------------------------------
            # Stage 2: find full horizon end from original event
            # ------------------------------------------------
            horizon_end_idx = entry_idx
            for k in range(entry_idx, len(ts_list)):
                dt_s = (ts_list[k] - event_ts).total_seconds()
                if dt_s > HORIZON_S:
                    break
                horizon_end_idx = k

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

            for feat in FINALIST_CONFIRMATIONS:
                rec[feat] = r.get(feat)

            event_level_records.append(rec)

        processed_days += 1
        if processed_days % 25 == 0:
            print(f"...processed {processed_days} days")

    if not event_level_records:
        raise ValueError("No event-level rows produced.")

    event_df = pl.from_dicts(event_level_records).sort("ts")
    event_df.write_csv(EVENT_LEVEL_OUT)

    event_rows = event_df.to_dicts()
    n_total = len(event_rows)

    print(f"Event rows analyzed     = {n_total:,}")

    # --------------------------------------------------------
    # Build confirmation flags
    # --------------------------------------------------------
    conf_defs: List[Tuple[str, str, str, float, float]] = []
    conf_names: List[str] = []

    for feat, label_ranges in FINALIST_CONFIRMATIONS.items():
        for label, lo, hi in label_ranges:
            conf_name = f"{feat}__{label}"
            conf_names.append(conf_name)
            conf_defs.append((conf_name, feat, label, lo, hi))

    for row in event_rows:
        for conf_name, feat, label, lo, hi in conf_defs:
            row[conf_name] = confirmation_pass(row, feat, lo, hi)

    # --------------------------------------------------------
    # Single-confirmation summary
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for conf_name, feat, label, lo, hi in conf_defs:
        selected = [r for r in event_rows if bool(r[conf_name])]
        n_selected = len(selected)
        activation_rate = n_selected / n_total if n_total > 0 else math.nan

        if n_selected > 0:
            hit_rate_4pt = sum(float(r["post_entry_mfe_pts"]) >= 4.0 for r in selected) / n_selected
            adv_rate_6pt = sum(float(r["post_entry_mae_pts"]) >= 6.0 for r in selected) / n_selected
            avg_mr = sum(float(r["post_entry_mfe_pts"]) for r in selected) / n_selected
            avg_mae = sum(float(r["post_entry_mae_pts"]) for r in selected) / n_selected
            avg_end = sum(float(r["post_entry_end_ret_pts"]) for r in selected) / n_selected
        else:
            hit_rate_4pt = math.nan
            adv_rate_6pt = math.nan
            avg_mr = math.nan
            avg_mae = math.nan
            avg_end = math.nan

        summary_records.append({
            "confirmation_name": conf_name,
            "feature": feat,
            "label": label,
            "feature_lo": lo,
            "feature_hi": hi,
            "n_selected": n_selected,
            "activation_rate": activation_rate,
            "hit_rate_4pt": hit_rate_4pt,
            "adv_rate_6pt": adv_rate_6pt,
            "avg_mr_pts": avg_mr,
            "avg_mae_pts": avg_mae,
            "avg_end_ret_pts": avg_end,
        })

    summary_df = (
        pl.from_dicts(summary_records)
        .sort(["activation_rate", "hit_rate_4pt"], descending=[True, True])
    )
    summary_df.write_csv(SUMMARY_OUT)

    # --------------------------------------------------------
    # Pairwise overlap / activation matrix stats
    # --------------------------------------------------------
    pair_records: List[Dict[str, Any]] = []

    for c1, c2 in combinations(conf_names, 2):
        x = [bool(r[c1]) for r in event_rows]
        y = [bool(r[c2]) for r in event_rows]

        a = sum(1 for i in range(n_total) if x[i] and y[i])          # both
        b = sum(1 for i in range(n_total) if x[i] and not y[i])      # x only
        c = sum(1 for i in range(n_total) if (not x[i]) and y[i])    # y only
        d = sum(1 for i in range(n_total) if (not x[i]) and (not y[i]))

        n1 = a + b
        n2 = a + c
        union_n = a + b + c

        jaccard = a / union_n if union_n > 0 else 0.0
        overlap_over_1 = a / n1 if n1 > 0 else 0.0
        overlap_over_2 = a / n2 if n2 > 0 else 0.0
        min_overlap = min(overlap_over_1, overlap_over_2)
        phi = phi_coefficient(a, b, c, d)

        if a >= MIN_PAIR_COUNT:
            both_rows = [r for r in event_rows if bool(r[c1]) and bool(r[c2])]
            hit_rate_4pt_both = sum(float(r["post_entry_mfe_pts"]) >= 4.0 for r in both_rows) / a
            adv_rate_6pt_both = sum(float(r["post_entry_mae_pts"]) >= 6.0 for r in both_rows) / a
            avg_mr_both = sum(float(r["post_entry_mfe_pts"]) for r in both_rows) / a
            avg_mae_both = sum(float(r["post_entry_mae_pts"]) for r in both_rows) / a
            avg_end_both = sum(float(r["post_entry_end_ret_pts"]) for r in both_rows) / a
            pair_status = "usable"
        else:
            hit_rate_4pt_both = math.nan
            adv_rate_6pt_both = math.nan
            avg_mr_both = math.nan
            avg_mae_both = math.nan
            avg_end_both = math.nan
            pair_status = "sparse"

        redundancy_flag = (
            "very_high_overlap"
            if jaccard >= 0.80 or min_overlap >= 0.90
            else "high_overlap"
            if jaccard >= 0.60 or min_overlap >= 0.75
            else "moderate_overlap"
            if jaccard >= 0.35 or min_overlap >= 0.50
            else "low_overlap"
        )

        pair_records.append({
            "confirmation_1": c1,
            "confirmation_2": c2,
            "n_conf_1": n1,
            "n_conf_2": n2,
            "intersection_n": a,
            "union_n": union_n,
            "jaccard_overlap": jaccard,
            "intersection_over_conf_1": overlap_over_1,
            "intersection_over_conf_2": overlap_over_2,
            "min_confirmation_coverage_overlap": min_overlap,
            "phi_correlation": phi,
            "pair_status": pair_status,
            "redundancy_flag": redundancy_flag,
            "hit_rate_4pt_both": hit_rate_4pt_both,
            "adv_rate_6pt_both": adv_rate_6pt_both,
            "avg_mr_pts_both": avg_mr_both,
            "avg_mae_pts_both": avg_mae_both,
            "avg_end_ret_pts_both": avg_end_both,
        })

    pair_df = (
        pl.from_dicts(pair_records)
        .sort(
            ["jaccard_overlap", "min_confirmation_coverage_overlap", "phi_correlation"],
            descending=[True, True, True]
        )
    )
    pair_df.write_csv(PAIRWISE_OUT)

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")
    print(f"Wrote: {PAIRWISE_OUT}")

    print("\nTop single-confirmation activations:")
    print(
        summary_df.select([
            "confirmation_name",
            "n_selected",
            "activation_rate",
            "hit_rate_4pt",
            "adv_rate_6pt",
            "avg_mr_pts",
            "avg_mae_pts",
            "avg_end_ret_pts",
        ]).head(20)
    )

    print("\nTop pairwise overlaps:")
    print(
        pair_df.select([
            "confirmation_1",
            "confirmation_2",
            "intersection_n",
            "jaccard_overlap",
            "intersection_over_conf_1",
            "intersection_over_conf_2",
            "min_confirmation_coverage_overlap",
            "phi_correlation",
            "pair_status",
            "redundancy_flag",
            "hit_rate_4pt_both",
            "adv_rate_6pt_both",
            "avg_mr_pts_both",
            "avg_mae_pts_both",
        ]).head(30)
    )


if __name__ == "__main__":
    main()
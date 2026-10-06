# entry_mechanism_comparator.py
from __future__ import annotations

from pathlib import Path
from bisect import bisect_right
from typing import Dict, List, Tuple, Any
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\entry_mechanism_comparator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "entry_mechanism_event_level.csv"
SUMMARY_OUT = OUTPUT_DIR / "entry_mechanism_summary.csv"

# ============================================================
# Core configuration
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "lower"

# We compare these parent signals
PARENT_RULE_NAMES: List[str] = [
    "SINGLE_A2",
    "SINGLE_C1",
    "ALL2_A2_C1",
]

# Refined entry logic:
# only "enter" once the original trigger has gone this far adverse
ENTRY_ADVERSE_PTS = 1.0

# Evaluation thresholds after refined entry
FAVORABLE_LEVELS_PTS = [2.0, 4.0, 6.0, 8.0]
ADVERSE_LEVELS_PTS = [4.0, 6.0, 8.0, 10.0]

# Mechanism binning
N_BINS = 2
MIN_BIN_EVENTS = 25

# ============================================================
# Parent rule definitions
# ============================================================
RULES: Dict[str, List[Tuple[str, str, float, float]]] = {
    # A2_depth_near_ask__cancel_rate_ask
    "A2": [
        ("depth_near_ask", "between", 127.0, 182.0),
        ("cancel_rate_ask", "between", 3.8, 95.8),
    ],
    # C1_realized_vol_1m__depth_mid_ask
    "C1": [
        ("realized_vol_1m", "between", 0.000694397, 0.00493388),
        ("depth_mid_ask", "between", 5.0, 149.0),
    ],
    # B1_depth_mid_ask__queue_imbalance_std_5s
    "B1": [
        ("depth_mid_ask", "between", 5.0, 150.0),
        ("queue_imbalance_std_5s", "between", 0.194964, 0.528415),
    ],
    # D1_cancel_rate_ask__bid_depth_curvature
    "D1": [
        ("cancel_rate_ask", "between", 6.8, 222.2),
        ("bid_depth_curvature", "between", -6.875, 25.375),
    ],
    # E1_cancel_rate_ask__cancel_ratio
    "E1": [
        ("cancel_rate_ask", "between", 6.8, 222.2),
        ("cancel_ratio", "between", 0.0805134, 0.319363),
    ],
}

# ============================================================
# Mechanism families
# Edit these freely later
# ============================================================
MECHANISMS: Dict[str, List[str]] = {
    "FLOW_EXHAUSTION": [
        "trade_sign_autocorr_60s",
        "trade_sign_autocorr_30s",
        "flow_toxicity",
        "flow_toxicity_10s",
        "absorption_ratio",
        "microprice_deviation",
        "microprice_pressure",
    ],
    "LIQUIDITY_STRUCTURE": [
        "wall_strength_opposite_3",
        "wall_strength_opposite_5",
        "wall_strength_opposite_3_2s_avg",
        "wall_strength_opposite_3_5s_avg",
        "wall_strength_opposite_5_2s_avg",
        "wall_strength_opposite_5_5s_avg",
        "wall_persistence_2s",
        "liquidity_compression",
        "weighted_OBI_l10",
        "OBI_l5",
        "queue_imbalance_mean_5s",
        "replenishment_rate_near_5s",
    ],
    "IMPACT_DECAY": [
        "impact_decay_5s",
        "impact_efficiency_3s",
        "impact_asymmetry_10s",
        "flow_impact",
        "convexity_imbalance",
        "bid_depth_curvature",
    ],
}

# Optional structural control family
MECHANISMS["STRUCTURAL_CONTROL"] = [
    "dist_day_low",
    "distance_from_prior_day_high",
    "distance_from_overnight_high",
    "z_anchor_vwap",
    "distance_from_session_vwap_anchor",
]

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
    out: Dict[str, Path] = {}
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
        df.filter((pl.col("type") == 2) & pl.col("price").is_not_null())
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


def parent_rule_pass(row: Dict[str, Any], parent_rule_name: str) -> bool:
    if parent_rule_name == "ALL2_A2_C1":
        return event_passes_rule(row, RULES["A2"]) and event_passes_rule(row, RULES["C1"])
    elif parent_rule_name == "SINGLE_A2":
        return event_passes_rule(row, RULES["A2"])
    elif parent_rule_name == "SINGLE_C1":
        return event_passes_rule(row, RULES["C1"])
    elif parent_rule_name == "VOTE_B1_C1_D1":
        votes = 0
        for nm in ["B1", "C1", "D1"]:
            if event_passes_rule(row, RULES[nm]):
                votes += 1
        return votes >= 2
    elif parent_rule_name == "VOTE4_A2_B1_C1_E1":
        votes = 0
        for nm in ["A2", "B1", "C1", "E1"]:
            if event_passes_rule(row, RULES[nm]):
                votes += 1
        return votes >= 2
    else:
        raise ValueError(f"Unsupported parent rule: {parent_rule_name}")


def compute_bin_edges(values: List[float], n_bins: int) -> List[float]:
    xs = sorted(values)
    if len(xs) == 0:
        return []
    edges: List[float] = []
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
# Raw-path refined entry simulator for a given event
# ============================================================
def simulate_refined_entry(
    event_ts,
    entry_price_trigger: float,
    ts_list: List,
    px_list: List[float],
) -> Dict[str, float] | None:
    j0 = bisect_right(ts_list, event_ts) - 1
    if j0 < 0:
        return None

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

        # lower-band long logic: adverse move = entry_price - current_px
        adverse_move = entry_price_trigger - px
        if adverse_move >= ENTRY_ADVERSE_PTS:
            entry_idx = j
            entry_ts = ts_list[j]
            entry_px = px
            break

    if entry_idx is None:
        return None

    future_px = px_list[entry_idx:horizon_end_idx + 1]
    if not future_px:
        return None

    post_mfe = max(p - entry_px for p in future_px)
    post_mae = max(entry_px - p for p in future_px)
    post_end = future_px[-1] - entry_px

    out = {
        "entry_price_refined": entry_px,
        "entry_delay_s": (entry_ts - event_ts).total_seconds(),
        "post_entry_mfe_pts": post_mfe,
        "post_entry_mae_pts": post_mae,
        "post_entry_end_ret_pts": post_end,
    }

    for lvl in FAVORABLE_LEVELS_PTS:
        out[f"hit_fav_{lvl:.1f}pt"] = float(post_mfe >= lvl)
    for lvl in ADVERSE_LEVELS_PTS:
        out[f"hit_adv_{lvl:.1f}pt"] = float(post_mae >= lvl)

    return out


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
    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    print("\n--------------------------------------")
    print("Entry Mechanism Comparator")
    print("--------------------------------------")
    print(f"HORIZON_S        = {HORIZON_S}")
    print(f"BAND_FILTER      = {BAND_FILTER}")
    print(f"SIDE_FILTER      = {SIDE_FILTER}")
    print(f"ENTRY_ADVERSE_PTS= {ENTRY_ADVERSE_PTS}")
    print(f"N_BINS           = {N_BINS}")
    print(f"MIN_BIN_EVENTS   = {MIN_BIN_EVENTS}")
    print(f"Rows analyzed    = {len(rows):,}")

    # --------------------------------------------------------
    # Step 1: build refined-entry event-level table once
    # --------------------------------------------------------
    rows_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_day.setdefault(r["date_ymd"], []).append(r)

    event_level_records: List[Dict[str, Any]] = []

    processed_days = 0
    for day_ymd, evs in rows_by_day.items():
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
            if entry_price is None or event_ts is None:
                continue

            sim = simulate_refined_entry(
                event_ts=event_ts,
                entry_price_trigger=float(entry_price),
                ts_list=ts_list,
                px_list=px_list,
            )
            if sim is None:
                continue

            rec = {
                "event_id": r["event_id"],
                "date_ymd": r["date_ymd"],
                "ts": r["ts"],
                "band_k": r["band_k"],
                "side": r["side"],
                "entry_price_trigger": float(entry_price),
            }

            # include all parent-rule and mechanism features so we can test later
            all_feats = set()
            for conds in RULES.values():
                for feat, _, _, _ in conds:
                    all_feats.add(feat)
            for feats in MECHANISMS.values():
                all_feats.update(feats)

            for feat in all_feats:
                rec[feat] = r.get(feat)

            for parent_rule_name in PARENT_RULE_NAMES:
                rec[f"parent_{parent_rule_name}"] = int(parent_rule_pass(r, parent_rule_name))

            rec.update(sim)
            event_level_records.append(rec)

        processed_days += 1
        if processed_days % 25 == 0:
            print(f"...processed {processed_days} days")

    if not event_level_records:
        raise ValueError("No refined-entry event rows were produced.")

    event_df = pl.from_dicts(event_level_records)
    event_df.write_csv(EVENT_LEVEL_OUT)

    print(f"Refined-entry event rows = {event_df.height:,}")

    # --------------------------------------------------------
    # Step 2: compare parent-only vs parent+mechanism-best-bin
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for parent_rule_name in PARENT_RULE_NAMES:
        parent_df = event_df.filter(pl.col(f"parent_{parent_rule_name}") == 1)
        if parent_df.height == 0:
            continue

        # parent-only baseline
        base_rec: Dict[str, Any] = {
            "parent_rule": parent_rule_name,
            "mechanism_name": "BASE_ONLY",
            "selected_feature": None,
            "selected_bin": None,
            "n_entered": parent_df.height,
            "selection_rate_vs_parent": 1.0,
            "mean_entry_delay_s": float(parent_df["entry_delay_s"].mean()),
            "mean_post_entry_mfe_pts": float(parent_df["post_entry_mfe_pts"].mean()),
            "mean_post_entry_mae_pts": float(parent_df["post_entry_mae_pts"].mean()),
            "mean_post_entry_end_ret_pts": float(parent_df["post_entry_end_ret_pts"].mean()),
        }
        for lvl in FAVORABLE_LEVELS_PTS:
            base_rec[f"hit_fav_{lvl:.1f}pt"] = float(parent_df[f"hit_fav_{lvl:.1f}pt"].mean())
        for lvl in ADVERSE_LEVELS_PTS:
            base_rec[f"hit_adv_{lvl:.1f}pt"] = float(parent_df[f"hit_adv_{lvl:.1f}pt"].mean())

        base_rec["mechanism_score"] = (
            1.20 * base_rec["hit_fav_4.0pt"]
            + 0.10 * base_rec["mean_post_entry_end_ret_pts"]
            + 0.02 * base_rec["mean_post_entry_mfe_pts"]
            - 1.00 * base_rec["hit_adv_6.0pt"]
            - 0.02 * base_rec["mean_post_entry_mae_pts"]
            - 0.0008 * base_rec["mean_entry_delay_s"]
        )
        summary_records.append(base_rec)

        # mechanism family tests
        for mech_name, mech_feats in MECHANISMS.items():
            best_row: Dict[str, Any] | None = None

            for feat in mech_feats:
                if feat not in parent_df.columns:
                    continue

                feat_df = parent_df.filter(pl.col(feat).is_not_null())
                if feat_df.height < max(MIN_BIN_EVENTS * N_BINS, MIN_BIN_EVENTS):
                    continue

                vals = [float(x) for x in feat_df[feat].to_list()]
                edges = compute_bin_edges(vals, N_BINS)
                if len(edges) < 2:
                    continue

                tmp_rows = feat_df.to_dicts()
                tmp_bin_records = []
                for row in tmp_rows:
                    x = row.get(feat)
                    if x is None:
                        continue
                    row["feature_bin"] = assign_bin(float(x), edges)
                    tmp_bin_records.append(row)

                tmp = pl.from_dicts(tmp_bin_records)
                grouped = (
                    tmp.group_by("feature_bin")
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
                            pl.col(f"hit_fav_{lvl:.1f}pt").mean().alias(f"hit_fav_{lvl:.1f}pt")
                            for lvl in FAVORABLE_LEVELS_PTS
                        ],
                        *[
                            pl.col(f"hit_adv_{lvl:.1f}pt").mean().alias(f"hit_adv_{lvl:.1f}pt")
                            for lvl in ADVERSE_LEVELS_PTS
                        ],
                    ])
                    .sort("feature_bin")
                )

                for g in grouped.to_dicts():
                    n_entered = int(g["n_entered"])
                    if n_entered < MIN_BIN_EVENTS:
                        continue

                    score = (
                        1.20 * float(g["hit_fav_4.0pt"])
                        + 0.10 * float(g["mean_post_entry_end_ret_pts"])
                        + 0.02 * float(g["mean_post_entry_mfe_pts"])
                        - 1.00 * float(g["hit_adv_6.0pt"])
                        - 0.02 * float(g["mean_post_entry_mae_pts"])
                        - 0.0008 * float(g["mean_entry_delay_s"])
                    )

                    rec = {
                        "parent_rule": parent_rule_name,
                        "mechanism_name": mech_name,
                        "selected_feature": feat,
                        "selected_bin": int(g["feature_bin"]),
                        "n_entered": n_entered,
                        "selection_rate_vs_parent": n_entered / parent_df.height,
                        "mean_entry_delay_s": float(g["mean_entry_delay_s"]),
                        "mean_post_entry_mfe_pts": float(g["mean_post_entry_mfe_pts"]),
                        "mean_post_entry_mae_pts": float(g["mean_post_entry_mae_pts"]),
                        "mean_post_entry_end_ret_pts": float(g["mean_post_entry_end_ret_pts"]),
                        "feature_min": g["feature_min"],
                        "feature_max": g["feature_max"],
                        "feature_median": g["feature_median"],
                        "mechanism_score": score,
                    }
                    for lvl in FAVORABLE_LEVELS_PTS:
                        rec[f"hit_fav_{lvl:.1f}pt"] = float(g[f"hit_fav_{lvl:.1f}pt"])
                    for lvl in ADVERSE_LEVELS_PTS:
                        rec[f"hit_adv_{lvl:.1f}pt"] = float(g[f"hit_adv_{lvl:.1f}pt"])

                    if best_row is None or rec["mechanism_score"] > best_row["mechanism_score"]:
                        best_row = rec

            if best_row is not None:
                summary_records.append(best_row)

    if not summary_records:
        raise ValueError("No summary rows produced.")

    summary_df = (
        pl.from_dicts(summary_records)
        .sort(["parent_rule", "mechanism_score"], descending=[False, True])
    )
    summary_df.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop mechanism comparison rows:")
    print(summary_df)

if __name__ == "__main__":
    main()
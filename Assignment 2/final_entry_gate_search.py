# final_entry_gate_search.py
from __future__ import annotations

from bisect import bisect_right
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\final_entry_gate_search"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "final_entry_gate_event_level.csv"
FOLD_RESULTS_OUT = OUTPUT_DIR / "final_entry_gate_fold_results.csv"
SUMMARY_OUT = OUTPUT_DIR / "final_entry_gate_summary.csv"


# ============================================================
# Core config
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "upper"

PARENT_RULE_NAME = "UNION_A2_C2"
ENTRY_ADVERSE_PTS = 1.0

N_FOLDS = 4              # 4 is a nice compromise from your recent tests
MIN_TEST_EVENTS = 20     # minimum test events for a fold to count as valid

FAVORABLE_TARGET_PTS = 4.0
ADVERSE_PENALTY_LEVEL_PTS = 6.0


# ============================================================
# Parent rules
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
# Final 6 confirmations
# format:
#   feature_name: [("label", lo, hi)]
# ============================================================
FINAL_CONFIRMATIONS: Dict[str, List[Tuple[str, float, float]]] = {
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
# Gate search settings
# ============================================================
USE_SINGLES = True
USE_ALL_2 = True
USE_ALL_3 = True
USE_VOTE_2_OF_3 = True
USE_HANDPICKED_VOTE_2_OF_4 = True

# hand-picked 2-of-4 sets so we do not explode combinatorially
HANDPICKED_2OF4: List[Tuple[str, str, str, str]] = [
    (
        "best_ask_queue_norm__high_bin",
        "aggressive_buy_volume__low_bin",
        "ask_liquidity_shock__high_bin",
        "volume_per_second__low_bin",
    ),
    (
        "best_ask_queue_norm__high_bin",
        "aggressive_buy_volume__low_bin",
        "wall_persistence_2s__high_bin",
        "impact_efficiency_30s__high_bin",
    ),
    (
        "ask_liquidity_shock__high_bin",
        "volume_per_second__low_bin",
        "wall_persistence_2s__high_bin",
        "impact_efficiency_30s__high_bin",
    ),
    (
        "best_ask_queue_norm__high_bin",
        "ask_liquidity_shock__high_bin",
        "volume_per_second__low_bin",
        "wall_persistence_2s__high_bin",
    ),
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


def event_passes_rule(
    row: Dict[str, Any],
    rule_conds: List[Tuple[str, str, float, float]],
) -> bool:
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


def fold_boundaries(n: int, n_folds: int) -> List[Tuple[int, int]]:
    out = []
    for i in range(n_folds):
        lo = int(round(i * n / n_folds))
        hi = int(round((i + 1) * n / n_folds))
        out.append((lo, hi))
    return out


def safe_mean(xs: List[float]) -> float:
    xs2 = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs2:
        return math.nan
    return sum(xs2) / len(xs2)


def safe_std(xs: List[float]) -> float:
    xs2 = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if len(xs2) <= 1:
        return 0.0
    mu = sum(xs2) / len(xs2)
    var = sum((x - mu) ** 2 for x in xs2) / (len(xs2) - 1)
    return math.sqrt(var)


def summarize_slice(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {
            "n_events": 0,
            "hit_rate_4pt": math.nan,
            "adv_rate_6pt": math.nan,
            "avg_mr_pts": math.nan,
            "avg_mae_pts": math.nan,
            "avg_end_ret_pts": math.nan,
        }

    mfe = [float(r["post_entry_mfe_pts"]) for r in rows]
    mae = [float(r["post_entry_mae_pts"]) for r in rows]
    end_ret = [float(r["post_entry_end_ret_pts"]) for r in rows]

    return {
        "n_events": n,
        "hit_rate_4pt": sum(x >= FAVORABLE_TARGET_PTS for x in mfe) / n,
        "adv_rate_6pt": sum(x >= ADVERSE_PENALTY_LEVEL_PTS for x in mae) / n,
        "avg_mr_pts": sum(mfe) / n,
        "avg_mae_pts": sum(mae) / n,
        "avg_end_ret_pts": sum(end_ret) / n,
    }


def coverage_rank(n_valid_test_folds: int, n_folds: int) -> int:
    # 0 = best coverage, larger = worse
    return max(0, n_folds - 1 - n_valid_test_folds)


def make_combo_name(prefix: str, feats: Tuple[str, ...]) -> str:
    return f"{prefix}__" + "__".join(feats)


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
    print("Final Entry Gate Search")
    print("--------------------------------------")
    print(f"HORIZON_S              = {HORIZON_S}")
    print(f"BAND_FILTER            = {BAND_FILTER}")
    print(f"SIDE_FILTER            = {SIDE_FILTER}")
    print(f"PARENT_RULE_NAME       = {PARENT_RULE_NAME}")
    print(f"ENTRY_ADVERSE_PTS      = {ENTRY_ADVERSE_PTS}")
    print(f"N_FOLDS                = {N_FOLDS}")
    print(f"N_CONFIRMATIONS        = {len(FINAL_CONFIRMATIONS)}")
    print(f"Rows analyzed          = {len(rows):,}")
    print(f"Parent rule rows       = {len(parent_rows):,}")

    # --------------------------------------------------------
    # build refined-entry event-level dataset from raw path
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
            event_ts = r.get("ts")
            trigger_px = r.get("entry_price")
            side = r.get("side")

            if event_ts is None or trigger_px is None or side is None:
                continue

            trigger_px = float(trigger_px)
            direction = 1.0 if side == "lower" else -1.0

            j0 = bisect_right(ts_list, event_ts) - 1
            if j0 < 0:
                continue

            entry_idx = None
            entry_ts = None
            entry_px = None

            # 1) find refined entry after adverse move
            for j in range(j0, len(ts_list)):
                dt_s = (ts_list[j] - event_ts).total_seconds()
                if dt_s > HORIZON_S:
                    break

                px = px_list[j]
                signed_move_from_event = (px - trigger_px) * direction
                adverse_move = -signed_move_from_event

                if adverse_move >= ENTRY_ADVERSE_PTS:
                    entry_idx = j
                    entry_ts = ts_list[j]
                    entry_px = px
                    break

            if entry_idx is None:
                continue

            # 2) now scan forward from entry_idx to the horizon end
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
            entry_delay_s = (entry_ts - event_ts).total_seconds()

            rec = {
                "event_id": r["event_id"],
                "date_ymd": r["date_ymd"],
                "ts": r["ts"],
                "entry_price_trigger": trigger_px,
                "entry_price_refined": entry_px,
                "entry_delay_s": entry_delay_s,
                "post_entry_mfe_pts": post_mfe,
                "post_entry_mae_pts": post_mae,
                "post_entry_end_ret_pts": end_ret,
            }

            for feat in FINAL_CONFIRMATIONS:
                rec[feat] = r.get(feat)

            event_level_records.append(rec)

        processed_days += 1
        if processed_days % 25 == 0:
            print(f"...processed {processed_days} days")

    if not event_level_records:
        raise ValueError("No refined-entry events produced.")

    event_df = pl.from_dicts(event_level_records).sort("ts")
    event_df.write_csv(EVENT_LEVEL_OUT)

    event_rows = event_df.to_dicts()
    print(f"Refined-entry event rows = {len(event_rows):,}")

    # --------------------------------------------------------
    # attach confirmation booleans
    # --------------------------------------------------------
    confirmation_names: List[str] = []
    conf_meta: Dict[str, Tuple[str, str, float, float]] = {}

    for feat, ranges in FINAL_CONFIRMATIONS.items():
        for label, lo, hi in ranges:
            cname = f"{feat}__{label}"
            confirmation_names.append(cname)
            conf_meta[cname] = (feat, label, lo, hi)

    for row in event_rows:
        for cname in confirmation_names:
            feat, _label, lo, hi = conf_meta[cname]
            row[cname] = confirmation_pass(row, feat, lo, hi)

    # --------------------------------------------------------
    # build candidate gate definitions
    # --------------------------------------------------------
    gate_defs: List[Dict[str, Any]] = []

    if USE_SINGLES:
        for c in confirmation_names:
            gate_defs.append({
                "gate_name": make_combo_name("SINGLE", (c,)),
                "logic": "SINGLE",
                "confirmations": (c,),
            })

    if USE_ALL_2:
        for feats in combinations(confirmation_names, 2):
            gate_defs.append({
                "gate_name": make_combo_name("ALL2", feats),
                "logic": "ALL_2",
                "confirmations": feats,
            })

    if USE_ALL_3:
        for feats in combinations(confirmation_names, 3):
            gate_defs.append({
                "gate_name": make_combo_name("ALL3", feats),
                "logic": "ALL_3",
                "confirmations": feats,
            })

    if USE_VOTE_2_OF_3:
        for feats in combinations(confirmation_names, 3):
            gate_defs.append({
                "gate_name": make_combo_name("VOTE2OF3", feats),
                "logic": "AT_LEAST_2_OF_3",
                "confirmations": feats,
            })

    if USE_HANDPICKED_VOTE_2_OF_4:
        for feats in HANDPICKED_2OF4:
            gate_defs.append({
                "gate_name": make_combo_name("VOTE2OF4", feats),
                "logic": "AT_LEAST_2_OF_4",
                "confirmations": feats,
            })

    # dedupe just in case
    seen = set()
    deduped_defs = []
    for g in gate_defs:
        key = (g["logic"], g["confirmations"])
        if key in seen:
            continue
        seen.add(key)
        deduped_defs.append(g)
    gate_defs = deduped_defs

    # --------------------------------------------------------
    # walk-forward evaluation
    # anchored train = everything before test block
    # --------------------------------------------------------
    fold_ranges = fold_boundaries(len(event_rows), N_FOLDS)
    fold_records: List[Dict[str, Any]] = []

    for fold_idx, (test_lo, test_hi) in enumerate(fold_ranges, start=1):
        train_rows_all = event_rows[:test_lo]
        test_rows_all = event_rows[test_lo:test_hi]

        if not train_rows_all or not test_rows_all:
            continue

        for gate in gate_defs:
            logic = gate["logic"]
            feats = gate["confirmations"]

            def gate_pass(row: Dict[str, Any]) -> bool:
                vals = [bool(row[f]) for f in feats]

                if logic == "SINGLE":
                    return vals[0]
                if logic == "ALL_2":
                    return all(vals)
                if logic == "ALL_3":
                    return all(vals)
                if logic == "AT_LEAST_2_OF_3":
                    return sum(vals) >= 2
                if logic == "AT_LEAST_2_OF_4":
                    return sum(vals) >= 2

                raise ValueError(f"Unsupported logic: {logic}")

            train_rows = [r for r in train_rows_all if gate_pass(r)]
            test_rows = [r for r in test_rows_all if gate_pass(r)]

            train_stats = summarize_slice(train_rows)
            test_stats = summarize_slice(test_rows)

            hit_deg = (
                test_stats["hit_rate_4pt"] - train_stats["hit_rate_4pt"]
                if not math.isnan(test_stats["hit_rate_4pt"]) and not math.isnan(train_stats["hit_rate_4pt"])
                else math.nan
            )
            mr_deg = (
                test_stats["avg_mr_pts"] - train_stats["avg_mr_pts"]
                if not math.isnan(test_stats["avg_mr_pts"]) and not math.isnan(train_stats["avg_mr_pts"])
                else math.nan
            )

            fold_records.append({
                "gate_name": gate["gate_name"],
                "logic": logic,
                "confirmations": " | ".join(feats),
                "fold_idx": fold_idx,

                "train_n_events": train_stats["n_events"],
                "train_hit_rate_4pt": train_stats["hit_rate_4pt"],
                "train_adv_rate_6pt": train_stats["adv_rate_6pt"],
                "train_avg_mr_pts": train_stats["avg_mr_pts"],
                "train_avg_mae_pts": train_stats["avg_mae_pts"],
                "train_avg_end_ret_pts": train_stats["avg_end_ret_pts"],

                "test_n_events": test_stats["n_events"],
                "test_hit_rate_4pt": test_stats["hit_rate_4pt"],
                "test_adv_rate_6pt": test_stats["adv_rate_6pt"],
                "test_avg_mr_pts": test_stats["avg_mr_pts"],
                "test_avg_mae_pts": test_stats["avg_mae_pts"],
                "test_avg_end_ret_pts": test_stats["avg_end_ret_pts"],

                "hit_rate_degradation_test_minus_train": hit_deg,
                "avg_mr_pts_degradation_test_minus_train": mr_deg,
            })

    if not fold_records:
        raise ValueError("No fold records produced.")

    fold_df = pl.from_dicts(fold_records)
    fold_df.write_csv(FOLD_RESULTS_OUT)

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    for gate_name in fold_df["gate_name"].unique().to_list():
        sub = fold_df.filter(pl.col("gate_name") == gate_name)
        valid = sub.filter(pl.col("test_n_events") >= MIN_TEST_EVENTS)

        n_valid = valid.height
        cov_rank = coverage_rank(n_valid, N_FOLDS)
        coverage_status = "ok" if n_valid > 0 else "too_sparse"

        target_df = valid if n_valid > 0 else sub

        mean_test_events = float(target_df["test_n_events"].mean()) if target_df.height > 0 else math.nan
        mean_test_hit_rate = float(target_df["test_hit_rate_4pt"].mean()) if target_df.height > 0 else math.nan
        std_test_hit_rate = float(target_df["test_hit_rate_4pt"].std()) if target_df.height > 1 else 0.0
        mean_test_adv_rate = float(target_df["test_adv_rate_6pt"].mean()) if target_df.height > 0 else math.nan
        mean_test_avg_mr = float(target_df["test_avg_mr_pts"].mean()) if target_df.height > 0 else math.nan
        std_test_avg_mr = float(target_df["test_avg_mr_pts"].std()) if target_df.height > 1 else 0.0
        mean_test_avg_mae = float(target_df["test_avg_mae_pts"].mean()) if target_df.height > 0 else math.nan
        mean_test_avg_end = float(target_df["test_avg_end_ret_pts"].mean()) if target_df.height > 0 else math.nan
        mean_hit_deg = float(target_df["hit_rate_degradation_test_minus_train"].mean()) if target_df.height > 0 else math.nan
        mean_mr_deg = float(target_df["avg_mr_pts_degradation_test_minus_train"].mean()) if target_df.height > 0 else math.nan

        # robust score: favors strong OOS hit/MR/end, penalizes adverse/MAE and negative degradation
        robust_score = (
            1.30 * (0.0 if math.isnan(mean_test_hit_rate) else mean_test_hit_rate)
            - 0.70 * (0.0 if math.isnan(mean_test_adv_rate) else mean_test_adv_rate)
            + 0.045 * (0.0 if math.isnan(mean_test_avg_mr) else mean_test_avg_mr)
            - 0.035 * (0.0 if math.isnan(mean_test_avg_mae) else mean_test_avg_mae)
            + 0.030 * (0.0 if math.isnan(mean_test_avg_end) else mean_test_avg_end)
            - 0.45 * max(0.0, -(0.0 if math.isnan(mean_hit_deg) else mean_hit_deg))
            - 0.02 * cov_rank
            + 0.02 * math.log1p(max(0.0, 0.0 if math.isnan(mean_test_events) else mean_test_events))
        )

        # stability score: same idea but explicitly punishes fold dispersion
        stability_score = (
            (0.0 if math.isnan(mean_test_hit_rate) else mean_test_hit_rate)
            - 0.55 * std_test_hit_rate
            - 0.20 * (0.0 if math.isnan(mean_test_adv_rate) else mean_test_adv_rate)
            + 0.025 * (0.0 if math.isnan(mean_test_avg_mr) else mean_test_avg_mr)
            - 0.015 * std_test_avg_mr
            - 0.01 * cov_rank
        )

        first = sub.row(0, named=True)

        summary_records.append({
            "gate_name": gate_name,
            "logic": first["logic"],
            "confirmations": first["confirmations"],
            "n_valid_test_folds": n_valid,
            "coverage_status": coverage_status,
            "coverage_rank": cov_rank,

            "mean_test_events": mean_test_events,
            "mean_test_hit_rate_4pt": mean_test_hit_rate,
            "std_test_hit_rate_4pt": std_test_hit_rate,
            "mean_test_adv_rate_6pt": mean_test_adv_rate,
            "mean_test_avg_mr_pts": mean_test_avg_mr,
            "std_test_avg_mr_pts": std_test_avg_mr,
            "mean_test_avg_mae_pts": mean_test_avg_mae,
            "mean_test_avg_end_ret_pts": mean_test_avg_end,

            "mean_hit_rate_degradation_test_minus_train": mean_hit_deg,
            "mean_avg_mr_pts_degradation_test_minus_train": mean_mr_deg,

            "robust_score": robust_score,
            "stability_score": stability_score,
        })

    summary_df = (
        pl.from_dicts(summary_records)
        .sort(
            ["coverage_rank", "robust_score", "stability_score", "mean_test_hit_rate_4pt"],
            descending=[False, True, True, True],
        )
    )
    summary_df.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {FOLD_RESULTS_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop entry-gate summaries:")
    print(summary_df.head(30))


if __name__ == "__main__":
    main()
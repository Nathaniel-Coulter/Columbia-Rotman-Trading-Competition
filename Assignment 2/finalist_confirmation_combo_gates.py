# finalist_confirmation_combo_gates.py
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\finalist_confirmation_combo_gates"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "finalist_confirmation_event_level.csv"
FOLD_RESULTS_OUT = OUTPUT_DIR / "finalist_confirmation_combo_fold_results.csv"
SUMMARY_OUT = OUTPUT_DIR / "finalist_confirmation_combo_summary.csv"

# ============================================================
# Core configuration
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "lower"

PARENT_RULE_NAME = "SINGLE_C1"
ENTRY_ADVERSE_PTS = 1.0

N_FOLDS = 3
MIN_TEST_EVENTS = 20

# Target / penalty levels for evaluation
FAVORABLE_TARGET_PTS = 4.0
ADVERSE_PENALTY_LEVEL_PTS = 6.0

# ============================================================
# Parent rule definitions
# ============================================================
RULES: Dict[str, List[Tuple[str, str, float, float]]] = {
    "A2": [
        ("depth_near_ask", "between", 127.0, 182.0),
        ("cancel_rate_ask", "between", 3.8, 95.8),
    ],
    "C1": [
        ("realized_vol_1m", "between", 0.000694397, 0.00493388),
        ("depth_mid_ask", "between", 5.0, 149.0),
    ],
    "B1": [
        ("depth_mid_ask", "between", 5.0, 150.0),
        ("queue_imbalance_std_5s", "between", 0.194964, 0.528415),
    ],
    "D1": [
        ("cancel_rate_ask", "between", 6.8, 222.2),
        ("bid_depth_curvature", "between", -6.875, 25.375),
    ],
    "E1": [
        ("cancel_rate_ask", "between", 6.8, 222.2),
        ("cancel_ratio", "between", 0.0805134, 0.319363),
    ],
}

# ============================================================
# Finalist confirmations for SINGLE_C1
# You can add/remove features here easily.
# format:
#   feature_name: [("label", lo, hi)]
# ============================================================
FINALIST_CONFIRMATIONS: Dict[str, List[Tuple[str, float, float]]] = {
    # required keepers
    "trade_sign_autocorr_60s": [
        ("low_bin", 0.5597464439006082, 0.8540285520220359),
    ],
    "wall_strength_opposite_3_2s_avg": [
        ("low_bin", 0.0, 92.50625711035268),
    ],
    "bid_depth_curvature": [
        ("high_bin", -5.25, 6.75),
    ],

    # strong finalists from later validation
    "spread": [
        ("low_bin", 0.25, 0.5),
    ],
    "microprice": [
        ("low_bin", 4901.460526315789, 5636.0),
    ],
    "depth_far_bid": [
        ("low_bin", 3.0, 115.0),
    ],
    "cancel_rate_bid": [
        ("high_bin", 11.2, 553.8),
    ],
    "depth_near_bid": [
        ("low_bin", 3.0, 84.0),
    ],
    "update_rate_bid": [
        ("high_bin", 106.2, 630.8),
    ],
    "volume_per_second": [
        ("high_bin", 103.9, 784.6),
    ],
    "microprice_minus_mid": [
        ("low_bin", -0.65625, 0.023734177215374075),
    ],
    "cancel_ratio": [
        ("high_bin", 0.08952879581151832, 0.31936284110197155),
    ],
}

# ============================================================
# Combo search settings
# ============================================================
USE_SINGLE = True
USE_ALL_2 = True
USE_ALL_3 = True
USE_VOTE_2_OF_3 = True

MAX_COMBO_SIZE = 3

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
    if PARENT_RULE_NAME == "ALL2_A2_C1":
        return event_passes_rule(row, RULES["A2"]) and event_passes_rule(row, RULES["C1"])
    elif PARENT_RULE_NAME == "SINGLE_A2":
        return event_passes_rule(row, RULES["A2"])
    elif PARENT_RULE_NAME == "SINGLE_C1":
        return event_passes_rule(row, RULES["C1"])
    elif PARENT_RULE_NAME == "VOTE_B1_C1_D1":
        votes = sum(event_passes_rule(row, RULES[nm]) for nm in ["B1", "C1", "D1"])
        return votes >= 2
    elif PARENT_RULE_NAME == "VOTE4_A2_B1_C1_E1":
        votes = sum(event_passes_rule(row, RULES[nm]) for nm in ["A2", "B1", "C1", "E1"])
        return votes >= 2
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


def make_combo_name(logic: str, feats: Tuple[str, ...]) -> str:
    return f"{logic}__" + "__".join(feats)


def fold_boundaries(n: int, n_folds: int) -> List[Tuple[int, int]]:
    bounds = []
    for i in range(n_folds):
        lo = int(round(i * n / n_folds))
        hi = int(round((i + 1) * n / n_folds))
        bounds.append((lo, hi))
    return bounds


def summarize_slice(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {
            "n_events": 0,
            "hit_rate": math.nan,
            "avg_mr_pts": math.nan,
            "avg_mae_pts": math.nan,
            "fav_target_rate": math.nan,
            "adv_penalty_rate": math.nan,
            "avg_end_ret_pts": math.nan,
        }

    hit_rate = sum(float(r["post_entry_mfe_pts"]) >= FAVORABLE_TARGET_PTS for r in rows) / n
    avg_mr = sum(float(r["post_entry_mfe_pts"]) for r in rows) / n
    avg_mae = sum(float(r["post_entry_mae_pts"]) for r in rows) / n
    fav_rate = sum(float(r["post_entry_mfe_pts"]) >= FAVORABLE_TARGET_PTS for r in rows) / n
    adv_rate = sum(float(r["post_entry_mae_pts"]) >= ADVERSE_PENALTY_LEVEL_PTS for r in rows) / n
    avg_end = sum(float(r["post_entry_end_ret_pts"]) for r in rows) / n

    return {
        "n_events": n,
        "hit_rate": hit_rate,
        "avg_mr_pts": avg_mr,
        "avg_mae_pts": avg_mae,
        "fav_target_rate": fav_rate,
        "adv_penalty_rate": adv_rate,
        "avg_end_ret_pts": avg_end,
    }


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
    print("Finalist Confirmation Combo Gates")
    print("--------------------------------------")
    print(f"HORIZON_S              = {HORIZON_S}")
    print(f"BAND_FILTER            = {BAND_FILTER}")
    print(f"SIDE_FILTER            = {SIDE_FILTER}")
    print(f"PARENT_RULE_NAME       = {PARENT_RULE_NAME}")
    print(f"ENTRY_ADVERSE_PTS      = {ENTRY_ADVERSE_PTS}")
    print(f"N_FOLDS                = {N_FOLDS}")
    print(f"N_CONFIRMATIONS        = {len(FINALIST_CONFIRMATIONS)}")
    print(f"Rows analyzed          = {len(rows):,}")
    print(f"Parent rule rows       = {len(parent_rows):,}")

    # --------------------------------------------------------
    # Build refined-entry event-level dataset
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

            if entry_price is None or event_ts is None:
                continue

            entry_price = float(entry_price)

            j0 = bisect_right(ts_list, event_ts) - 1
            if j0 < 0:
                continue

            entry_idx = None
            entry_ts = None
            entry_px = None

            # first pass: find refined entry
            for j in range(j0, len(ts_list)):
                dt_s = (ts_list[j] - event_ts).total_seconds()
                if dt_s > HORIZON_S:
                    break

                px = px_list[j]
                adverse_move = entry_price - px  # long logic

                if adverse_move >= ENTRY_ADVERSE_PTS:
                    entry_idx = j
                    entry_ts = ts_list[j]
                    entry_px = px
                    break

            if entry_idx is None:
                continue

            # second pass: find horizon end after trigger time
            horizon_end_idx = entry_idx
            for j in range(entry_idx, len(ts_list)):
                dt_s = (ts_list[j] - event_ts).total_seconds()
                if dt_s > HORIZON_S:
                    break
                horizon_end_idx = j

            future_px = px_list[entry_idx:horizon_end_idx + 1]
            if len(future_px) < 2:
                continue

            post_mfe = max(p - entry_px for p in future_px)
            post_mae = max(entry_px - p for p in future_px)
            end_ret = future_px[-1] - entry_px
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
        raise ValueError("No refined-entry event rows produced.")

    event_df = pl.from_dicts(event_level_records).sort("ts")
    event_df.write_csv(EVENT_LEVEL_OUT)

    event_rows = event_df.to_dicts()
    print(f"Refined-entry event rows = {len(event_rows):,}")

    # --------------------------------------------------------
    # Precompute confirmation booleans
    # --------------------------------------------------------
    conf_names: List[str] = []
    for feat, label_ranges in FINALIST_CONFIRMATIONS.items():
        for label, lo, hi in label_ranges:
            conf_name = f"{feat}__{label}"
            conf_names.append(conf_name)
            for row in event_rows:
                row[conf_name] = confirmation_pass(row, feat, lo, hi)

    # --------------------------------------------------------
    # Build combo definitions
    # --------------------------------------------------------
    combo_defs: List[Dict[str, Any]] = []

    if USE_SINGLE:
        for c in conf_names:
            combo_defs.append({
                "combo_name": make_combo_name("SINGLE", (c,)),
                "logic": "SINGLE",
                "confirmations": (c,),
            })

    if USE_ALL_2:
        for feats in combinations(conf_names, 2):
            combo_defs.append({
                "combo_name": make_combo_name("ALL2", feats),
                "logic": "ALL_2",
                "confirmations": feats,
            })

    if USE_ALL_3:
        for feats in combinations(conf_names, 3):
            combo_defs.append({
                "combo_name": make_combo_name("ALL3", feats),
                "logic": "ALL_3",
                "confirmations": feats,
            })

    if USE_VOTE_2_OF_3:
        for feats in combinations(conf_names, 3):
            combo_defs.append({
                "combo_name": make_combo_name("VOTE2OF3", feats),
                "logic": "AT_LEAST_2_OF_3",
                "confirmations": feats,
            })

    # --------------------------------------------------------
    # Walk-forward evaluation
    # anchored train = all rows before test block
    # --------------------------------------------------------
    fold_ranges = fold_boundaries(len(event_rows), N_FOLDS)
    fold_records: List[Dict[str, Any]] = []

    for fold_idx, (test_lo, test_hi) in enumerate(fold_ranges, start=1):
        test_rows_all = event_rows[test_lo:test_hi]
        train_rows_all = event_rows[:test_lo]

        if len(train_rows_all) == 0 or len(test_rows_all) == 0:
            continue

        for combo in combo_defs:
            logic = combo["logic"]
            feats = combo["confirmations"]

            def combo_pass(row: Dict[str, Any]) -> bool:
                vals = [bool(row[f]) for f in feats]
                if logic == "SINGLE":
                    return vals[0]
                if logic == "ALL_2":
                    return all(vals)
                if logic == "ALL_3":
                    return all(vals)
                if logic == "AT_LEAST_2_OF_3":
                    return sum(vals) >= 2
                raise ValueError(f"Unsupported logic: {logic}")

            train_rows = [r for r in train_rows_all if combo_pass(r)]
            test_rows = [r for r in test_rows_all if combo_pass(r)]

            train_stats = summarize_slice(train_rows)
            test_stats = summarize_slice(test_rows)

            fold_records.append({
                "combo_name": combo["combo_name"],
                "logic": logic,
                "confirmations": " | ".join(feats),
                "fold_idx": fold_idx,

                "train_n_events": train_stats["n_events"],
                "train_hit_rate": train_stats["hit_rate"],
                "train_avg_mr_pts": train_stats["avg_mr_pts"],
                "train_avg_mae_pts": train_stats["avg_mae_pts"],
                "train_adv_penalty_rate": train_stats["adv_penalty_rate"],
                "train_avg_end_ret_pts": train_stats["avg_end_ret_pts"],

                "test_n_events": test_stats["n_events"],
                "test_hit_rate": test_stats["hit_rate"],
                "test_avg_mr_pts": test_stats["avg_mr_pts"],
                "test_avg_mae_pts": test_stats["avg_mae_pts"],
                "test_adv_penalty_rate": test_stats["adv_penalty_rate"],
                "test_avg_end_ret_pts": test_stats["avg_end_ret_pts"],

                "hit_rate_degradation_test_minus_train":
                    (test_stats["hit_rate"] - train_stats["hit_rate"])
                    if (not math.isnan(test_stats["hit_rate"]) and not math.isnan(train_stats["hit_rate"]))
                    else math.nan,

                "avg_mr_pts_degradation_test_minus_train":
                    (test_stats["avg_mr_pts"] - train_stats["avg_mr_pts"])
                    if (not math.isnan(test_stats["avg_mr_pts"]) and not math.isnan(train_stats["avg_mr_pts"]))
                    else math.nan,
            })

    if not fold_records:
        raise ValueError("No fold records produced.")

    fold_df = pl.from_dicts(fold_records)
    fold_df.write_csv(FOLD_RESULTS_OUT)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    combo_names = fold_df["combo_name"].unique().to_list()

    for combo_name in combo_names:
        sub = fold_df.filter(pl.col("combo_name") == combo_name)

        test_n_events = sub["test_n_events"].to_list()
        valid_mask = [n >= MIN_TEST_EVENTS for n in test_n_events]

        valid = sub.filter(pl.col("test_n_events") >= MIN_TEST_EVENTS)
        n_valid = valid.height

        if n_valid == 0:
            coverage_status = "too_sparse"
        elif n_valid == 1:
            coverage_status = "one_fold_only"
        elif n_valid == 2:
            coverage_status = "two_folds_only"
        else:
            coverage_status = "ok"

        mean_test_events = float(valid["test_n_events"].mean()) if valid.height > 0 else math.nan
        mean_test_hit_rate = float(valid["test_hit_rate"].mean()) if valid.height > 0 else math.nan
        std_test_hit_rate = float(valid["test_hit_rate"].std()) if valid.height > 1 else math.nan
        mean_test_avg_mr_pts = float(valid["test_avg_mr_pts"].mean()) if valid.height > 0 else math.nan
        std_test_avg_mr_pts = float(valid["test_avg_mr_pts"].std()) if valid.height > 1 else math.nan
        mean_test_avg_mae_pts = float(valid["test_avg_mae_pts"].mean()) if valid.height > 0 else math.nan
        mean_test_adv_penalty_rate = float(valid["test_adv_penalty_rate"].mean()) if valid.height > 0 else math.nan
        mean_test_avg_end_ret_pts = float(valid["test_avg_end_ret_pts"].mean()) if valid.height > 0 else math.nan
        mean_hit_deg = float(valid["hit_rate_degradation_test_minus_train"].mean()) if valid.height > 0 else math.nan
        mean_mr_deg = float(valid["avg_mr_pts_degradation_test_minus_train"].mean()) if valid.height > 0 else math.nan

        if valid.height < 2:
            robust_score = math.nan
            stability_score = math.nan
        else:
            robust_score = (
                1.15 * (0.0 if math.isnan(mean_test_hit_rate) else mean_test_hit_rate)
                + 0.05 * (0.0 if math.isnan(mean_test_avg_mr_pts) else mean_test_avg_mr_pts)
                - 0.06 * (0.0 if math.isnan(mean_test_avg_mae_pts) else mean_test_avg_mae_pts)
                - 0.50 * (0.0 if math.isnan(mean_test_adv_penalty_rate) else mean_test_adv_penalty_rate)
                + 0.03 * (0.0 if math.isnan(mean_test_avg_end_ret_pts) else mean_test_avg_end_ret_pts)
                - 0.35 * max(0.0, -(0.0 if math.isnan(mean_hit_deg) else mean_hit_deg))
                + 0.02 * math.log1p(max(0.0, 0.0 if math.isnan(mean_test_events) else mean_test_events))                        
            )

            stability_score = (
                (0.0 if math.isnan(mean_test_hit_rate) else mean_test_hit_rate)
                - 0.50 * (0.0 if math.isnan(std_test_hit_rate) else std_test_hit_rate)
                + 0.03 * (0.0 if math.isnan(mean_test_avg_mr_pts) else mean_test_avg_mr_pts)
                - 0.02 * (0.0 if math.isnan(std_test_avg_mr_pts) else std_test_avg_mr_pts)
            )

        first = sub.row(0, named=True)
        summary_records.append({
            "combo_name": combo_name,
            "logic": first["logic"],
            "confirmations": first["confirmations"],
            "n_valid_test_folds": int(valid.height),
            "coverage_status": coverage_status,
            "mean_test_events": mean_test_events,
            "mean_test_hit_rate": mean_test_hit_rate,
            "std_test_hit_rate": std_test_hit_rate,
            "mean_test_avg_mr_pts": mean_test_avg_mr_pts,
            "std_test_avg_mr_pts": std_test_avg_mr_pts,
            "mean_test_avg_mae_pts": mean_test_avg_mae_pts,
            "mean_test_adv_penalty_rate": mean_test_adv_penalty_rate,
            "mean_test_avg_end_ret_pts": mean_test_avg_end_ret_pts,
            "mean_hit_rate_degradation_test_minus_train": mean_hit_deg,
            "mean_avg_mr_pts_degradation_test_minus_train": mean_mr_deg,
            "robust_score": robust_score,
            "stability_score": stability_score,
        })

    summary_df = pl.from_dicts(summary_records)

    summary_df = summary_df.with_columns(
        pl.when(pl.col("coverage_status") == "ok").then(0)
        .when(pl.col("coverage_status") == "two_folds_only").then(1)
        .when(pl.col("coverage_status") == "one_fold_only").then(2)
        .otherwise(3)
        .alias("coverage_rank")
    )

    summary_df = summary_df.sort(
        ["coverage_rank", "robust_score", "stability_score"],
        descending=[False, True, True]
    )
    summary_df.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {FOLD_RESULTS_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop combo summaries:")
    print(summary_df.head(30))


if __name__ == "__main__":
    main()
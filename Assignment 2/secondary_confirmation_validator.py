# secondary_confirmation_validator.py
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\secondary_confirmation_validator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "secondary_confirmation_validator_event_level.csv"
FOLD_RESULTS_OUT = OUTPUT_DIR / "secondary_confirmation_validator_fold_results.csv"
SUMMARY_OUT = OUTPUT_DIR / "secondary_confirmation_validator_summary.csv"


# ============================================================
# Core configuration
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "upper"

PARENT_RULE_NAME = "UNION_A2_C2"
ENTRY_ADVERSE_PTS = 1.0

N_FOLDS = 5

# evaluation targets
FAVORABLE_TARGET_PTS = 4.0
ADVERSE_PENALTY_LEVEL_PTS = 6.0

MIN_TEST_EVENTS_PER_FOLD = 5
MIN_VALID_TEST_FOLDS = 1

# ------------------------------------------------------------
# finalist secondary confirmations for SINGLE_C1
# edit freely
# format:
#   feature_name : [(label, lo, hi), ...]
# ------------------------------------------------------------
SECONDARY_CONFIRMATIONS: Dict[str, Dict[str, List[Tuple[str, float, float]]]] = {
    "UNION_A2_C2": {
        "depth_far_ask": [
            ("low_bin", 4.0, 120.0),
        ],
        "depth_mid_ask": [
            ("low_bin", 7.0, 107.0),
        ],
        "ask_depth_curvature": [
            ("high_bin", -5.5, 96.875),
        ],
        "wall_strength_opposite_5_2s_avg": [
            ("low_bin", 0.014858841010401188, 110.08548387096774),
        ],
        "depth_near_ask": [
            ("low_bin", 4.0, 87.0),
        ],
        "trade_sign_autocorr_60s": [
            ("low_bin", 0.5580088882045725, 0.8686430523580743),
        ],
        "wall_strength_opposite_5": [
            ("low_bin", 0.0, 164.0),
        ],
        "spread_volatility": [
            ("high_bin", 0.14742392202328267, 32.17508377755634),
        ],
        "trade_sign_autocorr_30s": [
            ("low_bin", 0.5769504127713069, 0.8642397220482174),
        ],
        "wall_strength_opposite_5_5s_avg": [
            ("low_bin", 0.0066181336863004635, 74.98831030818279),
        ],
        "wall_strength_opposite_3": [
            ("low_bin", 0.0, 97.0),
        ],
        "depth_mid_bid": [
            ("low_bin", 0.0, 106.0),
        ],
        "depth_far_bid": [
            ("low_bin", 8.0, 120.0),
        ],
        "wall_strength_opposite_3_2s_avg": [
            ("low_bin", 0.0, 57.93211009174312),
        ],
        "cancel_rate_bid": [
            ("high_bin", 11.8, 448.2),
        ],
        "trade_sign_autocorr_10s": [
            ("low_bin", 0.49213066985971254, 0.8582994003300181),
        ],
        "add_rate_bid": [
            ("high_bin", 12.2, 444.2),
        ],
        "wall_strength_opposite_3_5s_avg": [
            ("low_bin", 0.0, 40.75609756097561),
        ],
        "wall_persistence_2s": [
            ("high_bin", 0.24626006904487918, 0.5486725663716814),
        ],
        "add_rate_ask": [
            ("high_bin", 11.6, 506.2),
        ],
        "update_rate_ask": [
            ("high_bin", 114.0, 494.2),
        ],
        "CVD": [
            ("high_bin", 1117.0, 46314.0),
        ],
        "ask_curvature_l5": [
            ("high_bin", -0.6666666666666666, 271.0),
        ],
        "depth_curvature_pressure": [
            ("low_bin", -101.625, 0.0),
        ],
        "microprice_deviation": [
            ("low_bin", -6.192293884453411, 0.1414982774637506),
        ],
        "cancel_ratio": [
            ("high_bin", 0.09266666666666666, 0.32551358515573225),
        ],
        "cancel_rate_ask": [
            ("high_bin", 11.8, 427.0),
        ],
        "OFI_60s": [
            ("high_bin", -2.0, 1108.0),
        ],
        "impact_asymmetry_10s": [
            ("low_bin", -0.04439252336448598, -0.0005208333333333333),
        ],
        "sell_impact_efficiency_10s": [
            ("high_bin", 0.0005208333333333333, 0.04439252336448598),
        ],
        "impact_efficiency_30s": [
            ("high_bin", 0.009170653907496013, 1.1428571428571428),
        ],
        "OFI_30s": [
            ("high_bin", 0.0, 645.0),
        ],
        "impact_decay_3s": [
            ("high_bin", 0.8409090909090909, 84.0),
        ],
        "impact_decay_5s": [
            ("high_bin", 0.9898107714701601, 294.6),
        ],
        "bid_depth_curvature": [
            ("high_bin", -5.125, 11.5),
        ],
        "cvd_acceleration": [
            ("low_bin", -331.76666666666665, 1.4333333333333336),
        ],
        "aggressive_volume_ratio": [
            ("low_bin", 0.20019627085377822, 0.9483526268922529),
        ],
        "cvd_slope": [
            ("low_bin", -205.4, -3.1),
        ],
        "signed_volume": [
            ("low_bin", -2054.0, -31.0),
        ],
        "trade_imbalance": [
            ("low_bin", -0.6203429923122413, -0.03789126853377265),
        ],
        "impact_efficiency_10s": [
            ("high_bin", 0.009259259259259259, 0.8333333333333334),
        ],
        "bid_curvature_l5": [
            ("low_bin", -62.0, -0.6666666666666666),
        ],
        "flow_toxicity": [
            ("low_bin", -2.300635144671842, -0.04822043628013777),
        ],
        "flow_toxicity_10s": [
            ("low_bin", -2.300635144671842, -0.04822043628013777),
        ],
        "OFI": [
            ("high_bin", -8.0, 631.0),
        ],
        "liquidity_compression": [
            ("high_bin", 0.7115902964959568, 4.434579439252336),
        ],
        "trade_imbalance_5s": [
            ("low_bin", -0.7557894736842106, -0.006134969325153374),
        ],
        "queue_imbalance_std_5s": [
            ("high_bin", 0.21697247421416097, 0.6458511254366709),
        ],
        "convexity_imbalance": [
            ("low_bin", -11.555580507523018, -0.020305781175346427),
        ],
        "curvature_pressure_l5": [
            ("low_bin", -271.6666666666667, 0.33333333333333304),
        ],
        "absorption_ratio": [
            ("low_bin", 8.222222222222221, 116.66666666666667),
        ],
        "aggressive_sell_volume": [
            ("high_bin", 552.0, 5535.0),
        ],
        "toxicity_ratio": [
            ("low_bin", 0.001081081081081081, 0.22296947749612003),
        ],
        "queue_length_total_norm": [
            ("low_bin", 0.17484024596093561, 1.0064988377268906),
        ],
        "bid_queue_change_rate": [
            ("low_bin", -21.035225588770956, -0.2000687836478181),
        ],
        "update_rate_bid": [
            ("high_bin", 113.0, 554.6),
        ],
        "OFI_burst_ratio": [
            ("low_bin", 0.0, 10.729411764705882),
        ],
        "spread_change_rate": [
            ("high_bin", 0.0, 1.5569252033042975),
        ],
        "OFI_5s": [
            ("high_bin", -4.0, 679.0),
        ],
        "queue_imbalance_positive_ratio": [
            ("low_bin", 0.0, 0.4969818913480885),
        ],
        "volume_per_second": [
            ("low_bin", 12.4, 111.2),
        ],
        "impact_efficiency_3s": [
            ("high_bin", 0.009375, 0.375),
        ],
        "ask_convexity": [
            ("high_bin", 1.0914285714285714, 12.342465753424657),
        ],
        "convexity_ratio_ask": [
            ("high_bin", 1.0914285714285714, 12.342465753424657),
        ],
        "flow_impact": [
            ("high_bin", 0.008687258687258687, 0.75),
        ],
        "microprice": [
            ("low_bin", 4910.625, 5786.932291666667),
        ],
        "trade_imbalance_60s": [
            ("low_bin", -0.3731527093596059, -0.022815533980582524),
        ],
        "spread": [
            ("high_bin", 0.75, 9.5),
        ],
        "OBI_l10": [
            ("high_bin", 0.009900990099009901, 0.6643356643356644),
        ],
        "best_bid_queue_norm": [
            ("low_bin", 0.04070403160382349, 0.9453901606171746),
        ],
        "net_liquidity_shock": [
            ("high_bin", -0.19027515040007636, 9.103507952764158),
        ],
        "OBI_l3": [
            ("high_bin", -0.024390243902439025, 0.8107255520504731),
        ],
        "weighted_OBI_l10": [
            ("high_bin", -0.2402853883712501, 0.8621994967813962),
        ],
        "remove_rate_near_bid": [
            ("high_bin", 125.2, 4145.0),
        ],
        "bid_liquidity_shock": [
            ("high_bin", -0.10400572242443253, 8.404020661975492),
        ],
        "flow_toxicity_30s": [
            ("low_bin", -0.7123029875323453, -0.03379601689800845),
        ],
        "spread_mid_ratio": [
            ("high_bin", 0.00012819143253925863, 0.001683427103176361),
        ],
        "queue_imbalance_slope_5s": [
            ("high_bin", -0.004457299890139147, 0.26021279285629356),
        ],
        "depth_near_bid": [
            ("low_bin", 3.0, 84.0),
        ],
        "convexity_ratio_bid": [
            ("high_bin", 1.0825688073394495, 25.571428571428573),
        ],
        "bid_convexity": [
            ("high_bin", 1.0825688073394495, 25.571428571428573),
        ],
        "best_ask_queue_ratio": [
            ("high_bin", 0.5066666666666667, 0.9870740305522914),
        ],
        "liquidity_slope_ask": [
            ("low_bin", -45.01212121212121, -0.5272727272727272),
        ],
        "spread_expansion_ratio": [
            ("high_bin", 1.0491539081385979, 4.335724533715926),
        ],
        "queue_imbalance": [
            ("low_bin", -0.9741480611045829, -0.01818181818181818),
        ],
        "best_bid_queue_ratio": [
            ("low_bin", 0.012925969447708578, 0.4909090909090909),
        ],
        "microprice_minus_mid": [
            ("low_bin", -3.392857142856883, -0.006818181817834557),
        ],
        "microprice_pressure": [
            ("low_bin", -0.48707403055232135, -0.009090909090446075),
        ],
        "queue_length_ratio": [
            ("low_bin", 0.013095238095238096, 0.9642857142857143),
        ],
        "ask_queue_depletion_rate": [
            ("low_bin", 11.0, 62.2),
        ],
        "weighted_OBI_l3": [
            ("high_bin", -0.2948239493753042, 0.8234442836468886),
        ],
        "OBI_l1": [
            ("low_bin", -1.0, -0.02702702702702703),
        ],
        "liquidity_slope_bid": [
            ("low_bin", -50.58181818181818, -0.47878787878787876),
        ],
        "remove_rate_near_ask": [
            ("low_bin", 22.4, 116.0),
        ],
        "replenishment_rate_near_5s": [
            ("high_bin", 247.4, 4468.6),
        ],
        "OBI_l5": [
            ("high_bin", -0.006666666666666667, 0.8620689655172413),
        ],
        "ask_liquidity_shock": [
            ("high_bin", -0.0028267546300197546, 5.6005765787305695),
        ],
        "queue_imbalance_mean_5s": [
            ("high_bin", 0.012526286593471034, 0.40822404992557976),
        ],
        "weighted_OBI_l5": [
            ("high_bin", -0.27169579883471323, 0.914854423578775),
        ],
        "ask_queue_change_rate": [
            ("low_bin", -39.01187962268033, -0.20000616018973386),
        ],
        "avg_trade_size": [
            ("low_bin", 1.0229885057471264, 1.2617845117845117),
        ],
        "bid_queue_depletion_rate": [
            ("high_bin", 64.6, 3322.2),
        ],
        "trade_imbalance_30s": [
            ("low_bin", -0.45004500450045004, -0.03231939163498099),
        ],
        "aggressive_buy_volume": [
            ("low_bin", 70.0, 507.0),
        ],
        "trades_per_second": [
            ("low_bin", 11.1, 87.1),
        ],
        "flow_burst_ratio": [
            ("low_bin", 0.3771018326091064, 1.4481094127111827),
        ],
        "volume_burst_ratio": [
            ("low_bin", 0.3586286042174724, 1.474827245804541),
        ],
        "best_ask_queue_norm": [
            ("high_bin", 0.9811400683599983, 16.909106355664214),
        ],
        "buy_impact_efficiency_10s": [
            ("high_bin", 0.0, 0.04276315789473684),
        ],
    }
}

# if you want to test more later, just add them:
# "flow_toxicity": [("high_bin", -0.076699, 1.67872)]

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


def passes_secondary_confirmation(row: Dict[str, Any], feature: str, lo: float, hi: float) -> bool:
    v = row.get(feature)
    if v is None:
        return False
    try:
        x = float(v)
    except Exception:
        return False
    return lo <= x <= hi


def mean_safe(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else math.nan


def std_safe(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean_safe(xs)
    return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)

# ============================================================
# Main
# ============================================================
def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(FULL_TICK_ROOT)

    df = (
        pl.read_parquet(FEATURES_PATH)
        .filter(
            pl.col("band_k").is_in(BAND_FILTER) &
            (pl.col("side") == SIDE_FILTER)
        )
        .sort(["date_ymd", "ts"])
    )

    rows = df.to_dicts()
    parent_rows = [r for r in rows if parent_rule_pass(r)]
    if not parent_rows:
        raise ValueError("No rows passed parent rule.")

    day_to_path = build_day_to_path(FULL_TICK_ROOT)

    print("\n--------------------------------------")
    print("Secondary Confirmation Validator")
    print("--------------------------------------")
    print(f"HORIZON_S              = {HORIZON_S}")
    print(f"BAND_FILTER            = {BAND_FILTER}")
    print(f"SIDE_FILTER            = {SIDE_FILTER}")
    print(f"PARENT_RULE_NAME       = {PARENT_RULE_NAME}")
    print(f"ENTRY_ADVERSE_PTS      = {ENTRY_ADVERSE_PTS}")
    print(f"N_FOLDS                = {N_FOLDS}")
    print(f"N_SECONDARY_FEATURES   = {len(SECONDARY_CONFIRMATIONS[PARENT_RULE_NAME])}")
    print(f"Rows analyzed          = {len(rows):,}")
    print(f"Parent rule rows       = {len(parent_rows):,}")

    # --------------------------------------------------------
    # Build refined-entry event-level dataset once
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
            horizon_end_idx = None

            # first pass: find refined entry
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

            for feat in SECONDARY_CONFIRMATIONS[PARENT_RULE_NAME].keys():
                rec[feat] = r.get(feat)

            event_level_records.append(rec)

        processed_days += 1
        if processed_days % 25 == 0:
            print(f"...processed {processed_days} days")

    if not event_level_records:
        raise ValueError("No refined-entry event rows produced.")

    event_df = pl.from_dicts(event_level_records).sort("ts")
    event_df.write_csv(EVENT_LEVEL_OUT)

    print(f"Refined-entry event rows = {event_df.height:,}")

    # --------------------------------------------------------
    # Walk-forward validation
    # --------------------------------------------------------
    n = event_df.height
    fold_size = n // N_FOLDS

    fold_results: List[Dict[str, Any]] = []

    for feature, ranges in SECONDARY_CONFIRMATIONS[PARENT_RULE_NAME].items():
        feat_non_null = event_df.filter(pl.col(feature).is_not_null())
        if feat_non_null.height == 0:
            continue

        for label, lo, hi in ranges:
            for fold_idx in range(N_FOLDS):
                start = fold_idx * fold_size
                end = (fold_idx + 1) * fold_size if fold_idx < N_FOLDS - 1 else n

                test_df = event_df.slice(start, end - start)
                train_df = pl.concat(
                    [
                        event_df.slice(0, start),
                        event_df.slice(end, n - end),
                    ],
                    how="vertical",
                )

                train_rows = [
                    r for r in train_df.to_dicts()
                    if passes_secondary_confirmation(r, feature, lo, hi)
                ]
                test_rows = [
                    r for r in test_df.to_dicts()
                    if passes_secondary_confirmation(r, feature, lo, hi)
                ]

                train_n = len(train_rows)
                test_n = len(test_rows)

                is_valid_test_fold = int(test_n >= MIN_TEST_EVENTS_PER_FOLD)
                is_valid_train_fold = int(train_n >= MIN_TEST_EVENTS_PER_FOLD)

                train_hit = mean_safe(
                    [1.0 if r["post_entry_mfe_pts"] >= FAVORABLE_TARGET_PTS else 0.0 for r in train_rows]
                ) if train_rows else math.nan
                test_hit = mean_safe(
                    [1.0 if r["post_entry_mfe_pts"] >= FAVORABLE_TARGET_PTS else 0.0 for r in test_rows]
                ) if test_rows else math.nan

                train_avg_mr = mean_safe([float(r["post_entry_mfe_pts"]) for r in train_rows]) if train_rows else math.nan
                test_avg_mr = mean_safe([float(r["post_entry_mfe_pts"]) for r in test_rows]) if test_rows else math.nan

                train_avg_mae = mean_safe([float(r["post_entry_mae_pts"]) for r in train_rows]) if train_rows else math.nan
                test_avg_mae = mean_safe([float(r["post_entry_mae_pts"]) for r in test_rows]) if test_rows else math.nan

                hit_deg = test_hit - train_hit if (not math.isnan(train_hit) and not math.isnan(test_hit)) else math.nan
                mr_deg = test_avg_mr - train_avg_mr if (not math.isnan(train_avg_mr) and not math.isnan(test_avg_mr)) else math.nan

                fold_results.append({
                    "parent_rule": PARENT_RULE_NAME,
                    "secondary_feature": feature,
                    "secondary_label": label,
                    "feature_lo": lo,
                    "feature_hi": hi,
                    "fold_idx": fold_idx + 1,
                    "train_n_events": train_n,
                    "train_hit_rate": train_hit,
                    "train_avg_mr_pts": train_avg_mr,
                    "train_avg_mae_pts": train_avg_mae,
                    "test_n_events": test_n,
                    "test_hit_rate": test_hit,
                    "test_avg_mr_pts": test_avg_mr,
                    "test_avg_mae_pts": test_avg_mae,
                    "hit_rate_degradation_test_minus_train": hit_deg,
                    "avg_mr_pts_degradation_test_minus_train": mr_deg,
                    "is_valid_train_fold": is_valid_train_fold,
                    "is_valid_test_fold": is_valid_test_fold,
                })

    if not fold_results:
        raise ValueError("No fold results produced.")

    fold_df = pl.from_dicts(fold_results)
    fold_df.write_csv(FOLD_RESULTS_OUT)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary_records: List[Dict[str, Any]] = []

    group_keys = ["parent_rule", "secondary_feature", "secondary_label", "feature_lo", "feature_hi"]

    for key_vals, subdf in fold_df.group_by(group_keys, maintain_order=True):
        sub_all = subdf
        sub_valid = subdf.filter(pl.col("is_valid_test_fold") == 1)

        n_folds = sub_all.height
        n_valid_test_folds = sub_valid.height
        positive_test_folds = int((sub_all["test_n_events"] > 0).sum()) if sub_all.height > 0 else 0

        mean_test_n_events = float(sub_all["test_n_events"].mean()) if sub_all.height > 0 else math.nan
        min_test_n_events = int(sub_all["test_n_events"].min()) if sub_all.height > 0 else 0
        max_test_n_events = int(sub_all["test_n_events"].max()) if sub_all.height > 0 else 0

        if n_valid_test_folds == 0:
            summary_records.append({
                "parent_rule": key_vals[0],
                "secondary_feature": key_vals[1],
                "secondary_label": key_vals[2],
                "feature_lo": key_vals[3],
                "feature_hi": key_vals[4],
                "n_folds": n_folds,
                "positive_test_folds": positive_test_folds,
                "n_valid_test_folds": 0,
                "mean_test_n_events": mean_test_n_events,
                "min_test_n_events": min_test_n_events,
                "max_test_n_events": max_test_n_events,
                "mean_test_hit_rate": None,
                "std_test_hit_rate": None,
                "mean_test_avg_mr_pts": None,
                "std_test_avg_mr_pts": None,
                "mean_test_avg_mae_pts": None,
                "std_test_avg_mae_pts": None,
                "mean_hit_rate_degradation_test_minus_train": None,
                "mean_avg_mr_pts_degradation_test_minus_train": None,
                "coverage_status": "insufficient_coverage",
                "robust_score": None,
                "stability_score": None,
            })
            continue

        mean_test_hit = float(sub_valid["test_hit_rate"].mean())
        std_test_hit = float(sub_valid["test_hit_rate"].std()) if n_valid_test_folds > 1 else 0.0
        mean_test_mr = float(sub_valid["test_avg_mr_pts"].mean())
        std_test_mr = float(sub_valid["test_avg_mr_pts"].std()) if n_valid_test_folds > 1 else 0.0
        mean_test_mae = float(sub_valid["test_avg_mae_pts"].mean())
        std_test_mae = float(sub_valid["test_avg_mae_pts"].std()) if n_valid_test_folds > 1 else 0.0

        deg_sub = sub_valid.filter(
            pl.col("hit_rate_degradation_test_minus_train").is_not_null() &
            pl.col("avg_mr_pts_degradation_test_minus_train").is_not_null()
        )

        mean_deg = float(deg_sub["hit_rate_degradation_test_minus_train"].mean()) if deg_sub.height > 0 else math.nan
        mean_mr_deg = float(deg_sub["avg_mr_pts_degradation_test_minus_train"].mean()) if deg_sub.height > 0 else math.nan

        robust_score = (
            1.20 * mean_test_hit
            + 0.030 * mean_test_mr
            - 0.020 * mean_test_mae
            - 0.80 * abs(mean_deg if not math.isnan(mean_deg) else 0.0)
            - 0.20 * std_test_hit
        )

        stability_score = (
            0.60 * mean_test_hit
            - 0.25 * std_test_hit
            - 0.15 * abs(mean_deg if not math.isnan(mean_deg) else 0.0)
        )

        coverage_status = "ok" if n_valid_test_folds >= MIN_VALID_TEST_FOLDS else "insufficient_coverage"

        summary_records.append({
            "parent_rule": key_vals[0],
            "secondary_feature": key_vals[1],
            "secondary_label": key_vals[2],
            "feature_lo": key_vals[3],
            "feature_hi": key_vals[4],
            "n_folds": n_folds,
            "positive_test_folds": positive_test_folds,
            "n_valid_test_folds": n_valid_test_folds,
            "mean_test_n_events": mean_test_n_events,
            "min_test_n_events": min_test_n_events,
            "max_test_n_events": max_test_n_events,
            "mean_test_hit_rate": mean_test_hit,
            "std_test_hit_rate": std_test_hit,
            "mean_test_avg_mr_pts": mean_test_mr,
            "std_test_avg_mr_pts": std_test_mr,
            "mean_test_avg_mae_pts": mean_test_mae,
            "std_test_avg_mae_pts": std_test_mae,
            "mean_hit_rate_degradation_test_minus_train": mean_deg,
            "mean_avg_mr_pts_degradation_test_minus_train": mean_mr_deg,
            "coverage_status": coverage_status,
            "robust_score": robust_score if coverage_status == "ok" else None,
            "stability_score": stability_score if coverage_status == "ok" else None,
        })

    if not summary_records:
        raise ValueError("No summary records produced.")

    summary_df = pl.from_dicts(summary_records)

    summary_df = summary_df.with_columns(
        (pl.col("coverage_status") == "ok").cast(pl.Int8).alias("_coverage_ok")
    ).sort(
        ["_coverage_ok", "robust_score", "stability_score"],
        descending=[True, True, True],
    ).drop("_coverage_ok")

    summary_df.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {FOLD_RESULTS_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop secondary confirmation summaries:")
    print(summary_df.head(20))


if __name__ == "__main__":
    main()
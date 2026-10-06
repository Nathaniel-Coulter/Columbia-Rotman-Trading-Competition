# strategy_gate_simulator.py
from __future__ import annotations

from bisect import bisect_right
from itertools import product
from pathlib import Path
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LEVEL_OUT = OUTPUT_DIR / "strategy_gate_event_level.csv"
TRADE_LEVEL_OUT = OUTPUT_DIR / "strategy_gate_trade_level.csv"
FOLD_RESULTS_OUT = OUTPUT_DIR / "strategy_gate_fold_results.csv"
SUMMARY_OUT = OUTPUT_DIR / "strategy_gate_summary.csv"

# ============================================================
# Core configuration
# ============================================================
HORIZON_S = 600
BAND_FILTER = [1]
SIDE_FILTER = "upper"

PARENT_RULE_NAME = "UNION_A2_C2"
ENTRY_ADVERSE_PTS = 1.0

N_FOLDS = 4
MIN_TEST_TRADES = 15
ONE_TRADE_AT_A_TIME = True

TARGETS_PTS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
STOPS_PTS = [4.0, 5.0, 6.0, 7.0, 8.0]
TIME_EXITS_S = [120, 300, 600, 900]

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
# Final 6 confirmations
# ============================================================
FINALIST_CONFIRMATIONS: Dict[str, Tuple[str, float, float]] = {
    "best_ask_queue_norm__high_bin": (
        "best_ask_queue_norm", 0.9811400683599983, 16.909106355664214
    ),
    "aggressive_buy_volume__low_bin": (
        "aggressive_buy_volume", 70.0, 507.0
    ),
    "ask_liquidity_shock__high_bin": (
        "ask_liquidity_shock", -0.0028267546300197546, 5.6005765787305695
    ),
    "volume_per_second__low_bin": (
        "volume_per_second", 12.4, 111.2
    ),
    "wall_persistence_2s__high_bin": (
        "wall_persistence_2s", 0.24626006904487918, 0.5486725663716814
    ),
    "impact_efficiency_30s__high_bin": (
        "impact_efficiency_30s", 0.009170653907496013, 1.1428571428571428
    ),
}

# ============================================================
# Gate shortlist
# ============================================================
GATE_DEFS: List[Dict[str, Any]] = [
    {
        "gate_name": "G1_ALL2_bestask_aggbuy",
        "logic": "ALL",
        "confirmations": [
            "best_ask_queue_norm__high_bin",
            "aggressive_buy_volume__low_bin",
        ],
    },
    {
        "gate_name": "G2_ALL2_bestask_vps",
        "logic": "ALL",
        "confirmations": [
            "best_ask_queue_norm__high_bin",
            "volume_per_second__low_bin",
        ],
    },
    {
        "gate_name": "G3_VOTE2OF3_bestask_askshock_vps",
        "logic": "AT_LEAST_K",
        "k": 2,
        "confirmations": [
            "best_ask_queue_norm__high_bin",
            "ask_liquidity_shock__high_bin",
            "volume_per_second__low_bin",
        ],
    },
    {
        "gate_name": "G4_ALL2_aggbuy_impact30",
        "logic": "ALL",
        "confirmations": [
            "aggressive_buy_volume__low_bin",
            "impact_efficiency_30s__high_bin",
        ],
    },
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


def confirmation_pass(row: Dict[str, Any], feature_name: str, lo: float, hi: float) -> bool:
    v = row.get(feature_name)
    if v is None:
        return False
    try:
        x = float(v)
    except Exception:
        return False
    return lo <= x <= hi


def gate_pass(row: Dict[str, Any], gate_def: Dict[str, Any]) -> bool:
    vals = [bool(row[c]) for c in gate_def["confirmations"]]
    logic = gate_def["logic"]

    if logic == "ALL":
        return all(vals)
    if logic == "AT_LEAST_K":
        return sum(vals) >= int(gate_def["k"])

    raise ValueError(f"Unsupported gate logic: {logic}")


def fold_boundaries(n: int, n_folds: int) -> List[Tuple[int, int]]:
    out = []
    for i in range(n_folds):
        lo = int(round(i * n / n_folds))
        hi = int(round((i + 1) * n / n_folds))
        out.append((lo, hi))
    return out


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": math.nan,
            "avg_pnl_pts": math.nan,
            "avg_mfe_pts": math.nan,
            "avg_mae_pts": math.nan,
            "avg_end_ret_pts": math.nan,
            "target_hit_rate": math.nan,
            "stop_hit_rate": math.nan,
            "time_exit_rate": math.nan,
            "avg_hold_s": math.nan,
        }

    wins = [t["pnl_pts"] > 0 for t in trades]
    target_hits = [t["exit_reason"] == "target" for t in trades]
    stop_hits = [t["exit_reason"] == "stop" for t in trades]
    time_exits = [t["exit_reason"] == "time" for t in trades]

    return {
        "n_trades": n,
        "win_rate": sum(wins) / n,
        "avg_pnl_pts": sum(t["pnl_pts"] for t in trades) / n,
        "avg_mfe_pts": sum(t["mfe_pts"] for t in trades) / n,
        "avg_mae_pts": sum(t["mae_pts"] for t in trades) / n,
        "avg_end_ret_pts": sum(t["end_ret_pts"] for t in trades) / n,
        "target_hit_rate": sum(target_hits) / n,
        "stop_hit_rate": sum(stop_hits) / n,
        "time_exit_rate": sum(time_exits) / n,
        "avg_hold_s": sum(t["hold_s"] for t in trades) / n,
    }


def simulate_subset(
    rows_subset: List[Dict[str, Any]],
    target_pts: float,
    stop_pts: float,
    time_exit_s: int,
) -> List[Dict[str, Any]]:
    rows_subset = sorted(rows_subset, key=lambda r: r["entry_ts"])
    out: List[Dict[str, Any]] = []

    next_available_ts = None

    for r in rows_subset:
        if ONE_TRADE_AT_A_TIME and next_available_ts is not None and r["entry_ts"] <= next_available_ts:
            continue

        ts_list = r["day_ts_list"]
        px_list = r["day_px_list"]
        entry_idx = int(r["entry_idx"])
        entry_px = float(r["entry_price_refined"])
        entry_ts = r["entry_ts"]
        direction = float(r["direction"])

        exit_idx = entry_idx
        exit_reason = "time"

        last_idx_in_window = entry_idx
        for j in range(entry_idx, len(ts_list)):
            dt_s = (ts_list[j] - entry_ts).total_seconds()
            if dt_s > time_exit_s:
                break
            last_idx_in_window = j

        for j in range(entry_idx + 1, last_idx_in_window + 1):
            px = px_list[j]
            signed_move = (px - entry_px) * direction

            if signed_move >= target_pts:
                exit_idx = j
                exit_reason = "target"
                break
            if signed_move <= -stop_pts:
                exit_idx = j
                exit_reason = "stop"
                break
        else:
            exit_idx = last_idx_in_window
            exit_reason = "time"

        path_px = px_list[entry_idx:exit_idx + 1]
        exit_px = px_list[exit_idx]
        exit_ts = ts_list[exit_idx]

        signed_path = [(p - entry_px) * direction for p in path_px]
        pnl_pts = (exit_px - entry_px) * direction
        mfe_pts = max(signed_path)
        mae_pts = max(0.0, -min(signed_path))
        end_ret_pts = pnl_pts
        hold_s = (exit_ts - entry_ts).total_seconds()

        rec = {
            "event_id": r["event_id"],
            "date_ymd": r["date_ymd"],
            "trigger_ts": r["ts"],
            "entry_ts": entry_ts,
            "entry_price": entry_px,
            "exit_ts": exit_ts,
            "exit_price": exit_px,
            "target_pts": target_pts,
            "stop_pts": stop_pts,
            "time_exit_s": time_exit_s,
            "exit_reason": exit_reason,
            "pnl_pts": pnl_pts,
            "mfe_pts": mfe_pts,
            "mae_pts": mae_pts,
            "end_ret_pts": end_ret_pts,
            "hold_s": hold_s,
        }
        out.append(rec)

        if ONE_TRADE_AT_A_TIME:
            next_available_ts = exit_ts

    return out


def safe_mean(df: pl.DataFrame, col: str) -> float:
    if df.height == 0:
        return math.nan
    x = df[col].mean()
    return float(x) if x is not None else math.nan


def safe_std(df: pl.DataFrame, col: str) -> float:
    if df.height <= 1:
        return 0.0
    x = df[col].std()
    return float(x) if x is not None else 0.0


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
    print("Strategy Gate Simulator")
    print("--------------------------------------")
    print(f"HORIZON_S              = {HORIZON_S}")
    print(f"BAND_FILTER            = {BAND_FILTER}")
    print(f"SIDE_FILTER            = {SIDE_FILTER}")
    print(f"PARENT_RULE_NAME       = {PARENT_RULE_NAME}")
    print(f"ENTRY_ADVERSE_PTS      = {ENTRY_ADVERSE_PTS}")
    print(f"N_FOLDS                = {N_FOLDS}")
    print(f"N_GATES                = {len(GATE_DEFS)}")
    print(f"TARGETS_PTS            = {TARGETS_PTS}")
    print(f"STOPS_PTS              = {STOPS_PTS}")
    print(f"TIME_EXITS_S           = {TIME_EXITS_S}")
    print(f"ONE_TRADE_AT_A_TIME    = {ONE_TRADE_AT_A_TIME}")
    print(f"Rows analyzed          = {len(rows):,}")
    print(f"Parent rule rows       = {len(parent_rows):,}")

    # --------------------------------------------------------
    # Build refined-entry event-level dataset from raw ticks
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
            trigger_px = r.get("entry_price")
            event_ts = r.get("ts")
            side = r.get("side")

            if trigger_px is None or event_ts is None or side is None:
                continue

            trigger_px = float(trigger_px)
            direction = 1.0 if side == "lower" else -1.0

            j0 = bisect_right(ts_list, event_ts) - 1
            if j0 < 0:
                continue

            entry_idx = None
            entry_ts = None
            entry_px = None

            # first: find refined entry after adverse move
            for j in range(j0 + 1, len(ts_list)):
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

            # second: full remaining path to original horizon end
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
                "entry_ts": entry_ts,
                "entry_idx": int(entry_idx),
                "trigger_price": trigger_px,
                "entry_price_refined": float(entry_px),
                "entry_delay_s": entry_delay_s,
                "post_entry_mfe_pts": post_mfe,
                "post_entry_mae_pts": post_mae,
                "post_entry_end_ret_pts": end_ret,
                "day_ts_list": ts_list,
                "day_px_list": px_list,
                "direction": direction,
            }

            for conf_name, (feat, lo, hi) in FINALIST_CONFIRMATIONS.items():
                rec[conf_name] = confirmation_pass(r, feat, lo, hi)

            event_level_records.append(rec)

        processed_days += 1
        if processed_days % 25 == 0:
            print(f"...processed {processed_days} days")

    if not event_level_records:
        raise ValueError("No refined-entry event rows produced.")

    # save a file-friendly copy without list columns
    event_df_save = pl.from_dicts([
        {k: v for k, v in rec.items() if k not in {"day_ts_list", "day_px_list"}}
        for rec in event_level_records
    ]).sort("ts")
    event_df_save.write_csv(EVENT_LEVEL_OUT)

    event_rows = sorted(event_level_records, key=lambda r: r["ts"])
    print(f"Refined-entry event rows = {len(event_rows):,}")

    # --------------------------------------------------------
    # Walk-forward simulation
    # anchored train, forward test
    # --------------------------------------------------------
    fold_ranges = fold_boundaries(len(event_rows), N_FOLDS)
    fold_results: List[Dict[str, Any]] = []
    trade_results: List[Dict[str, Any]] = []

    param_grid = list(product(TARGETS_PTS, STOPS_PTS, TIME_EXITS_S))

    for fold_idx, (test_lo, test_hi) in enumerate(fold_ranges, start=1):
        train_all = event_rows[:test_lo]
        test_all = event_rows[test_lo:test_hi]

        if len(train_all) == 0 or len(test_all) == 0:
            continue

        print(f"\nFold {fold_idx}/{N_FOLDS} | test rows {test_lo}:{test_hi}")

        for gate_def in GATE_DEFS:
            gate_name = gate_def["gate_name"]
            gate_conf_text = " | ".join(gate_def["confirmations"])

            train_gate_rows = [r for r in train_all if gate_pass(r, gate_def)]
            test_gate_rows = [r for r in test_all if gate_pass(r, gate_def)]

            for target_pts, stop_pts, time_exit_s in param_grid:
                train_trades = simulate_subset(train_gate_rows, target_pts, stop_pts, time_exit_s)
                test_trades = simulate_subset(test_gate_rows, target_pts, stop_pts, time_exit_s)

                train_stats = summarize_trades(train_trades)
                test_stats = summarize_trades(test_trades)

                for t in test_trades:
                    trade_results.append({
                        "fold_idx": fold_idx,
                        "gate_name": gate_name,
                        "logic": gate_def["logic"],
                        "confirmations": gate_conf_text,
                        "target_pts": target_pts,
                        "stop_pts": stop_pts,
                        "time_exit_s": time_exit_s,
                        **t,
                    })

                fold_results.append({
                    "fold_idx": fold_idx,
                    "gate_name": gate_name,
                    "logic": gate_def["logic"],
                    "confirmations": gate_conf_text,
                    "target_pts": target_pts,
                    "stop_pts": stop_pts,
                    "time_exit_s": time_exit_s,

                    "train_n_trades": train_stats["n_trades"],
                    "train_win_rate": train_stats["win_rate"],
                    "train_avg_pnl_pts": train_stats["avg_pnl_pts"],
                    "train_avg_mfe_pts": train_stats["avg_mfe_pts"],
                    "train_avg_mae_pts": train_stats["avg_mae_pts"],
                    "train_avg_end_ret_pts": train_stats["avg_end_ret_pts"],
                    "train_target_hit_rate": train_stats["target_hit_rate"],
                    "train_stop_hit_rate": train_stats["stop_hit_rate"],
                    "train_time_exit_rate": train_stats["time_exit_rate"],
                    "train_avg_hold_s": train_stats["avg_hold_s"],

                    "test_n_trades": test_stats["n_trades"],
                    "test_win_rate": test_stats["win_rate"],
                    "test_avg_pnl_pts": test_stats["avg_pnl_pts"],
                    "test_avg_mfe_pts": test_stats["avg_mfe_pts"],
                    "test_avg_mae_pts": test_stats["avg_mae_pts"],
                    "test_avg_end_ret_pts": test_stats["avg_end_ret_pts"],
                    "test_target_hit_rate": test_stats["target_hit_rate"],
                    "test_stop_hit_rate": test_stats["stop_hit_rate"],
                    "test_time_exit_rate": test_stats["time_exit_rate"],
                    "test_avg_hold_s": test_stats["avg_hold_s"],

                    "win_rate_degradation_test_minus_train":
                        (test_stats["win_rate"] - train_stats["win_rate"])
                        if (not math.isnan(test_stats["win_rate"]) and not math.isnan(train_stats["win_rate"]))
                        else math.nan,

                    "avg_pnl_degradation_test_minus_train":
                        (test_stats["avg_pnl_pts"] - train_stats["avg_pnl_pts"])
                        if (not math.isnan(test_stats["avg_pnl_pts"]) and not math.isnan(train_stats["avg_pnl_pts"]))
                        else math.nan,
                })

    if not fold_results:
        raise ValueError("No fold results produced.")

    fold_df = pl.from_dicts(fold_results)
    trade_df = pl.from_dicts(trade_results) if trade_results else pl.DataFrame()

    fold_df.write_csv(FOLD_RESULTS_OUT)
    if trade_df.height > 0:
        trade_df.write_csv(TRADE_LEVEL_OUT)

    # --------------------------------------------------------
    # Summary by gate x params
    # --------------------------------------------------------
    group_cols = ["gate_name", "logic", "confirmations", "target_pts", "stop_pts", "time_exit_s"]

    summary = (
        fold_df
        .group_by(group_cols)
        .agg([
            pl.len().alias("n_total_folds"),
            (pl.col("test_n_trades") >= MIN_TEST_TRADES).sum().alias("n_valid_test_folds"),

            pl.col("test_n_trades").mean().alias("mean_test_trades"),
            pl.col("test_win_rate").mean().alias("mean_test_win_rate"),
            pl.col("test_win_rate").std().alias("std_test_win_rate"),

            pl.col("test_avg_pnl_pts").mean().alias("mean_test_avg_pnl_pts"),
            pl.col("test_avg_pnl_pts").std().alias("std_test_avg_pnl_pts"),

            pl.col("test_avg_mfe_pts").mean().alias("mean_test_avg_mfe_pts"),
            pl.col("test_avg_mae_pts").mean().alias("mean_test_avg_mae_pts"),
            pl.col("test_avg_end_ret_pts").mean().alias("mean_test_avg_end_ret_pts"),
            pl.col("test_target_hit_rate").mean().alias("mean_test_target_hit_rate"),
            pl.col("test_stop_hit_rate").mean().alias("mean_test_stop_hit_rate"),
            pl.col("test_time_exit_rate").mean().alias("mean_test_time_exit_rate"),
            pl.col("test_avg_hold_s").mean().alias("mean_test_avg_hold_s"),

            pl.col("win_rate_degradation_test_minus_train").mean().alias("mean_win_rate_degradation_test_minus_train"),
            pl.col("avg_pnl_degradation_test_minus_train").mean().alias("mean_avg_pnl_degradation_test_minus_train"),
        ])
        .with_columns([
            pl.when(pl.col("n_valid_test_folds") == (N_FOLDS - 1)).then(pl.lit(0))
            .when(pl.col("n_valid_test_folds") == max(0, N_FOLDS - 2)).then(pl.lit(1))
            .when(pl.col("n_valid_test_folds") == max(0, N_FOLDS - 3)).then(pl.lit(2))
            .otherwise(pl.lit(3))
            .alias("coverage_rank"),

            pl.when(pl.col("n_valid_test_folds") >= max(2, N_FOLDS - 1))
            .then(pl.lit("ok"))
            .when(pl.col("n_valid_test_folds") >= 1)
            .then(pl.lit("thin"))
            .otherwise(pl.lit("too_sparse"))
            .alias("coverage_status"),
        ])
    )

    # robust / stability scores
    summary = summary.with_columns([
        (
            0.95 * pl.col("mean_test_avg_pnl_pts")
            + 0.55 * pl.col("mean_test_win_rate")
            + 0.05 * pl.col("mean_test_avg_end_ret_pts")
            - 0.35 * pl.col("mean_test_stop_hit_rate")
            - 0.03 * pl.col("mean_test_avg_mae_pts")
            - 0.25 * pl.when(pl.col("mean_avg_pnl_degradation_test_minus_train") < 0)
                      .then(-pl.col("mean_avg_pnl_degradation_test_minus_train"))
                      .otherwise(0.0)
            - 0.20 * pl.when(pl.col("mean_win_rate_degradation_test_minus_train") < 0)
                      .then(-pl.col("mean_win_rate_degradation_test_minus_train"))
                      .otherwise(0.0)
            + 0.03 * pl.col("n_valid_test_folds")
        ).alias("robust_score"),

        (
            pl.col("mean_test_avg_pnl_pts")
            + 0.40 * pl.col("mean_test_win_rate")
            - 0.45 * pl.col("std_test_win_rate").fill_null(0.0)
            - 0.08 * pl.col("std_test_avg_pnl_pts").fill_null(0.0)
        ).alias("stability_score"),
    ])

    summary = summary.sort(
        ["coverage_rank", "robust_score", "stability_score"],
        descending=[False, True, True]
    )
    summary.write_csv(SUMMARY_OUT)

    print("\nDone.")
    print(f"Wrote: {EVENT_LEVEL_OUT}")
    print(f"Wrote: {TRADE_LEVEL_OUT}")
    print(f"Wrote: {FOLD_RESULTS_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")

    print("\nTop strategy summaries:")
    print(summary.head(30))


if __name__ == "__main__":
    main()
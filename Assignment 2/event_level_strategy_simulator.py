# event_level_strategy_simulator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import polars as pl


FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\event_level_strategy_simulator"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRADES_CSV = OUTPUT_DIR / "simulated_trades.csv"
OUT_COMBO_SUMMARY_CSV = OUTPUT_DIR / "combo_summary.csv"
OUT_FOLD_SUMMARY_CSV = OUTPUT_DIR / "combo_fold_summary.csv"


# --------------------------------------------------
# research subset
# --------------------------------------------------
BAND_FILTER = [1]
SIDE_FILTER = "lower"

# evaluate on this event horizon
HORIZON_S = 600

# walk-forward outer folds; we simulate ONLY on each test segment
N_FOLDS = 5
TRAIN_FRAC_WITHIN_FOLD = 0.60
VAL_FRAC_WITHIN_FOLD = 0.20
TEST_FRAC_WITHIN_FOLD = 0.20

# target / stop ladder for reporting
TARGETS_PTS = [1.0, 2.0, 4.0]
STOP_PTS = 2.0

# if True, only allow one active trade at a time inside each test fold
ONE_TRADE_AT_A_TIME = True


# --------------------------------------------------
# shortlist combos
# --------------------------------------------------
COMBOS: List[Dict[str, Any]] = [
    {
        "combo_name": "VOTE_B1_C1_D1",
        "logic": "AT_LEAST_2",
        "rules": [
            "B1_depth_mid_ask__queue_imbalance_std_5s",
            "C1_realized_vol_1m__depth_mid_ask",
            "D1_cancel_rate_ask__bid_depth_curvature",
        ],
    },
    {
        "combo_name": "ALL2_A2_C1",
        "logic": "ALL",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "C1_realized_vol_1m__depth_mid_ask",
        ],
    },
    {
        "combo_name": "SINGLE_A2",
        "logic": "SINGLE",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
        ],
    },
    {
        "combo_name": "VOTE4_A2_B1_C1_E1",
        "logic": "AT_LEAST_2",
        "rules": [
            "A2_depth_near_ask__cancel_rate_ask",
            "B1_depth_mid_ask__queue_imbalance_std_5s",
            "C1_realized_vol_1m__depth_mid_ask",
            "E1_cancel_rate_ask__cancel_ratio",
        ],
    },
    {
        "combo_name": "VOTE_B1_C1_E1",
        "logic": "AT_LEAST_2",
        "rules": [
            "B1_depth_mid_ask__queue_imbalance_std_5s",
            "C1_realized_vol_1m__depth_mid_ask",
            "E1_cancel_rate_ask__cancel_ratio",
        ],
    },
]


# --------------------------------------------------
# atomic rules frozen from prior research
# --------------------------------------------------
RULE_DEFS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "A2_depth_near_ask__cancel_rate_ask": {
        "depth_near_ask": (3.0, 127.0),
        "cancel_rate_ask": (6.8, 222.2),
    },
    "B1_depth_mid_ask__queue_imbalance_std_5s": {
        "depth_mid_ask": (5.0, 150.0),
        "queue_imbalance_std_5s": (0.194964, 0.528415),
    },
    "C1_realized_vol_1m__depth_mid_ask": {
        "realized_vol_1m": (0.000694397, 0.00493388),
        "depth_mid_ask": (5.0, 149.0),
    },
    "D1_cancel_rate_ask__bid_depth_curvature": {
        "cancel_rate_ask": (6.8, 222.2),
        "bid_depth_curvature": (-6.875, 25.375),
    },
    "E1_cancel_rate_ask__cancel_ratio": {
        "cancel_rate_ask": (6.8, 222.2),
        "cancel_ratio": (0.0805134, 0.319363),
    },
}


def ensure_columns(df: pl.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_fold_ranges(n: int, n_folds: int) -> List[Tuple[int, int]]:
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_folds)]


def split_inner_block(start: int, end: int) -> Dict[str, slice]:
    m = end - start
    train_end = start + int(m * TRAIN_FRAC_WITHIN_FOLD)
    val_end = start + int(m * (TRAIN_FRAC_WITHIN_FOLD + VAL_FRAC_WITHIN_FOLD))
    return {
        "train": slice(start, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, end),
    }


def build_rule_mask(pdf: pd.DataFrame, conds: Dict[str, Tuple[float, float]]) -> pd.Series:
    mask = pd.Series(True, index=pdf.index)
    for col, (lo, hi) in conds.items():
        mask = mask & pdf[col].notna() & (pdf[col] >= lo) & (pdf[col] <= hi)
    return mask


def combine_masks(mask_dict: Dict[str, pd.Series], combo_logic: str, combo_rules: List[str]) -> pd.Series:
    if combo_logic == "SINGLE":
        return mask_dict[combo_rules[0]].copy()

    mat = pd.concat([mask_dict[r].astype(int) for r in combo_rules], axis=1)
    vote_count = mat.sum(axis=1)

    if combo_logic == "ANY":
        return vote_count >= 1
    if combo_logic == "ALL":
        return vote_count == len(combo_rules)
    if combo_logic == "AT_LEAST_2":
        return vote_count >= 2

    raise ValueError(f"Unknown combo logic: {combo_logic}")


def safe_mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if pd.notna(x)]
    return float(np.mean(vals)) if vals else np.nan


def safe_std(xs: List[float]) -> float:
    vals = [float(x) for x in xs if pd.notna(x)]
    return float(np.std(vals, ddof=0)) if vals else np.nan


def simulate_on_test_block(
    df_block: pd.DataFrame,
    signal_mask: pd.Series,
    horizon_s: int,
    stop_pts: float,
    targets_pts: List[float],
    one_trade_at_a_time: bool = True,
) -> pd.DataFrame:
    """
    No look-ahead in signal generation:
    - signal_mask is precomputed from event-time features only
    - outcomes use already-stored forward metrics for that event
    - one-trade-at-a-time lockout uses event ts + horizon
    """
    rows = []
    active_until = pd.Timestamp.min

    mr_col = f"mr_pts_t{horizon_s}"
    mae_col = f"mae_pts_t{horizon_s}"

    for idx, row in df_block.loc[signal_mask].sort_values("ts").iterrows():
        ts = row["ts"]

        if one_trade_at_a_time and ts < active_until:
            continue

        mr_val = row[mr_col]
        mae_val = row[mae_col]

        trade = {
            "event_id": row["event_id"],
            "date_ymd": row["date_ymd"],
            "ts": ts,
            "mr_pts": mr_val,
            "mae_pts": mae_val,
            "hit_stop_proxy": float(mae_val >= stop_pts) if pd.notna(mae_val) else np.nan,
        }

        for tgt in targets_pts:
            trade[f"hit_{tgt:g}pt"] = float(mr_val >= tgt) if pd.notna(mr_val) else np.nan

            # simple target/stop proxy
            # +tgt if target reached, else -min(mae, stop)
            if pd.isna(mr_val) or pd.isna(mae_val):
                pnl_proxy = np.nan
            elif mr_val >= tgt:
                pnl_proxy = tgt
            else:
                pnl_proxy = -min(float(mae_val), stop_pts)

            trade[f"pnl_proxy_tgt_{tgt:g}_stop_{stop_pts:g}"] = pnl_proxy

        rows.append(trade)

        if one_trade_at_a_time:
            active_until = ts + pd.Timedelta(seconds=horizon_s)

    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame, targets_pts: List[float], stop_pts: float) -> Dict[str, Any]:
    if len(trades) == 0:
        out = {
            "n_trades": 0,
            "avg_mr_pts": np.nan,
            "avg_mae_pts": np.nan,
            "stop_hit_rate_proxy": np.nan,
        }
        for tgt in targets_pts:
            out[f"hit_rate_{tgt:g}pt"] = np.nan
            out[f"avg_pnl_proxy_tgt_{tgt:g}_stop_{stop_pts:g}"] = np.nan
        return out

    out = {
        "n_trades": int(len(trades)),
        "avg_mr_pts": float(trades["mr_pts"].mean()),
        "avg_mae_pts": float(trades["mae_pts"].mean()),
        "stop_hit_rate_proxy": float(trades["hit_stop_proxy"].mean()),
    }

    for tgt in targets_pts:
        out[f"hit_rate_{tgt:g}pt"] = float(trades[f"hit_{tgt:g}pt"].mean())
        out[f"avg_pnl_proxy_tgt_{tgt:g}_stop_{stop_pts:g}"] = float(
            trades[f"pnl_proxy_tgt_{tgt:g}_stop_{stop_pts:g}"].mean()
        )

    return out


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    mr_col = f"mr_pts_t{HORIZON_S}"
    mae_col = f"mae_pts_t{HORIZON_S}"

    needed_rule_cols = sorted({c for conds in RULE_DEFS.values() for c in conds.keys()})
    needed_cols = [
        "event_id", "date_ymd", "ts", "band_k", "side",
        mr_col, mae_col,
    ] + needed_rule_cols

    df = pl.read_parquet(FEATURES_PATH)
    ensure_columns(df, needed_cols)

    df = df.filter(
        pl.col("band_k").is_in(BAND_FILTER) &
        (pl.col("side") == SIDE_FILTER) &
        pl.col(mr_col).is_not_null() &
        pl.col(mae_col).is_not_null()
    ).sort("ts")

    pdf = df.to_pandas().reset_index(drop=True)

    if len(pdf) < 500:
        raise ValueError(f"Too few rows after filtering: {len(pdf)}")

    print("\n--------------------------------------")
    print("Event-Level Strategy Simulator")
    print("--------------------------------------")
    print(f"Rows analyzed: {len(pdf):,}")
    print(f"HORIZON_S            = {HORIZON_S}")
    print(f"BAND_FILTER          = {BAND_FILTER}")
    print(f"SIDE_FILTER          = {SIDE_FILTER}")
    print(f"TARGETS_PTS          = {TARGETS_PTS}")
    print(f"STOP_PTS             = {STOP_PTS}")
    print(f"N_FOLDS              = {N_FOLDS}")
    print(f"ONE_TRADE_AT_A_TIME  = {ONE_TRADE_AT_A_TIME}")

    atomic_masks = {
        rule_name: build_rule_mask(pdf, conds).astype(bool)
        for rule_name, conds in RULE_DEFS.items()
    }
    combo_masks = {
        combo["combo_name"]: combine_masks(atomic_masks, combo["logic"], combo["rules"]).astype(bool)
        for combo in COMBOS
    }

    fold_ranges = build_fold_ranges(len(pdf), N_FOLDS)

    trade_rows = []
    fold_summary_rows = []

    for fold_id, (outer_start, outer_end) in enumerate(fold_ranges, start=1):
        inner = split_inner_block(outer_start, outer_end)
        test_slice = inner["test"]
        df_test = pdf.iloc[test_slice].copy()

        print(f"\nFold {fold_id}/{N_FOLDS} | test rows {test_slice.start}:{test_slice.stop}")

        for combo in COMBOS:
            combo_name = combo["combo_name"]
            test_mask = combo_masks[combo_name].iloc[test_slice]

            trades = simulate_on_test_block(
                df_block=df_test,
                signal_mask=test_mask,
                horizon_s=HORIZON_S,
                stop_pts=STOP_PTS,
                targets_pts=TARGETS_PTS,
                one_trade_at_a_time=ONE_TRADE_AT_A_TIME,
            )

            if len(trades):
                trades.insert(0, "combo_name", combo_name)
                trades.insert(0, "fold_id", fold_id)
                trade_rows.append(trades)

            stats = summarize_trades(trades, TARGETS_PTS, STOP_PTS)
            stats.update({
                "fold_id": fold_id,
                "combo_name": combo_name,
                "logic": combo["logic"],
                "rule_names": " | ".join(combo["rules"]),
            })
            fold_summary_rows.append(stats)

    df_trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    df_fold_summary = pd.DataFrame(fold_summary_rows)

    summary_rows = []
    for combo_name, g in df_fold_summary.groupby("combo_name", sort=False):
        row = {
            "combo_name": combo_name,
            "logic": g["logic"].iloc[0],
            "rule_names": g["rule_names"].iloc[0],
            "positive_test_folds": int((g["n_trades"] > 0).sum()),
            "mean_test_trades": float(g["n_trades"].mean()),
            "min_test_trades": int(g["n_trades"].min()),
            "max_test_trades": int(g["n_trades"].max()),
            "mean_avg_mr_pts": float(g["avg_mr_pts"].mean()),
            "std_avg_mr_pts": float(g["avg_mr_pts"].std(ddof=0)),
            "mean_avg_mae_pts": float(g["avg_mae_pts"].mean()),
            "std_avg_mae_pts": float(g["avg_mae_pts"].std(ddof=0)),
            "mean_stop_hit_rate_proxy": float(g["stop_hit_rate_proxy"].mean()),
            "std_stop_hit_rate_proxy": float(g["stop_hit_rate_proxy"].std(ddof=0)),
        }

        for tgt in TARGETS_PTS:
            row[f"mean_hit_rate_{tgt:g}pt"] = float(g[f"hit_rate_{tgt:g}pt"].mean())
            row[f"std_hit_rate_{tgt:g}pt"] = float(g[f"hit_rate_{tgt:g}pt"].std(ddof=0))
            row[f"mean_avg_pnl_proxy_tgt_{tgt:g}_stop_{STOP_PTS:g}"] = float(
                g[f"avg_pnl_proxy_tgt_{tgt:g}_stop_{STOP_PTS:g}"].mean()
            )

        # simple ranking score centered on the 2pt and 4pt tradeability
        row["strategy_score"] = (
            0.30 * row["mean_hit_rate_1pt"] +
            0.25 * row["mean_hit_rate_2pt"] +
            0.20 * row["mean_hit_rate_4pt"] +
            0.15 * min(row["mean_test_trades"], 25.0) / 25.0 +
            0.10 * min(max(row["mean_avg_mr_pts"], 0.0), 12.0) / 12.0
        )

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["strategy_score", "mean_hit_rate_4pt", "mean_hit_rate_2pt", "mean_test_trades"],
        ascending=[False, False, False, False],
        kind="stable",
    )

    if len(df_trades):
        df_trades.to_csv(OUT_TRADES_CSV, index=False)
    df_fold_summary.to_csv(OUT_FOLD_SUMMARY_CSV, index=False)
    df_summary.to_csv(OUT_COMBO_SUMMARY_CSV, index=False)

    print("\nDone.")
    if len(df_trades):
        print(f"Wrote: {OUT_TRADES_CSV}")
    print(f"Wrote: {OUT_FOLD_SUMMARY_CSV}")
    print(f"Wrote: {OUT_COMBO_SUMMARY_CSV}")

    show_cols = [
        "combo_name",
        "positive_test_folds",
        "mean_test_trades",
        "mean_hit_rate_1pt",
        "mean_hit_rate_2pt",
        "mean_hit_rate_4pt",
        "mean_avg_mr_pts",
        "mean_avg_mae_pts",
        "mean_stop_hit_rate_proxy",
        f"mean_avg_pnl_proxy_tgt_1_stop_{STOP_PTS:g}",
        f"mean_avg_pnl_proxy_tgt_2_stop_{STOP_PTS:g}",
        f"mean_avg_pnl_proxy_tgt_4_stop_{STOP_PTS:g}",
        "strategy_score",
    ]
    print("\nTop combo strategy summaries:")
    print(df_summary[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
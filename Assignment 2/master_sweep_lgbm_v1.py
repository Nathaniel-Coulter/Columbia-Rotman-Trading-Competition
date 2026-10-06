# master_sweep_lgbm_v1.py
from __future__ import annotations

import json
from pathlib import Path
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm is not installed. Install with: pip install lightgbm") from e

try:
    import shap
except ImportError as e:
    raise ImportError("shap is not installed. Install with: pip install shap") from e


# ============================================================
# Paths
# ============================================================
FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events\event_features_v5_with_mr.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\master_sweep_lgbm_v1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_RESULTS_CSV = OUTPUT_DIR / "master_results.csv"
MASTER_RESULTS_PARQUET = OUTPUT_DIR / "master_results.parquet"
MASTER_CONFIG_JSON = OUTPUT_DIR / "master_config.json"


# ============================================================
# Sweep grid
# ============================================================
HORIZONS_S = [60, 180, 300, 600]
MILESTONE_TICKS = [4, 8, 12, 16, 24, 32, 40]  # 1,2,3,4,6,8,10 points
TICK_SIZE = 0.25

SIDE_BAND_COMBOS: List[Tuple[str, int]] = [
    ("upper", 1),
    ("lower", 1),
    ("upper", 2),
    ("lower", 2),
    ("upper", 3),
    ("lower", 3),
]

VOL_FILTERS = [
    {"name": "none", "col": None, "top_frac": None},
    {"name": "vol30_top30", "col": "realized_volatility_30s", "top_frac": 0.30},
    {"name": "vol1m_top30", "col": "realized_vol_1m", "top_frac": 0.30},
    {"name": "vol5m_top30", "col": "realized_vol_5m", "top_frac": 0.30},
]


# ============================================================
# Split / modeling
# ============================================================
TRAIN_FRAC = 0.65
VAL_FRAC = 0.20
TEST_FRAC = 0.15
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

MIN_ROWS_TOTAL = 400
MIN_POSITIVES_TRAIN = 40
MIN_POSITIVES_VAL = 20
MIN_POSITIVES_TEST = 20

LGB_PARAMS: Dict = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ============================================================
# Features
# ============================================================
FEATURE_COLS: List[str] = [
    "band_k",
    "side",
    "is_weekend",
    "entry_price",
    "distance0",
    "sigma0",
    "cooldown_seconds",
    "since_touch_any_seconds",
    "cooldown_bucket",
    "since_touch_any_bucket",
    "rv_pre_pts2",
    "trade_rate_pre_per_s",
    "sigma_lvl",
    "dsigma_dt",
    "vwap_slope_pts_per_s",
    "abs_vwap_slope_pts_per_s",
    "seconds_since_open",
    "time_of_day_sin",
    "time_of_day_cos",
    "session_bucket",
    "distance_from_session_vwap_anchor",
    "z_anchor_vwap",
    "vwap_drift",
    "vwap0_minus_open_pts",
    "range_position",
    "range_position_norm",
    "dist_day_high",
    "dist_day_low",
    "distance_from_range_extreme",
    "distance_from_range_extreme_norm",
    "midrange_position",
    "distance_from_session_midrange",
    "opening_range_width",
    "opening_range_position",
    "distance_from_prior_day_high",
    "distance_from_prior_day_low",
    "distance_from_overnight_high",
    "distance_from_overnight_low",
    "vwap_open",
    "session_high_running_t0",
    "session_low_running_t0",
    "midrange",
    "opening_range_high",
    "opening_range_low",
    "prior_day_rth_high",
    "prior_day_rth_low",
    "overnight_high",
    "overnight_low",
    "day_open",
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
    "vpoc",
    "vah",
    "val",
    "distance_from_vpoc",
    "distance_from_value_area_high",
    "distance_from_value_area_low",
    "inside_value_area_flag",
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
    "rsi_14",
    "stoch_k_14",
    "stoch_d_3",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_9",
    "ema_21",
    "ema_spread_9_21",
    "sma_20",
    "sma_50",
    "sma_spread_20_50",
    "obv_session",
    "distance_from_ema_21",
    "return_last_30s",
    "return_last_5m",
    "realized_vol_1m",
    "realized_vol_5m",
    "realized_volatility_30s",
    "vol_ratio",
    "tick_rate",
    "average_true_range",
]

CAT_COLS: List[str] = [
    "side",
    "session_bucket",
    "cooldown_bucket",
    "since_touch_any_bucket",
    "inside_value_area_flag",
]

ID_COLS: List[str] = ["event_id", "date_ymd", "ts"]

BASE_FEATURE_COLS: List[str] = [
    c for c in FEATURE_COLS
    if c not in {
        "time_of_day_sin",
        "time_of_day_cos",
        "opening_range_width",
        "range_position_norm",
        "distance_from_range_extreme",
        "distance_from_range_extreme_norm",
    }
]


# ============================================================
# Helpers
# ============================================================
def ensure_columns(df: pl.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    session_len_s = 6.5 * 60 * 60
    return df.with_columns([
        (pl.col("opening_range_high") - pl.col("opening_range_low")).alias("opening_range_width"),
        ((2.0 * np.pi * pl.col("seconds_since_open") / session_len_s).sin()).alias("time_of_day_sin"),
        ((2.0 * np.pi * pl.col("seconds_since_open") / session_len_s).cos()).alias("time_of_day_cos"),
        pl.min_horizontal(["dist_day_high", "dist_day_low"]).alias("distance_from_range_extreme"),
        (
            pl.min_horizontal(["dist_day_high", "dist_day_low"]) /
            (pl.col("dist_day_high") + pl.col("dist_day_low"))
        ).alias("distance_from_range_extreme_norm"),
        (
            pl.col("dist_day_low") /
            (pl.col("dist_day_high") + pl.col("dist_day_low"))
        ).alias("range_position_norm"),
    ])


def time_split_indices(n: int) -> Dict[str, np.ndarray]:
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    idx = np.arange(n)
    return {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:],
    }


def prep_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    for c in CAT_COLS:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("string").fillna("MISSING").astype("category")
    return pdf


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, split_name: str) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    unique_vals = np.unique(y_true)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(unique_vals) > 1 else np.nan
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(unique_vals) > 1 else np.nan

    y_prob_clip = np.clip(y_prob, 1e-12, 1 - 1e-12)

    return {
        "split": split_name,
        "n_rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_prob)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def build_shap_importance(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    max_rows: int = 1500,
) -> pd.DataFrame:
    if len(X_test) > max_rows:
        X_shap = X_test.iloc[:max_rows].copy()
    else:
        X_shap = X_test.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    if hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs_shap = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({
            "feature": X_shap.columns,
            "mean_abs_shap": mean_abs_shap,
        })
        .sort_values("mean_abs_shap", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def get_top_gain_features(model: lgb.LGBMClassifier, top_n: int = 10) -> List[str]:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    df_gain = pd.DataFrame({"feature": names, "gain": gain})
    df_gain = df_gain.sort_values("gain", ascending=False, kind="stable")
    return df_gain["feature"].head(top_n).tolist()


def get_top_shap_features(shap_imp_df: pd.DataFrame, top_n: int = 10) -> List[str]:
    return shap_imp_df["feature"].head(top_n).tolist()


def apply_vol_filter(df: pl.DataFrame, vol_filter: Dict) -> pl.DataFrame:
    if vol_filter["name"] == "none":
        return df

    vol_col = vol_filter["col"]
    top_frac = vol_filter["top_frac"]

    if vol_col is None or top_frac is None:
        return df

    df = df.filter(pl.col(vol_col).is_not_null())
    if df.height == 0:
        return df

    cutoff = df.select(pl.col(vol_col).quantile(1.0 - top_frac)).item()
    return df.filter(pl.col(vol_col) >= cutoff)


def run_one_experiment(
    df_all: pl.DataFrame,
    side: str,
    band_k: int,
    horizon_s: int,
    milestone_ticks: int,
    vol_filter: Dict,
) -> Dict:
    target_col = f"mr_pts_t{horizon_s}"
    threshold_points = milestone_ticks * TICK_SIZE

    required_cols = ID_COLS + BASE_FEATURE_COLS + [target_col]
    ensure_columns(df_all, required_cols)

    df = df_all.select(required_cols)

    # target exists
    df = df.filter(pl.col(target_col).is_not_null())

    # subset first
    df = df.filter(
        (pl.col("side") == side) &
        (pl.col("band_k") == band_k)
    )

    # derived features
    df = add_derived_features(df)

    # binary target
    df = df.with_columns(
        (pl.col(target_col) >= threshold_points).cast(pl.Int8).alias("target")
    )

    # vol regime filter
    df = apply_vol_filter(df, vol_filter)

    # sort chronologically after all filtering
    df = df.sort("ts")

    n_total = df.height
    if n_total < MIN_ROWS_TOTAL:
        return {
            "status": "skipped_too_few_rows",
            "side": side,
            "band_k": band_k,
            "horizon_s": horizon_s,
            "milestone_ticks": milestone_ticks,
            "threshold_points": threshold_points,
            "vol_filter_name": vol_filter["name"],
            "n_total": n_total,
        }

    pdf = prep_pandas(df)
    X = pdf[FEATURE_COLS].copy()
    y = pdf["target"].astype(int).to_numpy()

    split_idx = time_split_indices(len(pdf))

    X_train = X.iloc[split_idx["train"]].copy()
    y_train = y[split_idx["train"]]

    X_val = X.iloc[split_idx["val"]].copy()
    y_val = y[split_idx["val"]]

    X_test = X.iloc[split_idx["test"]].copy()
    y_test = y[split_idx["test"]]

    n_pos_train = int(y_train.sum())
    n_pos_val = int(y_val.sum())
    n_pos_test = int(y_test.sum())

    if (
        n_pos_train < MIN_POSITIVES_TRAIN
        or n_pos_val < MIN_POSITIVES_VAL
        or n_pos_test < MIN_POSITIVES_TEST
    ):
        return {
            "status": "skipped_too_few_positives",
            "side": side,
            "band_k": band_k,
            "horizon_s": horizon_s,
            "milestone_ticks": milestone_ticks,
            "threshold_points": threshold_points,
            "vol_filter_name": vol_filter["name"],
            "n_total": n_total,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_pos_train": n_pos_train,
            "n_pos_val": n_pos_val,
            "n_pos_test": n_pos_test,
        }

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_binary(y_val, val_pred, "val")
    test_metrics = evaluate_binary(y_test, test_pred, "test")

    shap_imp_df = build_shap_importance(model, X_test, max_rows=1500)
    top_shap_10 = get_top_shap_features(shap_imp_df, top_n=10)
    top_gain_10 = get_top_gain_features(model, top_n=10)

    out = {
        "status": "ok",
        "side": side,
        "band_k": band_k,
        "horizon_s": horizon_s,
        "target_col": target_col,
        "milestone_ticks": milestone_ticks,
        "threshold_points": threshold_points,
        "vol_filter_name": vol_filter["name"],
        "vol_filter_col": vol_filter["col"],
        "vol_filter_top_frac": vol_filter["top_frac"],

        "n_total": n_total,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_pos_train": n_pos_train,
        "n_pos_val": n_pos_val,
        "n_pos_test": n_pos_test,

        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),

        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_log_loss": val_metrics["log_loss"],
        "val_brier": val_metrics["brier_score"],

        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_log_loss": test_metrics["log_loss"],
        "test_brier": test_metrics["brier_score"],

        "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),

        "top_shap_1": top_shap_10[0] if len(top_shap_10) > 0 else None,
        "top_shap_2": top_shap_10[1] if len(top_shap_10) > 1 else None,
        "top_shap_3": top_shap_10[2] if len(top_shap_10) > 2 else None,
        "top_shap_4": top_shap_10[3] if len(top_shap_10) > 3 else None,
        "top_shap_5": top_shap_10[4] if len(top_shap_10) > 4 else None,
        "top_shap_6": top_shap_10[5] if len(top_shap_10) > 5 else None,
        "top_shap_7": top_shap_10[6] if len(top_shap_10) > 6 else None,
        "top_shap_8": top_shap_10[7] if len(top_shap_10) > 7 else None,
        "top_shap_9": top_shap_10[8] if len(top_shap_10) > 8 else None,
        "top_shap_10": top_shap_10[9] if len(top_shap_10) > 9 else None,

        "top_gain_1": top_gain_10[0] if len(top_gain_10) > 0 else None,
        "top_gain_2": top_gain_10[1] if len(top_gain_10) > 1 else None,
        "top_gain_3": top_gain_10[2] if len(top_gain_10) > 2 else None,
        "top_gain_4": top_gain_10[3] if len(top_gain_10) > 3 else None,
        "top_gain_5": top_gain_10[4] if len(top_gain_10) > 4 else None,
        "top_gain_6": top_gain_10[5] if len(top_gain_10) > 5 else None,
        "top_gain_7": top_gain_10[6] if len(top_gain_10) > 6 else None,
        "top_gain_8": top_gain_10[7] if len(top_gain_10) > 7 else None,
        "top_gain_9": top_gain_10[8] if len(top_gain_10) > 8 else None,
        "top_gain_10": top_gain_10[9] if len(top_gain_10) > 9 else None,
    }
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features parquet: {FEATURES_PATH}")

    print(f"Reading: {FEATURES_PATH}")
    df_all = pl.read_parquet(FEATURES_PATH)

    # base required columns across all runs
    required_cols = ID_COLS + BASE_FEATURE_COLS
    for h in HORIZONS_S:
        required_cols.append(f"mr_pts_t{h}")
    ensure_columns(df_all, required_cols)

    rows_out: List[Dict] = []

    sweep_grid = list(product(SIDE_BAND_COMBOS, HORIZONS_S, MILESTONE_TICKS, VOL_FILTERS))
    print(f"Total planned runs: {len(sweep_grid)}")

    for i, ((side, band_k), horizon_s, milestone_ticks, vol_filter) in enumerate(sweep_grid, start=1):
        print(
            f"[{i}/{len(sweep_grid)}] "
            f"side={side} band={band_k} horizon={horizon_s}s "
            f"ticks={milestone_ticks} vol={vol_filter['name']}"
        )

        try:
            row = run_one_experiment(
                df_all=df_all,
                side=side,
                band_k=band_k,
                horizon_s=horizon_s,
                milestone_ticks=milestone_ticks,
                vol_filter=vol_filter,
            )
        except Exception as e:
            row = {
                "status": "error",
                "side": side,
                "band_k": band_k,
                "horizon_s": horizon_s,
                "milestone_ticks": milestone_ticks,
                "threshold_points": milestone_ticks * TICK_SIZE,
                "vol_filter_name": vol_filter["name"],
                "error_message": str(e),
            }

        rows_out.append(row)

        # incremental save every run
        df_results = pd.DataFrame(rows_out)
        df_results.to_csv(MASTER_RESULTS_CSV, index=False)
        pl.from_pandas(df_results).write_parquet(MASTER_RESULTS_PARQUET)

    config = {
        "features_path": str(FEATURES_PATH),
        "output_dir": str(OUTPUT_DIR),
        "horizons_s": HORIZONS_S,
        "milestone_ticks": MILESTONE_TICKS,
        "tick_size": TICK_SIZE,
        "side_band_combos": SIDE_BAND_COMBOS,
        "vol_filters": VOL_FILTERS,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "min_rows_total": MIN_ROWS_TOTAL,
        "min_positives_train": MIN_POSITIVES_TRAIN,
        "min_positives_val": MIN_POSITIVES_VAL,
        "min_positives_test": MIN_POSITIVES_TEST,
        "feature_count": len(FEATURE_COLS),
        "categorical_cols": CAT_COLS,
        "lightgbm_params": LGB_PARAMS,
    }
    with open(MASTER_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Wrote: {MASTER_RESULTS_CSV}")
    print(f"Wrote: {MASTER_RESULTS_PARQUET}")
    print(f"Wrote: {MASTER_CONFIG_JSON}")


if __name__ == "__main__":
    main()
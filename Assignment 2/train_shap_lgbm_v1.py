# train_shap_lgbm_v1.py
# ------------------------------------------------------------
# Event-level LightGBM + SHAP model for VWAP mean-reversion.
#
# Target:
#   y = 1[mfe_pts_t60 >= 1.0]   # 1.0 point = 4 ticks in ES
#
# Split:
#   train = first 65%
#   val   = next 20%
#   test  = last 15%
#   sorted by ts (time-based, no leakage)
#
# Outputs:
#   metrics.csv
#   feature_importance_shap.csv
#   test_predictions.parquet
#   config.json
#   feature_importance_bar_chart.png
#
# Notes:
# - Reads event_features_v1.parquet
# - Uses only the explicit feature list below
# - Keeps structural levels, as requested
# - LightGBM handles NaNs in numeric columns
# - Categorical columns are cast to pandas category dtype
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "lightgbm is not installed. Install with: pip install lightgbm"
    ) from e

try:
    import shap
except ImportError as e:
    raise ImportError(
        "shap is not installed. Install with: pip install shap"
    ) from e

# ------------------------------------------------------------
# Modeling target / split
# ------------------------------------------------------------
TARGET_HORIZON_COL = "mfe_pts_t600"
TARGET_THRESHOLD_POINTS = 4.0  # 1 ES point = 4 ticks

TRAIN_FRAC = 0.65
VAL_FRAC = 0.20
TEST_FRAC = 0.15

assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
FEATURES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v1.parquet"
)

RUN_NAME = f"shap_lgbm_{TARGET_HORIZON_COL}_{TARGET_THRESHOLD_POINTS}pt"

OUTPUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs"
) / RUN_NAME

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUT_DIR / "metrics.csv"
SHAP_IMPORTANCE_CSV = OUTPUT_DIR / "feature_importance_shap.csv"
TEST_PREDICTIONS_PARQUET = OUTPUT_DIR / "test_predictions.parquet"
CONFIG_JSON = OUTPUT_DIR / "config.json"
BAR_CHART_PNG = OUTPUT_DIR / "feature_importance_bar_chart.png"

# ------------------------------------------------------------
# shap_lgbm_v1 feature list (user-approved)
# ------------------------------------------------------------
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
]

CAT_COLS: List[str] = [
    "side",
    "session_bucket",
    "cooldown_bucket",
    "since_touch_any_bucket",
]

ID_COLS: List[str] = [
    "event_id",
    "date_ymd",
    "ts",
]

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

REQUIRED_COLS = ID_COLS + BASE_FEATURE_COLS + [TARGET_HORIZON_COL]

# ------------------------------------------------------------
# LightGBM params
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_columns(df: pl.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_target(df: pl.DataFrame) -> pl.DataFrame:
    # Drop rows where target horizon is null
    df = df.filter(pl.col(TARGET_HORIZON_COL).is_not_null())

    # Binary target: did event achieve >= 1.0 point favorable excursion within 60s?
    df = df.with_columns(
        (pl.col(TARGET_HORIZON_COL) >= TARGET_THRESHOLD_POINTS)
        .cast(pl.Int8)
        .alias("target")
    )
    return df

def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    # RTH session length in seconds: 14:30 -> 21:00 = 6.5 hours
    session_len_s = 6.5 * 60 * 60

    return df.with_columns([
        # Opening range width
        (pl.col("opening_range_high") - pl.col("opening_range_low"))
        .alias("opening_range_width"),

        # Cyclical intraday clock from seconds_since_open
        (
            (2.0 * np.pi * pl.col("seconds_since_open") / session_len_s)
            .sin()
        ).alias("time_of_day_sin"),

        (
            (2.0 * np.pi * pl.col("seconds_since_open") / session_len_s)
            .cos()
        ).alias("time_of_day_cos"),

        pl.min_horizontal(["dist_day_high", "dist_day_low"]).alias("distance_from_range_extreme"),

        (
            pl.min_horizontal(["dist_day_high", "dist_day_low"]) /
            (pl.col("dist_day_high") + pl.col("dist_day_low"))
        ).alias("distance_from_range_extreme_norm"),

        (
            pl.col("dist_day_low") /
            (pl.col("dist_day_high") + pl.col("dist_day_low"))
        ).alias("range_position_norm")
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


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, split_name: str) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Guard small/degenerate edge cases
    unique_vals = np.unique(y_true)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(unique_vals) > 1 else np.nan
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(unique_vals) > 1 else np.nan

    # Clip for log loss
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


def prep_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()

    for c in CAT_COLS:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna("MISSING").astype("category")

    return pdf


def build_shap_importance(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    max_rows: int = 5000,
) -> pd.DataFrame:
    if len(X_test) > max_rows:
        X_shap = X_test.iloc[:max_rows].copy()
    else:
        X_shap = X_test.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Binary classification handling varies by SHAP / LightGBM versions
    if isinstance(shap_values, list):
        # Usually [class0, class1]
        sv = shap_values[1]
    else:
        sv = shap_values

    # If SHAP returns 3D, squeeze positive class
    if hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs_shap = np.abs(sv).mean(axis=0)

    imp = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False, kind="stable")

    return imp


def save_bar_chart(imp_df: pd.DataFrame, path: Path, top_n: int = 25) -> None:
    plot_df = imp_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features parquet: {FEATURES_PATH}")

    print(f"Reading: {FEATURES_PATH}")
    df = pl.read_parquet(FEATURES_PATH)

    ensure_columns(df, REQUIRED_COLS)

    # Keep only what we need for this model run
    keep_cols = ID_COLS + BASE_FEATURE_COLS + [TARGET_HORIZON_COL]
    df = df.select(keep_cols)

    # Build target and sort by time
    df = make_target(df).sort("ts")
    df = add_derived_features(df)
    ensure_columns(df, FEATURE_COLS + ["target"])

    n_total = df.height
    if n_total < 100:
        raise ValueError(f"Too few usable rows after target filtering: {n_total}")

    print(f"Rows after target filtering: {n_total:,}")

    # Convert to pandas for LightGBM
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

    meta_test = pdf.iloc[split_idx["test"]][ID_COLS].copy()

    print(
        f"Split sizes | train={len(X_train):,} "
        f"val={len(X_val):,} test={len(X_test):,}"
    )
    print(
        f"Positive rates | train={y_train.mean():.4f} "
        f"val={y_val.mean():.4f} test={y_test.mean():.4f}"
    )

    model = lgb.LGBMClassifier(**LGB_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    metrics_rows = [
        evaluate_binary(y_val, val_pred, "val"),
        evaluate_binary(y_test, test_pred, "test"),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Test predictions parquet
    test_pred_df = pd.DataFrame({
        "event_id": meta_test["event_id"].values,
        "date_ymd": meta_test["date_ymd"].values,
        "ts": meta_test["ts"].values,
        "y_true": y_test,
        "y_pred_proba": test_pred,
    })
    pl.from_pandas(test_pred_df).write_parquet(TEST_PREDICTIONS_PARQUET)

    # SHAP feature importance
    shap_imp_df = build_shap_importance(model, X_test, max_rows=5000)
    shap_imp_df.to_csv(SHAP_IMPORTANCE_CSV, index=False)
    save_bar_chart(shap_imp_df, BAR_CHART_PNG, top_n=25)

    # Config / manifest
    config = {
        "features_path": str(FEATURES_PATH),
        "output_dir": str(OUTPUT_DIR),
        "target": {
            "source_column": TARGET_HORIZON_COL,
            "rule": f"{TARGET_HORIZON_COL} >= {TARGET_THRESHOLD_POINTS}",
            "label_name": "target",
        },
        "split": {
            "type": "time_based_sorted_by_ts",
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
        },
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CAT_COLS,
        "id_cols": ID_COLS,
        "lightgbm_params": LGB_PARAMS,
        "n_rows_total_after_target_filter": int(n_total),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_iteration_": int(getattr(model, "best_iteration_", -1) or -1),
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Wrote: {METRICS_CSV}")
    print(f"Wrote: {SHAP_IMPORTANCE_CSV}")
    print(f"Wrote: {TEST_PREDICTIONS_PARQUET}")
    print(f"Wrote: {CONFIG_JSON}")
    print(f"Wrote: {BAR_CHART_PNG}")

    print("\nValidation/Test metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop 15 SHAP features:")
    print(shap_imp_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
# inspect_master_results_v1.py
from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\master_sweep_lgbm_v1\master_results.csv"
)

TOP_N = 25
RANK_COL = "test_roc_auc"   # can change to val_roc_auc if you want


def main():
    df = pd.read_csv(CSV_PATH)

    print("\n=== BASIC SHAPE ===")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns)}")

    print("\n=== STATUS COUNTS ===")
    print(df["status"].value_counts(dropna=False))

    n_ok = int((df["status"] == "ok").sum())
    n_not_ok = int((df["status"] != "ok").sum())

    print("\n=== OK VS NON-OK ===")
    print(f"status == 'ok': {n_ok:,}")
    print(f"status != 'ok': {n_not_ok:,}")

    print("\n=== NON-OK BREAKDOWN ===")
    print(df.loc[df["status"] != "ok", "status"].value_counts(dropna=False))

    if "error_message" in df.columns:
        df_err = df.loc[df["status"] == "error", ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name", "error_message"]]
        if not df_err.empty:
            print("\n=== ERROR ROWS ===")
            print(df_err.head(10).to_string(index=False))

    df_ok = df.loc[df["status"] == "ok"].copy()
    if df_ok.empty:
        print("\nNo successful runs found.")
        return

    # numeric sort safety
    df_ok[RANK_COL] = pd.to_numeric(df_ok[RANK_COL], errors="coerce")
    df_ok = df_ok.dropna(subset=[RANK_COL]).sort_values(RANK_COL, ascending=False)

    print(f"\n=== TOP {TOP_N} OK RUNS BY {RANK_COL} ===")
    cols_to_show = [
        "side",
        "band_k",
        "horizon_s",
        "milestone_ticks",
        "threshold_points",
        "vol_filter_name",
        "n_total",
        "n_train",
        "n_val",
        "n_test",
        "n_pos_train",
        "n_pos_val",
        "n_pos_test",
        "val_roc_auc",
        "test_roc_auc",
        "val_pr_auc",
        "test_pr_auc",
        "test_log_loss",
        "test_brier",
        "best_iteration",
        "top_shap_1",
        "top_shap_2",
        "top_shap_3",
        "top_gain_1",
        "top_gain_2",
        "top_gain_3",
    ]
    cols_to_show = [c for c in cols_to_show if c in df_ok.columns]
    print(df_ok[cols_to_show].head(TOP_N).to_string(index=False))

    # concentration in top N
    top_df = df_ok.head(TOP_N).copy()

    print(f"\n=== TOP {TOP_N} PARAMETER CONCENTRATION ===")
    for c in ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]:
        if c in top_df.columns:
            print(f"\n-- {c} --")
            print(top_df[c].value_counts(dropna=False))

    # top decile concentration
    top_decile_n = max(1, len(df_ok) // 10)
    top_decile = df_ok.head(top_decile_n).copy()

    print(f"\n=== TOP DECILE ({top_decile_n} RUNS) PARAMETER CONCENTRATION ===")
    for c in ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]:
        if c in top_decile.columns:
            print(f"\n-- {c} --")
            print(top_decile[c].value_counts(dropna=False))

    # best average score by region
    print(f"\n=== AVERAGE {RANK_COL} BY PARAMETER ===")
    for c in ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]:
        if c in df_ok.columns:
            print(f"\n-- {c} --")
            print(
                df_ok.groupby(c, dropna=False)[RANK_COL]
                .agg(["count", "mean", "median", "max"])
                .sort_values("mean", ascending=False)
                .round(6)
            )

    # strongest exact combos
    combo_cols = ["side", "band_k", "horizon_s", "milestone_ticks", "vol_filter_name"]
    combo_cols = [c for c in combo_cols if c in df_ok.columns]

    print(f"\n=== BEST EXACT COMBINATIONS BY MEAN {RANK_COL} ===")
    combo_summary = (
        df_ok.groupby(combo_cols, dropna=False)[RANK_COL]
        .agg(["count", "mean", "median", "max"])
        .sort_values(["mean", "max"], ascending=False)
        .round(6)
    )
    print(combo_summary.head(20).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
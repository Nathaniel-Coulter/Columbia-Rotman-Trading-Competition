# local_parameter_sensitivity_mc.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import math
import random

import polars as pl


# ============================================================
# Paths
# ============================================================

TRADE_LEVEL_PATH = Path(
r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\strategy_gate_simulator\strategy_gate_trade_level.csv"
)

OUTPUT_DIR = Path(
r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\outputs\local_parameter_sensitivity_mc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Monte Carlo settings
# ============================================================

N_SEEDS = 20
BASE_SEED = 42
N_FOLDS = 4
MIN_TEST_TRADES = 25


# ============================================================
# Parameter neighborhood
# ============================================================

TARGET_GRID = [3.0, 4.0, 5.0]
STOP_GRID   = [5.0, 6.0, 7.0, 8.0]
TIME_GRID   = [600, 900]

GATE_NAME = "G1_ALL2_bestask_aggbuy"

# ============================================================
# Build strategy grid
# ============================================================

STRATEGIES = []

for t in TARGET_GRID:
    for s in STOP_GRID:
        for tm in TIME_GRID:

            STRATEGIES.append(
                {
                    "strategy_name": f"G1_t{int(t)}_s{int(s)}_t{int(tm)}",
                    "gate_name": GATE_NAME,
                    "target_pts": t,
                    "stop_pts": s,
                    "time_exit_s": tm,
                }
            )


# ============================================================
# Helpers
# ============================================================

def pct(num, den):
    if den == 0:
        return math.nan
    return num / den


def safe_mean(df,col):
    if df.height == 0:
        return math.nan
    x = df[col].mean()
    return float(x) if x is not None else math.nan


def safe_std(df,col):

    if df.height <= 1:
        return 0.0

    x = df[col].std()
    return float(x) if x is not None else 0.0


def fold_boundaries(n,n_folds):

    out=[]

    for i in range(n_folds):

        lo=int(round(i*n/n_folds))
        hi=int(round((i+1)*n/n_folds))

        out.append((lo,hi))

    return out


# ============================================================
# Trade summarizer
# ============================================================

def summarize_trade_block(df):

    n=df.height

    if n==0:

        return {
        "n_trades":0,
        "win_rate":math.nan,
        "avg_pnl_pts":math.nan,
        "avg_mfe_pts":math.nan,
        "avg_mae_pts":math.nan,
        "avg_end_ret_pts":math.nan,
        "profit_factor_proxy":math.nan,
        "time_exit_rate":math.nan,
        }

    pnl=df["pnl_pts"]

    mfe=df["mfe_pts"]
    mae=df["mae_pts"]

    wins=(pnl>0).sum()

    exit_counts=(
        df.group_by("exit_reason")
        .agg(pl.len().alias("n"))
        .to_dicts()
    )

    exit_map={r["exit_reason"]:r["n"] for r in exit_counts}

    pnl_list=[float(x) for x in pnl.to_list()]

    wins=[x for x in pnl_list if x>0]
    losses=[x for x in pnl_list if x<0]

    total_profit=sum(wins)
    gross_loss=abs(sum(losses))

    pf=total_profit/gross_loss if gross_loss>0 else math.nan

    return {

        "n_trades":n,
        "win_rate":float((pnl>0).mean()),
        "avg_pnl_pts":float(pnl.mean()),
        "avg_mfe_pts":float(mfe.mean()),
        "avg_mae_pts":float(mae.mean()),
        "avg_end_ret_pts":float(df["end_ret_pts"].mean()),
        "profit_factor_proxy":pf,
        "time_exit_rate":pct(exit_map.get("time",0),n),
    }


# ============================================================
# Attach trade_day
# ============================================================

def attach_trade_day(df):

    if "date_ymd" in df.columns:

        return df.with_columns(
            pl.col("date_ymd").cast(pl.Utf8).alias("trade_day")
        )

    raise ValueError("trade_day column not found")


# ============================================================
# Main
# ============================================================

def main():

    df=pl.read_csv(
        TRADE_LEVEL_PATH,
        try_parse_dates=True,
        infer_schema_length=10000,
    )

    df=attach_trade_day(df)

    print("\n--------------------------------------")
    print("LOCAL PARAMETER SENSITIVITY MC")
    print("--------------------------------------")

    print(f"Rows loaded = {df.height:,}")
    print(f"Strategy grid = {len(STRATEGIES)}")


    # ========================================================
    # Filter strategies
    # ========================================================

    frames=[]

    for s in STRATEGIES:

        sub=df.filter(

        (pl.col("gate_name")==s["gate_name"]) &
        (pl.col("target_pts")==s["target_pts"]) &
        (pl.col("stop_pts")==s["stop_pts"]) &
        (pl.col("time_exit_s")==s["time_exit_s"])

        ).with_columns(

        pl.lit(s["strategy_name"]).alias("strategy_name")

        )

        frames.append(sub)

    trade_df=pl.concat(frames,how="vertical")

    trade_df.write_csv(OUTPUT_DIR/"strategy_subset.csv")


    unique_days=sorted(set(trade_df["trade_day"].to_list()))

    all_seed_results=[]
    all_seed_folds=[]


    # ========================================================
    # Monte Carlo seeds
    # ========================================================

    for seed in range(BASE_SEED,BASE_SEED+N_SEEDS):

        print(f"\nRunning seed {seed}")

        rng=random.Random(seed)

        shuffled_days=unique_days[:]
        rng.shuffle(shuffled_days)

        fold_ranges=fold_boundaries(len(shuffled_days),N_FOLDS)

        fold_records=[]


        for fold_idx,(test_lo,test_hi) in enumerate(fold_ranges,start=1):

            test_days=set(shuffled_days[test_lo:test_hi])
            train_days=set(shuffled_days[:test_lo])

            if len(test_days)==0 or len(train_days)==0:
                continue

            for s in STRATEGIES:

                sdf=trade_df.filter(pl.col("strategy_name")==s["strategy_name"])

                train_df=sdf.filter(pl.col("trade_day").is_in(sorted(train_days)))
                test_df=sdf.filter(pl.col("trade_day").is_in(sorted(test_days)))

                train_stats=summarize_trade_block(train_df)
                test_stats=summarize_trade_block(test_df)

                fold_records.append({

                "seed":seed,
                "strategy_name":s["strategy_name"],
                "target_pts":s["target_pts"],
                "stop_pts":s["stop_pts"],
                "time_exit_s":s["time_exit_s"],
                "fold_idx":fold_idx,

                "train_n_trades":train_stats["n_trades"],
                "test_n_trades":test_stats["n_trades"],

                "test_win_rate":test_stats["win_rate"],
                "test_avg_pnl_pts":test_stats["avg_pnl_pts"],
                "test_avg_mfe_pts":test_stats["avg_mfe_pts"],
                "test_avg_mae_pts":test_stats["avg_mae_pts"],
                "test_avg_end_ret_pts":test_stats["avg_end_ret_pts"],
                "test_profit_factor_proxy":test_stats["profit_factor_proxy"],

                })

        fold_df=pl.from_dicts(fold_records)

        all_seed_folds.append(fold_df)


        summary_records=[]

        for s in STRATEGIES:

            fs=fold_df.filter(pl.col("strategy_name")==s["strategy_name"])

            valid=fs.filter(pl.col("test_n_trades")>=MIN_TEST_TRADES)

            use_df=valid if valid.height>0 else fs

            mean_pnl=safe_mean(use_df,"test_avg_pnl_pts")
            mean_wr=safe_mean(use_df,"test_win_rate")

            std_pnl=safe_std(use_df,"test_avg_pnl_pts")
            std_wr=safe_std(use_df,"test_win_rate")

            robust_score=(

            1.1*(0 if math.isnan(mean_pnl) else mean_pnl)
            +0.4*(0 if math.isnan(mean_wr) else mean_wr)
            -0.03*(0 if math.isnan(safe_mean(use_df,"test_avg_mae_pts")) else safe_mean(use_df,"test_avg_mae_pts"))

            )

            stability_score=(

            (0 if math.isnan(mean_pnl) else mean_pnl)
            +0.3*(0 if math.isnan(mean_wr) else mean_wr)
            -0.25*std_wr
            -0.1*std_pnl

            )

            summary_records.append({

            "seed":seed,
            "strategy_name":s["strategy_name"],
            "target_pts":s["target_pts"],
            "stop_pts":s["stop_pts"],
            "time_exit_s":s["time_exit_s"],
            "mean_test_pnl":mean_pnl,
            "mean_test_win_rate":mean_wr,
            "robust_score":robust_score,
            "stability_score":stability_score,

            })

        seed_summary=pl.from_dicts(summary_records)

        all_seed_results.append(seed_summary)


    # ========================================================
    # Aggregate seeds
    # ========================================================

    all_results=pl.concat(all_seed_results,how="vertical")

    avg_results=(
        all_results
        .group_by(["strategy_name","target_pts","stop_pts","time_exit_s"])
        .agg([

            pl.mean("mean_test_pnl").alias("avg_pnl"),
            pl.std("mean_test_pnl").alias("pnl_stability"),

            pl.mean("mean_test_win_rate").alias("avg_win_rate"),
            pl.std("mean_test_win_rate").alias("win_rate_stability"),

            pl.mean("robust_score").alias("avg_robust_score"),
            pl.mean("stability_score").alias("avg_stability_score")

        ])
        .sort("avg_robust_score",descending=True)
    )

    avg_results.write_csv(OUTPUT_DIR/"local_parameter_surface_summary.csv")

    print("\n======================================")
    print("LOCAL PARAMETER SURFACE RESULTS")
    print("======================================")

    print(avg_results.head(30))


if __name__=="__main__":
    main()
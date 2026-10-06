# v5_spot_check.py
import polars as pl

df = pl.read_parquet(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\shock_events\shock_event_features_v5.parquet"
)

check_cols = [
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

existing = [c for c in check_cols if c in df.columns]
missing = [c for c in check_cols if c not in df.columns]

print("Missing columns:", missing)

if existing:
    print("\nNull counts:")
    print(
        df.select(
            [pl.len().alias("n")]
            + [pl.col(c).null_count().alias(f"{c}_nulls") for c in existing]
        )
    )

    print("\nMin / Max sanity:")
    exprs = []
    for c in existing:
        exprs.extend([
            pl.col(c).drop_nulls().min().alias(f"{c}_min"),
            pl.col(c).drop_nulls().max().alias(f"{c}_max"),
        ])
    print(df.select(exprs))

    print("\nMeans:")
    print(
        df.select(
            [pl.col(c).drop_nulls().mean().alias(f"{c}_mean") for c in existing]
        )
    )

    print("\nSample rows:")
    sample_cols = [c for c in [
        "date_ymd",
        "ts",
        "band_k",
        "side",
        "entry_price",
        "sigma0",
        "rsi_14",
        "stoch_k_14",
        "macd",
        "ema_21",
        "distance_from_ema_21",
        "return_last_30s",
        "realized_vol_1m",
        "realized_vol_5m",
        "vol_ratio",
        "tick_rate",
        "average_true_range",
    ] if c in df.columns]

    print(
        df.select(sample_cols)
        .filter(pl.col("rsi_14").is_not_null() if "rsi_14" in df.columns else pl.lit(True))
        .head(10)
    )

    print("\nIndicator consistency checks:")
    consistency_exprs = []

    if all(c in df.columns for c in ["ema_9", "ema_21", "ema_spread_9_21"]):
        consistency_exprs.append(
            (
                (pl.col("ema_9") - pl.col("ema_21")) - pl.col("ema_spread_9_21")
            ).abs().drop_nulls().max().alias("ema_spread_max_abs_error")
        )

    if all(c in df.columns for c in ["sma_20", "sma_50", "sma_spread_20_50"]):
        consistency_exprs.append(
            (
                (pl.col("sma_20") - pl.col("sma_50")) - pl.col("sma_spread_20_50")
            ).abs().drop_nulls().max().alias("sma_spread_max_abs_error")
        )

    if all(c in df.columns for c in ["macd", "macd_signal", "macd_hist"]):
        consistency_exprs.append(
            (
                (pl.col("macd") - pl.col("macd_signal")) - pl.col("macd_hist")
            ).abs().drop_nulls().max().alias("macd_hist_max_abs_error")
        )

    if consistency_exprs:
        print(df.select(consistency_exprs))

    print("\nRange sanity flags:")
    range_exprs = []

    if "rsi_14" in df.columns:
        range_exprs.append(
            (
                ((pl.col("rsi_14") < 0) | (pl.col("rsi_14") > 100))
                .sum()
                .alias("rsi_out_of_range_count")
            )
        )

    if "stoch_k_14" in df.columns:
        range_exprs.append(
            (
                ((pl.col("stoch_k_14") < 0) | (pl.col("stoch_k_14") > 100))
                .sum()
                .alias("stoch_k_out_of_range_count")
            )
        )

    if "stoch_d_3" in df.columns:
        range_exprs.append(
            (
                ((pl.col("stoch_d_3") < 0) | (pl.col("stoch_d_3") > 100))
                .sum()
                .alias("stoch_d_out_of_range_count")
            )
        )

    if "tick_rate" in df.columns:
        range_exprs.append(
            (pl.col("tick_rate") < 0).sum().alias("tick_rate_negative_count")
        )

    if "realized_vol_1m" in df.columns:
        range_exprs.append(
            (pl.col("realized_vol_1m") < 0).sum().alias("realized_vol_1m_negative_count")
        )

    if "realized_vol_5m" in df.columns:
        range_exprs.append(
            (pl.col("realized_vol_5m") < 0).sum().alias("realized_vol_5m_negative_count")
        )

    if "average_true_range" in df.columns:
        range_exprs.append(
            (pl.col("average_true_range") < 0).sum().alias("atr_negative_count")
        )

    if range_exprs:
        print(df.select(range_exprs))
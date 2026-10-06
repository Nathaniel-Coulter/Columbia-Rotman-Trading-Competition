# build_event_features_v5.py
from __future__ import annotations

from bisect import bisect_right, bisect_left
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import math
import polars as pl


# -----------------------------
# Paths
# -----------------------------
BASE_EVENTS_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v4.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUT_FILE = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v5.parquet"
)


# -----------------------------
# Constants
# -----------------------------
TICK = 0.25

RET_30S = 30
RET_5M = 300

VOL_30S = 30
VOL_1M = 60
VOL_5M = 300

TICK_RATE_WINDOW_S = 30

RSI_N = 14
STOCH_N = 14
STOCH_D_N = 3
EMA_FAST = 9
EMA_SLOW = 21
SMA_FAST = 20
SMA_SLOW = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_N = 14


# -----------------------------
# Helpers
# -----------------------------
def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


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
    return {
        f"{y:04d}-{m:02d}-{d:02d}": p
        for (y, m, d, p) in iter_day_parquets(root)
    }


def window_bounds(us_list: List[int], t_us: int, window_s: int) -> tuple[int, int]:
    start_us = t_us - int(window_s * 1_000_000)
    i0 = bisect_left(us_list, start_us)
    i1 = bisect_right(us_list, t_us)
    return i0, i1


def lagged_value_at_or_before(
    us_list: List[int],
    values: List[Optional[float]],
    t_us: int,
    lag_s: int,
) -> Optional[float]:
    target_us = t_us - int(lag_s * 1_000_000)
    j = bisect_right(us_list, target_us) - 1
    if j < 0:
        return None
    return values[j]


def realized_vol_from_prices(prices: List[float]) -> Optional[float]:
    if len(prices) < 2:
        return None
    rets = []
    prev = prices[0]
    for p in prices[1:]:
        if prev is None or p is None or prev <= 0 or p <= 0:
            prev = p
            continue
        rets.append(math.log(p / prev))
        prev = p
    if not rets:
        return None
    return math.sqrt(sum(r * r for r in rets))


def rolling_sma(xs: List[Optional[float]], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(xs)
    if n <= 0:
        return out
    window: List[float] = []
    s = 0.0
    for i, x in enumerate(xs):
        if x is None:
            window.append(float("nan"))
        else:
            window.append(float(x))
            s += float(x)

        if i >= n:
            old = window[i - n]
            if not math.isnan(old):
                s -= old

        if i >= n - 1:
            vals = window[max(0, i - n + 1): i + 1]
            if any(math.isnan(v) for v in vals):
                out[i] = None
            else:
                out[i] = s / n
    return out


def rolling_ema(xs: List[Optional[float]], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(xs)
    if not xs:
        return out
    alpha = 2.0 / (n + 1.0)
    ema: Optional[float] = None
    for i, x in enumerate(xs):
        if x is None:
            out[i] = ema
            continue
        x = float(x)
        if ema is None:
            ema = x
        else:
            ema = alpha * x + (1.0 - alpha) * ema
        out[i] = ema
    return out


def rolling_rsi_wilder(closes: List[Optional[float]], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) < n + 1:
        return out

    gains: List[float] = [0.0] * len(closes)
    losses: List[float] = [0.0] * len(closes)

    for i in range(1, len(closes)):
        c0 = closes[i - 1]
        c1 = closes[i]
        if c0 is None or c1 is None:
            continue
        d = float(c1) - float(c0)
        gains[i] = max(d, 0.0)
        losses[i] = max(-d, 0.0)

    avg_gain = sum(gains[1:n + 1]) / n
    avg_loss = sum(losses[1:n + 1]) / n

    if avg_loss == 0:
        out[n] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[n] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(n + 1, len(closes)):
        avg_gain = ((avg_gain * (n - 1)) + gains[i]) / n
        avg_loss = ((avg_loss * (n - 1)) + losses[i]) / n
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


def rolling_stoch(
    highs: List[Optional[float]],
    lows: List[Optional[float]],
    closes: List[Optional[float]],
    n: int,
    d_n: int,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    k_vals: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if i < n - 1:
            continue
        h_window = highs[i - n + 1:i + 1]
        l_window = lows[i - n + 1:i + 1]
        c = closes[i]
        if c is None or any(v is None for v in h_window) or any(v is None for v in l_window):
            continue
        hh = max(float(v) for v in h_window if v is not None)
        ll = min(float(v) for v in l_window if v is not None)
        if hh == ll:
            k_vals[i] = 50.0
        else:
            k_vals[i] = 100.0 * (float(c) - ll) / (hh - ll)

        # clip %K to [0,100] to prevent floating point drift
        if k_vals[i] is not None:
            k_vals[i] = min(max(k_vals[i], 0.0), 100.0)

    d_vals = rolling_sma(k_vals, d_n)
    # clip %D to [0,100]
    d_vals = [
        None if x is None else min(max(float(x), 0.0), 100.0)
        for x in d_vals
    ]

    return k_vals, d_vals


def rolling_atr(
    highs: List[Optional[float]],
    lows: List[Optional[float]],
    closes: List[Optional[float]],
    n: int,
) -> List[Optional[float]]:
    tr: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        h = highs[i]
        l = lows[i]
        if h is None or l is None:
            continue
        if i == 0 or closes[i - 1] is None:
            tr[i] = float(h) - float(l)
        else:
            pc = float(closes[i - 1])
            tr[i] = max(
                float(h) - float(l),
                abs(float(h) - pc),
                abs(float(l) - pc),
            )
    return rolling_sma(tr, n)


# -----------------------------
# Loaders
# -----------------------------
def load_trade_day(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)

    required = {"ts", "type", "price", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Combined parquet missing required trade cols: {missing}")

    return (
        df
        .filter(
            (pl.col("type") == 2) &
            pl.col("price").is_not_null() &
            pl.col("volume").is_not_null()
        )
        .select(["ts", "price", "volume"])
        .sort("ts")
    )


# -----------------------------
# Null filler
# -----------------------------
def add_null_v5_columns(r: Dict[str, Any]) -> None:
    r.update({
        # indicators
        "rsi_14": None,
        "stoch_k_14": None,
        "stoch_d_3": None,
        "macd": None,
        "macd_signal": None,
        "macd_hist": None,
        "ema_9": None,
        "ema_21": None,
        "ema_spread_9_21": None,
        "sma_20": None,
        "sma_50": None,
        "sma_spread_20_50": None,
        "obv_session": None,
        "distance_from_ema_21": None,

        # volatility / speed
        "return_last_30s": None,
        "return_last_5m": None,
        "realized_vol_1m": None,
        "realized_vol_5m": None,
        "realized_volatility_30s": None,
        "vol_ratio": None,
        "tick_rate": None,
        "average_true_range": None,
    })


# -----------------------------
# Per-day worker
# -----------------------------
def process_event_day(
    day_ymd: str,
    full_path: Optional[str],
    df_day_events: pl.DataFrame,
) -> List[Dict[str, Any]]:
    rows_out_day: List[Dict[str, Any]] = []

    if full_path is None:
        for r in df_day_events.to_dicts():
            add_null_v5_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    full_path = Path(full_path)
    if not full_path.exists():
        for r in df_day_events.to_dicts():
            add_null_v5_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    df_tr = load_trade_day(full_path)
    if df_tr.height == 0:
        for r in df_day_events.to_dicts():
            add_null_v5_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    # -------------------------
    # Raw trade series
    # -------------------------
    tr_ts = df_tr["ts"].to_list()
    tr_px = [float(x) for x in df_tr["price"].to_list()]
    tr_vol = [float(x) for x in df_tr["volume"].to_list()]
    tr_us = [dt_to_us(x) for x in tr_ts]

    # -------------------------
    # 1-minute completed bars for indicators
    # -------------------------
    df_min = (
        df_tr
        .with_columns(pl.col("ts").dt.truncate("1m").alias("bar_min"))
        .group_by("bar_min")
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ])
        .sort("bar_min")
    )

    min_ts = df_min["bar_min"].to_list()
    min_us = [dt_to_us(x) for x in min_ts]
    min_open = [float(x) if x is not None else None for x in df_min["open"].to_list()]
    min_high = [float(x) if x is not None else None for x in df_min["high"].to_list()]
    min_low = [float(x) if x is not None else None for x in df_min["low"].to_list()]
    min_close = [float(x) if x is not None else None for x in df_min["close"].to_list()]
    min_vol = [float(x) if x is not None else 0.0 for x in df_min["volume"].to_list()]

    # indicators on completed minute bars
    ema_9_series = rolling_ema(min_close, EMA_FAST)
    ema_21_series = rolling_ema(min_close, EMA_SLOW)
    sma_20_series = rolling_sma(min_close, SMA_FAST)
    sma_50_series = rolling_sma(min_close, SMA_SLOW)

    macd_fast = rolling_ema(min_close, MACD_FAST)
    macd_slow = rolling_ema(min_close, MACD_SLOW)
    macd_series: List[Optional[float]] = []
    for a, b in zip(macd_fast, macd_slow):
        if a is None or b is None:
            macd_series.append(None)
        else:
            macd_series.append(a - b)
    macd_signal_series = rolling_ema(macd_series, MACD_SIGNAL)
    macd_hist_series: List[Optional[float]] = []
    for m, s in zip(macd_series, macd_signal_series):
        if m is None or s is None:
            macd_hist_series.append(None)
        else:
            macd_hist_series.append(m - s)

    rsi_14_series = rolling_rsi_wilder(min_close, RSI_N)
    stoch_k_series, stoch_d_series = rolling_stoch(min_high, min_low, min_close, STOCH_N, STOCH_D_N)
    atr_14_series = rolling_atr(min_high, min_low, min_close, ATR_N)

    # session OBV on minute bars
    obv_session_series: List[Optional[float]] = [None] * len(min_close)
    obv_val = 0.0
    for i in range(len(min_close)):
        if min_close[i] is None:
            obv_session_series[i] = None
            continue
        if i == 0 or min_close[i - 1] is None:
            obv_session_series[i] = 0.0
            continue
        if min_close[i] > min_close[i - 1]:
            obv_val += min_vol[i]
        elif min_close[i] < min_close[i - 1]:
            obv_val -= min_vol[i]
        obv_session_series[i] = obv_val

    # -------------------------
    # Event processing
    # -------------------------
    for r in df_day_events.to_dicts():
        t0: datetime = r["ts"]
        t0_us = dt_to_us(t0)

        # -------------------------
        # latest trade <= event
        # -------------------------
        j_tr = bisect_right(tr_us, t0_us) - 1
        if j_tr < 0:
            add_null_v5_columns(r)
            rows_out_day.append(r)
            continue

        cur_px = tr_px[j_tr]

        # -------------------------
        # returns
        # -------------------------
        px_30s_ago = lagged_value_at_or_before(tr_us, tr_px, t0_us, RET_30S)
        px_5m_ago = lagged_value_at_or_before(tr_us, tr_px, t0_us, RET_5M)

        return_last_30s = None
        if px_30s_ago is not None and px_30s_ago > 0:
            return_last_30s = math.log(cur_px / px_30s_ago)

        return_last_5m = None
        if px_5m_ago is not None and px_5m_ago > 0:
            return_last_5m = math.log(cur_px / px_5m_ago)

        # -------------------------
        # realized vols from trade-time returns
        # -------------------------
        def realized_vol_window(window_s: int) -> Optional[float]:
            i0, i1 = window_bounds(tr_us, t0_us, window_s)
            pxs = tr_px[i0:i1]
            return realized_vol_from_prices(pxs)

        realized_volatility_30s = realized_vol_window(VOL_30S)
        realized_vol_1m = realized_vol_window(VOL_1M)
        realized_vol_5m = realized_vol_window(VOL_5M)

        vol_ratio = safe_div(realized_volatility_30s, realized_vol_5m)

        # -------------------------
        # tick rate
        # -------------------------
        i0_tick, i1_tick = window_bounds(tr_us, t0_us, TICK_RATE_WINDOW_S)
        tick_rate = (i1_tick - i0_tick) / TICK_RATE_WINDOW_S

        # -------------------------
        # latest completed 1-minute bar only
        # avoid intra-minute look-ahead
        # -------------------------
        current_minute_start_us = (t0_us // 60_000_000) * 60_000_000
        j_min = bisect_left(min_us, current_minute_start_us) - 1

        if j_min >= 0:
            ema_9 = ema_9_series[j_min]
            ema_21 = ema_21_series[j_min]
            ema_spread_9_21 = None if (ema_9 is None or ema_21 is None) else (ema_9 - ema_21)

            sma_20 = sma_20_series[j_min]
            sma_50 = sma_50_series[j_min]
            sma_spread_20_50 = None if (sma_20 is None or sma_50 is None) else (sma_20 - sma_50)

            macd = macd_series[j_min]
            macd_signal = macd_signal_series[j_min]
            macd_hist = macd_hist_series[j_min]

            rsi_14 = rsi_14_series[j_min]
            stoch_k_14 = stoch_k_series[j_min]
            stoch_d_3 = stoch_d_series[j_min]

            obv_session = obv_session_series[j_min]
            average_true_range = atr_14_series[j_min]

            sigma0 = r.get("sigma0")
            entry_price = r.get("entry_price")
            distance_from_ema_21 = None
            if ema_21 is not None and entry_price is not None:
                distance_from_ema_21 = safe_div(float(entry_price) - float(ema_21), sigma0)
        else:
            ema_9 = None
            ema_21 = None
            ema_spread_9_21 = None
            sma_20 = None
            sma_50 = None
            sma_spread_20_50 = None
            macd = None
            macd_signal = None
            macd_hist = None
            rsi_14 = None
            stoch_k_14 = None
            stoch_d_3 = None
            obv_session = None
            average_true_range = None
            distance_from_ema_21 = None

        r.update({
            # indicators
            "rsi_14": rsi_14,
            "stoch_k_14": stoch_k_14,
            "stoch_d_3": stoch_d_3,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "ema_spread_9_21": ema_spread_9_21,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_spread_20_50": sma_spread_20_50,
            "obv_session": obv_session,
            "distance_from_ema_21": distance_from_ema_21,

            # volatility / speed
            "return_last_30s": return_last_30s,
            "return_last_5m": return_last_5m,
            "realized_vol_1m": realized_vol_1m,
            "realized_vol_5m": realized_vol_5m,
            "realized_volatility_30s": realized_volatility_30s,
            "vol_ratio": vol_ratio,
            "tick_rate": tick_rate,
            "average_true_range": average_true_range,
        })
        rows_out_day.append(r)

    return rows_out_day


# -----------------------------
# Main
# -----------------------------
def main():
    if not BASE_EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing base events parquet: {BASE_EVENTS_PATH}")
    if not FULL_TICK_ROOT.exists():
        raise FileNotFoundError(f"Missing full tick root: {FULL_TICK_ROOT}")

    day_to_full = build_day_to_path(FULL_TICK_ROOT)
    print(f"Found {len(day_to_full):,} full tick days")

    df_events = pl.read_parquet(BASE_EVENTS_PATH).with_columns([
        pl.col("date_ymd").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime),
    ])

    required = {"event_id", "date_ymd", "ts"}
    missing = [c for c in required if c not in df_events.columns]
    if missing:
        raise ValueError(f"Base events parquet missing required columns: {missing}")

    event_days = sorted(df_events.select("date_ymd").unique().to_series().to_list())

    day_jobs = []
    for day_ymd in event_days:
        df_day_events = (
            df_events
            .filter(pl.col("date_ymd") == day_ymd)
            .sort("ts")
        )
        full_path = day_to_full.get(day_ymd)
        day_jobs.append((
            day_ymd,
            None if full_path is None else str(full_path),
            df_day_events,
        ))

    rows_out: List[Dict[str, Any]] = []
    max_workers = 7

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_event_day, day_ymd, full_path, df_day_events)
            for (day_ymd, full_path, df_day_events) in day_jobs
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            rows_day = fut.result()
            rows_out.extend(rows_day)

            if i % 25 == 0:
                print(f"...processed {i} event days")

    if not rows_out:
        print("No rows produced.")
        return

    df_out = pl.from_dicts(rows_out).sort(["date_ymd", "ts", "band_k", "side"])

    new_float_cols = [
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

    cast_exprs = []
    for c in new_float_cols:
        if c in df_out.columns:
            cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    if "ts" in df_out.columns:
        cast_exprs.append(pl.col("ts").cast(pl.Datetime, strict=False))

    df_out = df_out.with_columns(cast_exprs)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(OUT_FILE)

    print("\nDone.")
    print(f"Wrote: {OUT_FILE}")
    print(f"Rows: {df_out.height:,}  Cols: {len(df_out.columns)}")

    print("\nQuick sanity:")
    sanity_cols = [c for c in [
        "rsi_14",
        "macd",
        "ema_21",
        "obv_session",
        "distance_from_ema_21",
        "return_last_30s",
        "realized_vol_1m",
        "realized_vol_5m",
        "vol_ratio",
        "tick_rate",
        "average_true_range",
    ] if c in df_out.columns]

    exprs = [pl.len().alias("n")] + [
        pl.col(c).null_count().alias(f"{c}_nulls") for c in sanity_cols
    ]
    print(df_out.select(exprs))


if __name__ == "__main__":
    main()
# vwap_band_reversion_summary.py
# Adds conditional features (pre-touch RV, trade rate, session bucket, trend day, vwap-open distance, etc.)
# and prints side-by-side reversion summaries from touch_outcomes.parquet.

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, date, time
from bisect import bisect_left
from typing import Dict, List, Tuple, Optional

import polars as pl

# -----------------------------
# Paths (edit if needed)
# -----------------------------
OUTCOMES_PATH = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\touch_outcomes.parquet"
TRADES_ROOT   = r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_trades_parquet"

# -----------------------------
# Market constants
# -----------------------------
TICK = 0.25  # ES tick size in points

# Horizons you asked for (must exist in your parquet as mfe_pts_t{H} / end_ret_pts_t{H})
HORIZONS = [1, 5, 10, 15, 30, 60, 120, 180, 300, 600, 900]

# Thresholds for "bounce" (ticks)
THRESH_TICKS = [1, 2, 4, 8, 12, 16, 20, 40]  # 40 ticks = 10 points

# Pre-touch windows for vol/speed features (seconds)
RV_WIN_S = 60        # realized variance window (pre-touch)
RATE_WIN_S = 60      # trade rate window (pre-touch)
# VWAP slope + sigma dynamics windows (seconds)
VWAP_SLOPE_WIN_S = 300   # 5 min for VWAP slope
SIGMA_DDT_WIN_S  = 60    # compare last 60s vs the 60s before that

# Session time buckets (assumes RTH-like clock; change if your session differs)
SESSION_OPEN = time(9, 30)
SESSION_MID  = time(12, 0)
SESSION_CLOSE = time(16, 0)

# Cooldown bucketing (seconds)
COOLDOWN_BINS = [0, 30, 60, 120, 300, 900, 3600, 10**9]
COOLDOWN_LABELS = ["<30s", "30-60s", "1-2m", "2-5m", "5-15m", "15-60m", ">=60m"]

# Since-touch-any bucketing (seconds)
TOUCHANY_BINS = [0, 1, 5, 30, 120, 600, 3600, 10**9]
TOUCHANY_LABELS = ["<1s", "1-5s", "5-30s", "30-120s", "2-10m", "10-60m", ">=60m"]


# -----------------------------
# Helpers
# -----------------------------
def iter_day_trade_parquets(trades_root: str) -> List[Tuple[int, int, int, str]]:
    """
    Expected layout:
      TRADES_ROOT\year=YYYY\month=MM\day=DD\*.parquet
    Returns list of (year, month, day, parquet_path)
    """
    out: List[Tuple[int, int, int, str]] = []
    if not os.path.exists(trades_root):
        return out

    for root, _, files in os.walk(trades_root):
        for f in files:
            if not f.lower().endswith(".parquet"):
                continue
            p = os.path.join(root, f)

            # Try to parse year=YYYY\month=MM\day=DD from the path
            parts = p.replace("/", "\\").split("\\")
            y = m = d = None
            for part in parts:
                if part.startswith("year="):
                    y = int(part.split("=")[1])
                elif part.startswith("month="):
                    m = int(part.split("=")[1])
                elif part.startswith("day="):
                    d = int(part.split("=")[1])
            if y and m and d:
                out.append((y, m, d, p))

    out.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return out


def dt_to_us(dt: datetime) -> int:
    # naive datetime -> microseconds since epoch
    return int(dt.timestamp() * 1_000_000)


def time_bucket(ts: datetime) -> str:
    t = ts.time()
    if t < SESSION_OPEN:
        return "pre"
    if t < SESSION_MID:
        return "morning"
    if t <= SESSION_CLOSE:
        return "afternoon"
    return "post"


def bucket_value(x: Optional[float], bins: List[float], labels: List[str]) -> Optional[str]:
    if x is None:
        return None
    for i in range(len(labels)):
        if bins[i] <= x < bins[i + 1]:
            return labels[i]
    return labels[-1]


def realized_variance_points(price: List[float], start_i: int, end_i: int) -> float:
    """
    RV = sum (dp)^2 over [start_i, end_i) in points^2
    """
    if end_i - start_i <= 1:
        return 0.0
    s = 0.0
    prev = price[start_i]
    for j in range(start_i + 1, end_i):
        pj = price[j]
        dp = pj - prev
        s += dp * dp
        prev = pj
    return s

def prefix_sums(xs: List[float]) -> List[float]:
    out = [0.0]
    s = 0.0
    for x in xs:
        s += float(x)
        out.append(s)
    return out

def range_sum(pref: List[float], a: int, b: int) -> float:
    # sum over [a, b)
    a = max(0, a)
    b = max(a, b)
    b = min(b, len(pref) - 1)
    return pref[b] - pref[a]

def session_vwap_at(
    tr_us: List[int],
    pv_pref: List[float],
    v_pref: List[float],
    sess_open_us: int,
    t_us: int,
) -> Optional[float]:
    """
    Session VWAP from session open up to time t (exclusive).
    Uses prefix sums of price*vol and vol.
    """
    i_open = bisect_left(tr_us, sess_open_us)
    i_t = bisect_left(tr_us, t_us)
    vol = range_sum(v_pref, i_open, i_t)
    if vol <= 0:
        return None
    pv = range_sum(pv_pref, i_open, i_t)
    return pv / vol

# -----------------------------
# 1) Load outcomes
# -----------------------------
df_events = pl.read_parquet(OUTCOMES_PATH)
print("date_ymd dtype:", df_events.schema.get("date_ymd"))
print("sample date_ymd values:", df_events.select("date_ymd").head(5))
print("has cooldown_seconds:", "cooldown_seconds" in df_events.columns)
print("has since_touch_any_seconds:", "since_touch_any_seconds" in df_events.columns)

required = {"date_ymd", "ts", "band_k", "side", "vwap0"}
missing = [c for c in required if c not in df_events.columns]
if missing:
    raise ValueError(f"touch_outcomes parquet missing required cols: {missing}")

# Make sure ts is Datetime (Polars) and also have a Python datetime for per-day processing
df_events = df_events.with_columns([
    pl.col("date_ymd").cast(pl.Utf8),
    pl.col("ts").cast(pl.Datetime),
])

# Build map: date_ymd -> trade parquet path
day_files = iter_day_trade_parquets(TRADES_ROOT)
day_to_path: Dict[str, str] = {
    f"{y:04d}-{m:02d}-{d:02d}": p for (y, m, d, p) in day_files
}
needed_days = set(df_events.select("date_ymd").unique().to_series().to_list())
found_days = set(day_to_path.keys())

print(f"[coverage] outcomes days: {len(needed_days)}")
print(f"[coverage] trade days found: {len(found_days)}")
print(f"[coverage] missing days: {len(needed_days - found_days)}")
print(f"[coverage] example missing: {sorted(list(needed_days - found_days))[:10]}")
print("example day_to_path keys:", list(day_to_path.keys())[:5])
# -----------------------------
# 2) Enrich events with per-day trade-derived features
# -----------------------------
rows_out: List[dict] = []
skipped_days_no_trades: List[str] = []

# We’ll process day-by-day to avoid loading all trades into memory
for day_ymd, df_day in df_events.partition_by("date_ymd", as_dict=True).items():
    # partition_by(as_dict=True) can give tuple keys depending on Polars version
    if isinstance(day_ymd, tuple):
        day_ymd = day_ymd[0]
    trade_path = day_to_path.get(day_ymd)
    if trade_path is None or (not os.path.exists(trade_path)):
        skipped_days_no_trades.append(day_ymd)
        # still emit rows with None features
        for r in df_day.to_dicts():
            r.update({
                "rv_pre_pts2": None,
                "trade_rate_pre_per_s": None,
                "day_open": None,
                "day_close": None,
                "day_trend": None,
                "vwap0_minus_open_pts": None,
                "session_bucket": None,
                "seconds_since_open": None,
                "cooldown_bucket": None,
                "since_touch_any_bucket": None,
            })
            rows_out.append(r)
        continue

    df_tr = pl.read_parquet(trade_path).select(["ts", "price", "volume"]).sort("ts")

    if df_tr.height == 0:
        skipped_days_no_trades.append(day_ymd)
        for r in df_day.to_dicts():
            r.update({
                "rv_pre_pts2": None,
                "trade_rate_pre_per_s": None,
                "day_open": None,
                "day_close": None,
                "day_trend": None,
                "vwap0_minus_open_pts": None,
                "session_bucket": None,
                "seconds_since_open": None,
                "cooldown_bucket": None,
                "since_touch_any_bucket": None,
            })
            rows_out.append(r)
        continue

    # trade arrays
    tr_ts = df_tr["ts"].to_list()
    tr_us = [dt_to_us(t) for t in tr_ts]
    tr_px = [float(x) for x in df_tr["price"].to_list()]
    tr_vol = [float(x) for x in df_tr["volume"].to_list()]
    tr_pv  = [p * v for p, v in zip(tr_px, tr_vol)]

    v_pref  = prefix_sums(tr_vol)
    pv_pref = prefix_sums(tr_pv)

    # daily open/close from this parquet
    day_open = tr_px[0]
    day_close = tr_px[-1]
    day_trend = "up" if day_close > day_open else ("down" if day_close < day_open else "flat")

    # event arrays (for speed)
    ev_dicts = df_day.to_dicts()

        # compute per-event features
    for r in ev_dicts:
        t0: datetime = r["ts"]
        t0_us = dt_to_us(t0)

        # indices for pre-touch windows (strictly before t0)
        i0 = bisect_left(tr_us, t0_us)

        # window starts (in us)
        rv_start_us   = t0_us - int(RV_WIN_S * 1_000_000)
        rate_start_us = t0_us - int(RATE_WIN_S * 1_000_000)

        i_rv0   = bisect_left(tr_us, rv_start_us)
        i_rate0 = bisect_left(tr_us, rate_start_us)

        # realized variance over last RV_WIN_S seconds
        rv = realized_variance_points(tr_px, i_rv0, i0)

        # trade arrival rate over last RATE_WIN_S seconds
        n_rate = max(0, i0 - i_rate0)
        trade_rate = n_rate / float(RATE_WIN_S) if RATE_WIN_S > 0 else None

        # sigma level (points / sqrt(sec)) from RV window
        sigma_lvl = (rv / float(RV_WIN_S)) ** 0.5 if RV_WIN_S > 0 else None

        # dσ/dt: compare current 60s window vs prior 60s window
        t_prev_end_us = t0_us - int(SIGMA_DDT_WIN_S * 1_000_000)
        t_prev_sta_us = t0_us - int(2 * SIGMA_DDT_WIN_S * 1_000_000)

        i_prev_end = bisect_left(tr_us, t_prev_end_us)
        i_prev_sta = bisect_left(tr_us, t_prev_sta_us)

        rv_prev = realized_variance_points(tr_px, i_prev_sta, i_prev_end)
        sigma_prev = (rv_prev / float(SIGMA_DDT_WIN_S)) ** 0.5 if SIGMA_DDT_WIN_S > 0 else None

        dsigma_dt = None
        if sigma_lvl is not None and sigma_prev is not None and SIGMA_DDT_WIN_S > 0:
            dsigma_dt = (sigma_lvl - sigma_prev) / float(SIGMA_DDT_WIN_S)

        # session features
        sess_bucket = time_bucket(t0)
        sess_open_dt = datetime.combine(t0.date(), SESSION_OPEN)
        seconds_since_open = (t0 - sess_open_dt).total_seconds()
        sess_open_us = dt_to_us(sess_open_dt)

        # session VWAP at t0 and at (t0 - VWAP_SLOPE_WIN_S)
        t1_us = t0_us - int(VWAP_SLOPE_WIN_S * 1_000_000)

        vwap_t0 = session_vwap_at(tr_us, pv_pref, v_pref, sess_open_us, t0_us)
        vwap_t1 = session_vwap_at(tr_us, pv_pref, v_pref, sess_open_us, t1_us)

        vwap_slope_pts_per_s = None
        abs_vwap_slope_pts_per_s = None
        if vwap_t0 is not None and vwap_t1 is not None and VWAP_SLOPE_WIN_S > 0:
            vwap_slope_pts_per_s = (vwap_t0 - vwap_t1) / float(VWAP_SLOPE_WIN_S)
            abs_vwap_slope_pts_per_s = abs(vwap_slope_pts_per_s)

        # distance VWAP from daily open at touch (signed)
        vwap0 = r.get("vwap0", None)
        vwap0_minus_open = (float(vwap0) - float(day_open)) if vwap0 is not None else None

        # cooldown / since_touch_any bucketing (already in outcomes)
        cd = r.get("cooldown_seconds", None)
        st = r.get("since_touch_any_seconds", None)
        cd_bucket = bucket_value(cd, COOLDOWN_BINS, COOLDOWN_LABELS) if cd is not None else None
        st_bucket = bucket_value(st, TOUCHANY_BINS, TOUCHANY_LABELS) if st is not None else None

        # ALWAYS update + append (never gate on vwap_t0/vwap_t1 existing)
        r.update({
            "rv_pre_pts2": rv,
            "trade_rate_pre_per_s": trade_rate,
            "day_open": day_open,
            "day_close": day_close,
            "day_trend": day_trend,
            "vwap0_minus_open_pts": vwap0_minus_open,
            "session_bucket": sess_bucket,
            "seconds_since_open": seconds_since_open,
            "cooldown_bucket": cd_bucket,
            "since_touch_any_bucket": st_bucket,
            "sigma_lvl": sigma_lvl,
            "dsigma_dt": dsigma_dt,
            "vwap_slope_pts_per_s": vwap_slope_pts_per_s,
            "abs_vwap_slope_pts_per_s": abs_vwap_slope_pts_per_s,
        })
        rows_out.append(r)

if sigma_lvl is not None and sigma_prev is not None and SIGMA_DDT_WIN_S > 0:
    dsigma_dt = (sigma_lvl - sigma_prev) / float(SIGMA_DDT_WIN_S)

    # trade arrival rate over last RATE_WIN_S seconds
    n_rate = max(0, i0 - i_rate0)
    trade_rate = n_rate / float(RATE_WIN_S)

    # session features
    sess_bucket = time_bucket(t0)

    # seconds since session open (same calendar day)
    # (If your trades include Globex and you want different anchors, adjust this.)
    sess_open_dt = datetime.combine(t0.date(), SESSION_OPEN)
    seconds_since_open = (t0 - sess_open_dt).total_seconds()

    sess_open_us = dt_to_us(sess_open_dt)

    # session VWAP at t0 and at (t0 - VWAP_SLOPE_WIN_S)
    t1_us = t0_us - int(VWAP_SLOPE_WIN_S * 1_000_000)

    vwap_t0 = session_vwap_at(tr_us, pv_pref, v_pref, sess_open_us, t0_us)
    vwap_t1 = session_vwap_at(tr_us, pv_pref, v_pref, sess_open_us, t1_us)

    vwap_slope_pts_per_s = None
    abs_vwap_slope_pts_per_s = None

if vwap_t0 is not None and vwap_t1 is not None and VWAP_SLOPE_WIN_S > 0:
    vwap_slope_pts_per_s = (vwap_t0 - vwap_t1) / float(VWAP_SLOPE_WIN_S)
    abs_vwap_slope_pts_per_s = abs(vwap_slope_pts_per_s)

    # distance VWAP from daily open at touch (signed)
    vwap0 = r.get("vwap0", None)
    vwap0_minus_open = (float(vwap0) - float(day_open)) if vwap0 is not None else None

    # cooldown / since_touch_any bucketing (already in outcomes)
    cd = r.get("cooldown_seconds", None)
    st = r.get("since_touch_any_seconds", None)
    cd_bucket = bucket_value(cd, COOLDOWN_BINS, COOLDOWN_LABELS) if cd is not None else None
    st_bucket = bucket_value(st, TOUCHANY_BINS, TOUCHANY_LABELS) if st is not None else None

    r.update({
        "rv_pre_pts2": rv,
        "trade_rate_pre_per_s": trade_rate,
        "day_open": day_open,
        "day_close": day_close,
        "day_trend": day_trend,
        "vwap0_minus_open_pts": vwap0_minus_open,
        "session_bucket": sess_bucket,
        "seconds_since_open": seconds_since_open,
        "cooldown_bucket": cd_bucket,
        "since_touch_any_bucket": st_bucket,
        "sigma_lvl": sigma_lvl,
        "dsigma_dt": dsigma_dt,
        "vwap_slope_pts_per_s": vwap_slope_pts_per_s,
        "abs_vwap_slope_pts_per_s": abs_vwap_slope_pts_per_s,
    })

    rows_out.append(r)

df = pl.from_dicts(rows_out)
print("events in outcomes:", df_events.height)
print("rows_out:", len(rows_out))
print(df.select(pl.len()).item())

print(df.select([
    pl.col("rv_pre_pts2").null_count().alias("rv_nulls"),
    pl.col("trade_rate_pre_per_s").null_count().alias("rate_nulls"),
    pl.count().alias("n"),
    pl.col("rv_pre_pts2").drop_nulls().min().alias("rv_min"),
    pl.col("rv_pre_pts2").drop_nulls().max().alias("rv_max"),
    pl.col("trade_rate_pre_per_s").drop_nulls().min().alias("rate_min"),
    pl.col("trade_rate_pre_per_s").drop_nulls().max().alias("rate_max"),
]))

if skipped_days_no_trades:
    print(f"\n[warn] Missing trades parquet for {len(skipped_days_no_trades)} days (features will be null).")
    print("       Example:", skipped_days_no_trades[:10])

# -----------------------------
# 3) Build regimes (quantiles) – robust
# -----------------------------
df = df.with_columns([
    pl.col("rv_pre_pts2").cast(pl.Float64),
    pl.col("trade_rate_pre_per_s").cast(pl.Float64),
])

# Always create regime columns first (so later group_by never fails)
df = df.with_columns([
    pl.lit(None).cast(pl.Utf8).alias("vol_regime"),
    pl.lit(None).cast(pl.Utf8).alias("speed_regime"),
])

# Compute quantiles on non-null subset (safe even if many nulls)
rv_nonnull = df.select(pl.col("rv_pre_pts2").drop_nulls())
rt_nonnull = df.select(pl.col("trade_rate_pre_per_s").drop_nulls())

rv_q33 = rv_nonnull.select(pl.col("rv_pre_pts2").quantile(0.33)).item() if rv_nonnull.height > 0 else None
rv_q66 = rv_nonnull.select(pl.col("rv_pre_pts2").quantile(0.66)).item() if rv_nonnull.height > 0 else None

rt_q33 = rt_nonnull.select(pl.col("trade_rate_pre_per_s").quantile(0.33)).item() if rt_nonnull.height > 0 else None
rt_q66 = rt_nonnull.select(pl.col("trade_rate_pre_per_s").quantile(0.66)).item() if rt_nonnull.height > 0 else None

# Now fill regimes if we actually have thresholds
if rv_q33 is not None and rv_q66 is not None:
    df = df.with_columns(
        pl.when(pl.col("rv_pre_pts2").is_null()).then(pl.lit(None))
        .when(pl.col("rv_pre_pts2") < rv_q33).then(pl.lit("low"))
        .when(pl.col("rv_pre_pts2") < rv_q66).then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("vol_regime")
    )

if rt_q33 is not None and rt_q66 is not None:
    df = df.with_columns(
        pl.when(pl.col("trade_rate_pre_per_s").is_null()).then(pl.lit(None))
        .when(pl.col("trade_rate_pre_per_s") < rt_q33).then(pl.lit("low"))
        .when(pl.col("trade_rate_pre_per_s") < rt_q66).then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("speed_regime")
    )

print(f"[regimes] rv_q33={rv_q33} rv_q66={rv_q66} | rt_q33={rt_q33} rt_q66={rt_q66}")
print(df.select([
    pl.col("vol_regime").is_null().mean().alias("vol_regime_null_frac"),
    pl.col("speed_regime").is_null().mean().alias("speed_regime_null_frac"),
]))

# --- VWAP slope regimes (abs slope quantiles) ---
df = df.with_columns([
    pl.lit(None).cast(pl.Utf8).alias("slope_regime"),
])

sl_nonnull = df.select(pl.col("abs_vwap_slope_pts_per_s").drop_nulls())
sl_q33 = sl_nonnull.select(pl.col("abs_vwap_slope_pts_per_s").quantile(0.33)).item() if sl_nonnull.height > 0 else None
sl_q66 = sl_nonnull.select(pl.col("abs_vwap_slope_pts_per_s").quantile(0.66)).item() if sl_nonnull.height > 0 else None

if sl_q33 is not None and sl_q66 is not None:
    df = df.with_columns(
        pl.when(pl.col("abs_vwap_slope_pts_per_s").is_null()).then(pl.lit(None))
        .when(pl.col("abs_vwap_slope_pts_per_s") < sl_q33).then(pl.lit("low"))
        .when(pl.col("abs_vwap_slope_pts_per_s") < sl_q66).then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("slope_regime")
    )

# Cross bucket: "vol=high | slope=low" etc
df = df.with_columns(
    (pl.lit("vol=") + pl.col("vol_regime").fill_null("null") +
     pl.lit(" | slope=") + pl.col("slope_regime").fill_null("null")).alias("vol_slope_bucket")
)

print(f"[slope regimes] sl_q33={sl_q33} sl_q66={sl_q66}")


# -----------------------------
# 4) Summary builder
# -----------------------------
def summarize_for_metric(
    df_in: pl.DataFrame,
    metric_prefix: str,  # "mfe_pts_t" or "end_ret_pts_t"
    group_cols: List[str],
    horizons: List[int],
    thresh_ticks: List[int],
) -> pl.DataFrame:
    frames: List[pl.DataFrame] = []

    for H in horizons:
        col = f"{metric_prefix}{H}"
        if col not in df_in.columns:
            print(f"[skip] {col} not in parquet")
            continue

        exprs = [
            pl.col(col).is_not_null().sum().alias("n_nonnull"),
            pl.col(col).mean().alias("mean_pts"),
            pl.col(col).median().alias("median_pts"),
            (pl.col(col) / TICK).mean().alias("mean_ticks"),
            (pl.col(col) / TICK).median().alias("median_ticks"),
        ]

        for tt in thresh_ticks:
            thr_pts = tt * TICK
            exprs.append((pl.col(col) >= thr_pts).mean().alias(f"p_ge_{tt}tick"))
            exprs.append(
                pl.when(pl.col(col) >= thr_pts)
                  .then(pl.col(col))
                  .otherwise(0.0)
                  .mean()
                  .alias(f"ev_ge_{tt}tick_pts")
            )

        out = (
            df_in
            .group_by(group_cols)
            .agg(exprs)
            .with_columns(pl.lit(H).alias("horizon_s"))
        )
        frames.append(out)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames).sort(group_cols + ["horizon_s"])


def best_per_horizon(
    summary: pl.DataFrame,
    best_ticks: int = 4,
    by: str = "ev",  # "ev" or "p"
) -> pl.DataFrame:
    p_col = f"p_ge_{best_ticks}tick"
    ev_col = f"ev_ge_{best_ticks}tick_pts"

    if by == "p":
        sort_cols = ["horizon_s", p_col, ev_col, "mean_pts"]
        desc = [False, True, True, True]
    else:
        sort_cols = ["horizon_s", ev_col, p_col, "mean_pts"]
        desc = [False, True, True, True]

    return (
        summary
        .sort(sort_cols, descending=desc)
        .group_by("horizon_s")
        .head(1)
    )


# -----------------------------
# 5) Baseline summary (your original view, but richer)
# -----------------------------
BASE_GROUP = ["band_k", "side"]
mfe_base = summarize_for_metric(df, "mfe_pts_t", BASE_GROUP, HORIZONS, THRESH_TICKS)

print("\n=== BASELINE (MFE) grouped by band_k/side ===")
print(mfe_base.select([c for c in [
    "horizon_s","band_k","side","n_nonnull","mean_pts","mean_ticks",
    "p_ge_4tick","ev_ge_4tick_pts","p_ge_8tick","ev_ge_8tick_pts","p_ge_40tick","ev_ge_40tick_pts"
] if c in mfe_base.columns]))

print("\nBest by probability (p_ge_4tick) [MFE]:")
print(best_per_horizon(mfe_base, best_ticks=4, by="p").select([
    "horizon_s","band_k","side","n_nonnull","p_ge_4tick","ev_ge_4tick_pts","mean_pts"
]))

print("\nBest by EV (ev_ge_4tick_pts) [MFE]:")
print(best_per_horizon(mfe_base, best_ticks=4, by="ev").select([
    "horizon_s","band_k","side","n_nonnull","p_ge_4tick","ev_ge_4tick_pts","mean_pts"
]))

# Immediate-trade EV (end_ret) baseline
endret_base = summarize_for_metric(df, "end_ret_pts_t", BASE_GROUP, HORIZONS, THRESH_TICKS)
if endret_base.height > 0:
    print("\n=== BASELINE (END-RET) grouped by band_k/side ===")
    print(endret_base.select([c for c in [
        "horizon_s","band_k","side","n_nonnull","mean_pts","mean_ticks",
        "p_ge_4tick","ev_ge_4tick_pts","p_ge_8tick","ev_ge_8tick_pts","p_ge_40tick","ev_ge_40tick_pts"
    ] if c in endret_base.columns]))

    print("\nBest by probability (p_ge_4tick) [END-RET]:")
    print(best_per_horizon(endret_base, best_ticks=4, by="p").select([
        "horizon_s","band_k","side","n_nonnull","p_ge_4tick","ev_ge_4tick_pts","mean_pts"
    ]))

    print("\nBest by EV (ev_ge_4tick_pts) [END-RET]:")
    print(best_per_horizon(endret_base, best_ticks=4, by="ev").select([
        "horizon_s","band_k","side","n_nonnull","p_ge_4tick","ev_ge_4tick_pts","mean_pts"
    ]))
else:
    print("\n[info] end_ret_pts_tH columns not found (skip END-RET summaries).")

# -----------------------------
# 6) Conditional summaries (separate, not exploding into one giant table)
# -----------------------------
def print_conditional(title: str, group_cols: List[str]):
    s = summarize_for_metric(df, "mfe_pts_t", group_cols, HORIZONS, THRESH_TICKS)
    if s.height == 0:
        print(f"\n=== {title} ===\n(empty)")
        return

    print(f"\n=== {title} (MFE) ===")
    # show best per horizon within each condition level if you want
    # Here: print the full table but trimmed.
    keep = [c for c in [
        "horizon_s", *group_cols, "n_nonnull", "p_ge_4tick", "ev_ge_4tick_pts", "mean_pts"
    ] if c in s.columns]
    print(s.select(keep))

# Vol regime
print_conditional("By vol_regime + band_k/side", ["vol_regime","band_k","side"])
# Speed regime
print_conditional("By speed_regime + band_k/side", ["speed_regime","band_k","side"])
# Session bucket
print_conditional("By session_bucket + band_k/side", ["session_bucket","band_k","side"])
# Day trend
print_conditional("By day_trend + band_k/side", ["day_trend","band_k","side"])
# Cooldown bucket
print_conditional("By cooldown_bucket + band_k/side", ["cooldown_bucket","band_k","side"])
# Since-touch-any bucket
print_conditional("By since_touch_any_bucket + band_k/side", ["since_touch_any_bucket","band_k","side"])
print_conditional("By vol_regime + slope_regime + band_k/side", ["vol_regime","slope_regime","band_k","side"])

# -----------------------------
# 7) Quick sanity peek of new feature distributions
# -----------------------------
print("\n=== Feature sanity ===")
print(df.select([
    pl.col("rv_pre_pts2").drop_nulls().quantile(0.50).alias("rv_med"),
    pl.col("trade_rate_pre_per_s").drop_nulls().quantile(0.50).alias("rate_med"),
    pl.col("rv_pre_pts2").drop_nulls().quantile(0.90).alias("rv_q90"),
    pl.col("trade_rate_pre_per_s").drop_nulls().quantile(0.90).alias("rate_q90"),
]))
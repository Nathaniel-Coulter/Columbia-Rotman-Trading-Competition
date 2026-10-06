# build_event_features_v1.py
# Builds V1 event-level features for SHAP / LightGBM from:
#   1) touch_outcomes.parquet
#   2) per-day trades parquet
#
# It preserves the original event rows and adds:
# - current summary-style features:
#     rv_pre_pts2, trade_rate_pre_per_s, sigma_lvl, dsigma_dt,
#     vwap_slope_pts_per_s, abs_vwap_slope_pts_per_s,
#     session_bucket, seconds_since_open, day_open, day_close, day_trend,
#     vwap0_minus_open_pts, cooldown_bucket, since_touch_any_bucket
# - V1 structural features:
#     vwap_open
#     distance_from_session_vwap_anchor
#     z_anchor_vwap
#     vwap_drift
#     session_high_running_t0
#     session_low_running_t0
#     range_position
#     dist_day_high
#     dist_day_low
#     midrange
#     midrange_position
#     opening_range_high
#     opening_range_low
#     opening_range_position
#     prior_day_rth_high
#     prior_day_rth_low
#     distance_from_prior_day_high
#     distance_from_prior_day_low
#     overnight_high
#     overnight_low
#     distance_from_overnight_high
#     distance_from_overnight_low
#
# Notes:
# - "day" here means RTH session day, not full 24h future-known day.
# - running session high/low are computed ONLY up to event time (no leakage).
# - opening range = first 15 minutes of RTH, and opening_range_position is null before 09:45.
# - prior-day high/low use PRIOR RTH ONLY.
# - overnight high/low use:
#       previous calendar day's post-RTH prints (t > 16:00)
#       + current calendar day's pre-RTH prints (t < 09:30)

from __future__ import annotations

import os
from bisect import bisect_left
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import polars as pl

# -----------------------------
# Paths
# -----------------------------
OUTCOMES_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\touch_outcomes.parquet"
)

TRADES_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_trades_parquet"
)

OUT_FILE = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v1.parquet"
)

# -----------------------------
# Constants
# -----------------------------
TICK = 0.25

RV_WIN_S = 60
RATE_WIN_S = 60
VWAP_SLOPE_WIN_S = 300
SIGMA_DDT_WIN_S = 60

RTH_OPEN = time(14, 30)
RTH_CLOSE = time(21, 0)
OPENING_RANGE_MINUTES = 15
OPENING_RANGE_END = time(14, 45)

COOLDOWN_BINS = [0, 30, 60, 120, 300, 900, 3600, 10**9]
COOLDOWN_LABELS = ["<30s", "30-60s", "1-2m", "2-5m", "5-15m", "15-60m", ">=60m"]

TOUCHANY_BINS = [0, 1, 5, 30, 120, 600, 3600, 10**9]
TOUCHANY_LABELS = ["<1s", "1-5s", "5-30s", "30-120s", "2-10m", "10-60m", ">=60m"]


# -----------------------------
# Helpers
# -----------------------------
def iter_day_trade_parquets(trades_root: Path):
    for ydir in sorted(trades_root.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            for ddir in sorted(mdir.glob("day=*")):
                p = ddir / "part-000.parquet"
                if not p.exists():
                    continue
                year = int(ydir.name.split("=")[1])
                month = int(mdir.name.split("=")[1])
                day = int(ddir.name.split("=")[1])
                yield year, month, day, p


def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def time_bucket(ts: datetime) -> str:
    t = ts.time()
    if t < RTH_OPEN:
        return "pre"
    if t < time(12, 0):
        return "morning"
    if t <= RTH_CLOSE:
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
    i_open = bisect_left(tr_us, sess_open_us)
    i_t = bisect_left(tr_us, t_us)
    vol = range_sum(v_pref, i_open, i_t)
    if vol <= 0:
        return None
    pv = range_sum(pv_pref, i_open, i_t)
    return pv / vol


def prefix_max(xs: List[float]) -> List[float]:
    out: List[float] = []
    cur = float("-inf")
    for x in xs:
        if x > cur:
            cur = x
        out.append(cur)
    return out


def prefix_min(xs: List[float]) -> List[float]:
    out: List[float] = []
    cur = float("inf")
    for x in xs:
        if x < cur:
            cur = x
        out.append(cur)
    return out


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def build_day_to_path(trades_root: Path) -> Dict[str, Path]:
    return {
        f"{y:04d}-{m:02d}-{d:02d}": p
        for (y, m, d, p) in iter_day_trade_parquets(trades_root)
    }


def load_trade_day(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .select(["ts", "price", "volume"])
        .sort("ts")
    )


def extract_rth_arrays(df_tr: pl.DataFrame):
    rows = df_tr.to_dicts()
    ts = [r["ts"] for r in rows if RTH_OPEN <= r["ts"].time() <= RTH_CLOSE]
    px = [float(r["price"]) for r in rows if RTH_OPEN <= r["ts"].time() <= RTH_CLOSE]
    vol = [float(r["volume"]) for r in rows if RTH_OPEN <= r["ts"].time() <= RTH_CLOSE]
    ts_us = [dt_to_us(x) for x in ts]
    return ts, ts_us, px, vol


def extract_pre_rth_arrays(df_tr: pl.DataFrame):
    rows = df_tr.to_dicts()
    ts = [r["ts"] for r in rows if r["ts"].time() < RTH_OPEN]
    px = [float(r["price"]) for r in rows if r["ts"].time() < RTH_OPEN]
    return ts, px


def extract_post_rth_arrays(df_tr: pl.DataFrame):
    rows = df_tr.to_dicts()
    ts = [r["ts"] for r in rows if r["ts"].time() > RTH_CLOSE]
    px = [float(r["price"]) for r in rows if r["ts"].time() > RTH_CLOSE]
    return ts, px


def rth_high_low(df_tr: pl.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    rows = [r for r in df_tr.to_dicts() if RTH_OPEN <= r["ts"].time() <= RTH_CLOSE]
    if not rows:
        return None, None
    px = [float(r["price"]) for r in rows]
    return max(px), min(px)


# -----------------------------
# Main
# -----------------------------
def main():
    if not OUTCOMES_PATH.exists():
        raise FileNotFoundError(f"Missing outcomes parquet: {OUTCOMES_PATH}")

    day_to_path = build_day_to_path(TRADES_ROOT)
    print(f"Found {len(day_to_path):,} day partitions under {TRADES_ROOT}")

    df_events = pl.read_parquet(OUTCOMES_PATH).with_columns([
        pl.col("date_ymd").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime),
    ])

    required = {"event_id", "date_ymd", "ts", "entry_price", "vwap0", "sigma0"}
    missing = [c for c in required if c not in df_events.columns]
    if missing:
        raise ValueError(f"touch_outcomes parquet missing required columns: {missing}")

    rows_out: List[Dict[str, Any]] = []
    missing_trade_days: List[str] = []

    event_days = sorted(df_events.select("date_ymd").unique().to_series().to_list())

    for idx_day, day_ymd in enumerate(event_days, start=1):
        trade_path = day_to_path.get(day_ymd)
        if trade_path is None or not trade_path.exists():
            missing_trade_days.append(day_ymd)
            for r in df_events.filter(pl.col("date_ymd") == day_ymd).to_dicts():
                r.update({
                    "day_open": None,
                    "day_close": None,
                    "day_trend": None,
                    "session_bucket": None,
                    "seconds_since_open": None,
                    "cooldown_bucket": bucket_value(r.get("cooldown_seconds"), COOLDOWN_BINS, COOLDOWN_LABELS),
                    "since_touch_any_bucket": bucket_value(r.get("since_touch_any_seconds"), TOUCHANY_BINS, TOUCHANY_LABELS),
                    "rv_pre_pts2": None,
                    "trade_rate_pre_per_s": None,
                    "sigma_lvl": None,
                    "dsigma_dt": None,
                    "vwap_slope_pts_per_s": None,
                    "abs_vwap_slope_pts_per_s": None,
                    "vwap_open": None,
                    "distance_from_session_vwap_anchor": None,
                    "z_anchor_vwap": None,
                    "vwap_drift": None,
                    "vwap0_minus_open_pts": None,
                    "session_high_running_t0": None,
                    "session_low_running_t0": None,
                    "range_position": None,
                    "dist_day_high": None,
                    "dist_day_low": None,
                    "midrange": None,
                    "midrange_position": None,
                    "distance_from_session_midrange": None,
                    "z_range": None,
                    "opening_range_high": None,
                    "opening_range_low": None,
                    "opening_range_position": None,
                    "prior_day_rth_high": None,
                    "prior_day_rth_low": None,
                    "distance_from_prior_day_high": None,
                    "distance_from_prior_day_low": None,
                    "overnight_high": None,
                    "overnight_low": None,
                    "distance_from_overnight_high": None,
                    "distance_from_overnight_low": None,
                })
                rows_out.append(r)
            continue

        df_tr = load_trade_day(trade_path)
        df_day_events = df_events.filter(pl.col("date_ymd") == day_ymd)

        # previous calendar trade parquet if available
        prev_trade_path = None
        prev_idx = idx_day - 2
        prev_day_ymd = event_days[prev_idx] if prev_idx >= 0 else None
        if prev_day_ymd is not None:
            prev_trade_path = day_to_path.get(prev_day_ymd)

        df_prev_tr = load_trade_day(prev_trade_path) if prev_trade_path and prev_trade_path.exists() else None

        # current-day arrays
        tr_ts = df_tr["ts"].to_list()
        tr_us = [dt_to_us(x) for x in tr_ts]
        tr_px = [float(x) for x in df_tr["price"].to_list()]
        tr_vol = [float(x) for x in df_tr["volume"].to_list()]
        tr_pv = [p * v for p, v in zip(tr_px, tr_vol)]
        v_pref = prefix_sums(tr_vol)
        pv_pref = prefix_sums(tr_pv)

        # RTH arrays
        rth_ts, rth_us, rth_px, rth_vol = extract_rth_arrays(df_tr)
        if len(rth_ts) == 0:
            # no RTH prints for this day
            for r in df_day_events.to_dicts():
                r.update({
                    "day_open": None,
                    "day_close": None,
                    "day_trend": None,
                    "session_bucket": time_bucket(r["ts"]),
                    "seconds_since_open": (r["ts"] - datetime.combine(r["ts"].date(), RTH_OPEN)).total_seconds(),
                    "cooldown_bucket": bucket_value(r.get("cooldown_seconds"), COOLDOWN_BINS, COOLDOWN_LABELS),
                    "since_touch_any_bucket": bucket_value(r.get("since_touch_any_seconds"), TOUCHANY_BINS, TOUCHANY_LABELS),
                    "rv_pre_pts2": None,
                    "trade_rate_pre_per_s": None,
                    "sigma_lvl": None,
                    "dsigma_dt": None,
                    "vwap_slope_pts_per_s": None,
                    "abs_vwap_slope_pts_per_s": None,
                    "vwap_open": None,
                    "distance_from_session_vwap_anchor": None,
                    "z_anchor_vwap": None,
                    "vwap_drift": None,
                    "vwap0_minus_open_pts": None,
                    "session_high_running_t0": None,
                    "session_low_running_t0": None,
                    "range_position": None,
                    "dist_day_high": None,
                    "dist_day_low": None,
                    "midrange": None,
                    "midrange_position": None,
                    "opening_range_high": None,
                    "opening_range_low": None,
                    "opening_range_position": None,
                    "prior_day_rth_high": None,
                    "prior_day_rth_low": None,
                    "distance_from_prior_day_high": None,
                    "distance_from_prior_day_low": None,
                    "overnight_high": None,
                    "overnight_low": None,
                    "distance_from_overnight_high": None,
                    "distance_from_overnight_low": None,
                })
                rows_out.append(r)
            continue

        rth_high_pref = prefix_max(rth_px)
        rth_low_pref = prefix_min(rth_px)

        day_open = rth_px[0]
        day_close = rth_px[-1]
        day_trend = "up" if day_close > day_open else ("down" if day_close < day_open else "flat")
        vwap_open = day_open  # session-anchored VWAP at open

        # opening range
        or_end_dt = datetime.combine(rth_ts[0].date(), OPENING_RANGE_END)
        or_end_us = dt_to_us(or_end_dt)
        or_i1 = bisect_left(rth_us, or_end_us)
        if or_i1 > 0:
            opening_range_high = max(rth_px[:or_i1])
            opening_range_low = min(rth_px[:or_i1])
        else:
            opening_range_high = None
            opening_range_low = None
        
        # DEBUG — outside event loop
        if idx_day <= 2:
            print("\nDEBUG DAY:", day_ymd)
            print("first 5 rth_ts:", rth_ts[:5])
            print("OPENING_RANGE_END:", OPENING_RANGE_END)
            print("or_end_dt:", or_end_dt)
            print("or_i1:", or_i1)
            print("opening_range_high:", opening_range_high)
            print("opening_range_low:", opening_range_low)

        # prior-day RTH
        prior_day_rth_high = None
        prior_day_rth_low = None
        if df_prev_tr is not None:
            prior_day_rth_high, prior_day_rth_low = rth_high_low(df_prev_tr)

        # overnight = prev post-RTH + current pre-RTH
        overnight_prices: List[float] = []
        if df_prev_tr is not None:
            _, prev_post_px = extract_post_rth_arrays(df_prev_tr)
            overnight_prices.extend(prev_post_px)
        _, curr_pre_px = extract_pre_rth_arrays(df_tr)
        overnight_prices.extend(curr_pre_px)

        overnight_high = max(overnight_prices) if overnight_prices else None
        overnight_low = min(overnight_prices) if overnight_prices else None

        for r in df_day_events.to_dicts():
            t0: datetime = r["ts"]
            t0_us = dt_to_us(t0)
            entry_price = float(r["entry_price"])
            sigma0 = r.get("sigma0")

            # pre-touch RV / rate on full current-day prints
            i0_all = bisect_left(tr_us, t0_us)
            rv_start_us = t0_us - int(RV_WIN_S * 1_000_000)
            rate_start_us = t0_us - int(RATE_WIN_S * 1_000_000)

            i_rv0 = bisect_left(tr_us, rv_start_us)
            i_rate0 = bisect_left(tr_us, rate_start_us)

            rv = realized_variance_points(tr_px, i_rv0, i0_all)
            n_rate = max(0, i0_all - i_rate0)
            trade_rate = n_rate / float(RATE_WIN_S) if RATE_WIN_S > 0 else None
            sigma_lvl = (rv / float(RV_WIN_S)) ** 0.5 if RV_WIN_S > 0 else None

            # d sigma / dt
            t_prev_end_us = t0_us - int(SIGMA_DDT_WIN_S * 1_000_000)
            t_prev_sta_us = t0_us - int(2 * SIGMA_DDT_WIN_S * 1_000_000)
            i_prev_end = bisect_left(tr_us, t_prev_end_us)
            i_prev_sta = bisect_left(tr_us, t_prev_sta_us)
            rv_prev = realized_variance_points(tr_px, i_prev_sta, i_prev_end)
            sigma_prev = (rv_prev / float(SIGMA_DDT_WIN_S)) ** 0.5 if SIGMA_DDT_WIN_S > 0 else None
            dsigma_dt = None
            if sigma_lvl is not None and sigma_prev is not None and SIGMA_DDT_WIN_S > 0:
                dsigma_dt = (sigma_lvl - sigma_prev) / float(SIGMA_DDT_WIN_S)

            # session bucket / time since open
            sess_bucket = time_bucket(t0)
            sess_open_dt = datetime.combine(t0.date(), RTH_OPEN)
            seconds_since_open = (t0 - sess_open_dt).total_seconds()
            sess_open_us = dt_to_us(sess_open_dt)

            # vwap slope
            t1_us = t0_us - int(VWAP_SLOPE_WIN_S * 1_000_000)
            vwap_t0 = session_vwap_at(tr_us, pv_pref, v_pref, sess_open_us, t0_us)
            vwap_t1 = session_vwap_at(tr_us, pv_pref, v_pref, sess_open_us, t1_us)
            vwap_slope_pts_per_s = None
            abs_vwap_slope_pts_per_s = None
            if vwap_t0 is not None and vwap_t1 is not None and VWAP_SLOPE_WIN_S > 0:
                vwap_slope_pts_per_s = (vwap_t0 - vwap_t1) / float(VWAP_SLOPE_WIN_S)
                abs_vwap_slope_pts_per_s = abs(vwap_slope_pts_per_s)

            # current-vwap structural
            vwap0 = float(r["vwap0"])
            distance_from_session_vwap_anchor = entry_price - vwap_open
            z_anchor_vwap = None
            if sigma0 is not None and float(sigma0) > 0:
                z_anchor_vwap = distance_from_session_vwap_anchor / float(sigma0)
            vwap_drift = vwap0 - vwap_open
            vwap0_minus_open_pts = vwap_drift

            # running RTH session range up to event time
            i_rth = bisect_left(rth_us, t0_us)
            session_high_running_t0 = None
            session_low_running_t0 = None
            range_position = None
            dist_day_high = None
            dist_day_low = None
            midrange = None
            midrange_position = None
            distance_from_session_midrange = None
            z_range = None

            if i_rth > 0:
                session_high_running_t0 = rth_high_pref[i_rth - 1]
                session_low_running_t0 = rth_low_pref[i_rth - 1]
                dist_day_high = session_high_running_t0 - entry_price
                dist_day_low = entry_price - session_low_running_t0
                width = session_high_running_t0 - session_low_running_t0
                if width > 0:
                    midrange = 0.5 * (session_high_running_t0 + session_low_running_t0)

                    range_position = (entry_price - session_low_running_t0) / width
                    midrange_position = (entry_price - midrange) / width

                    distance_from_session_midrange = entry_price - midrange
                    z_range = (entry_price - midrange) / width

            #DEBUG 
            if idx_day <= 2 and len(rows_out) < 10:
                print("\nDEBUG EVENT")
                print("event ts:", t0)
                print("event time():", t0.time())
                print("OPENING_RANGE_END:", OPENING_RANGE_END)
                print("opening_range_high:", opening_range_high)
                print("opening_range_low:", opening_range_low)

            # opening range position
            opening_range_position = None
            if t0.time() >= OPENING_RANGE_END and opening_range_high is not None and opening_range_low is not None:
                or_width = opening_range_high - opening_range_low
                if or_width > 0:
                    opening_range_position = (entry_price - opening_range_low) / or_width

            # prior day distances
            distance_from_prior_day_high = None if prior_day_rth_high is None else entry_price - prior_day_rth_high
            distance_from_prior_day_low = None if prior_day_rth_low is None else entry_price - prior_day_rth_low

            # overnight distances
            distance_from_overnight_high = None if overnight_high is None else entry_price - overnight_high
            distance_from_overnight_low = None if overnight_low is None else entry_price - overnight_low

            r.update({
                # old summary-style enrichments
                "day_open": day_open,
                "day_close": day_close,
                "day_trend": day_trend,
                "session_bucket": sess_bucket,
                "seconds_since_open": seconds_since_open,
                "cooldown_bucket": bucket_value(r.get("cooldown_seconds"), COOLDOWN_BINS, COOLDOWN_LABELS),
                "since_touch_any_bucket": bucket_value(r.get("since_touch_any_seconds"), TOUCHANY_BINS, TOUCHANY_LABELS),
                "rv_pre_pts2": rv,
                "trade_rate_pre_per_s": trade_rate,
                "sigma_lvl": sigma_lvl,
                "dsigma_dt": dsigma_dt,
                "vwap_slope_pts_per_s": vwap_slope_pts_per_s,
                "abs_vwap_slope_pts_per_s": abs_vwap_slope_pts_per_s,
                "vwap0_minus_open_pts": vwap0_minus_open_pts,

                # new V1 structural features
                "vwap_open": vwap_open,
                "distance_from_session_vwap_anchor": distance_from_session_vwap_anchor,
                "z_anchor_vwap": z_anchor_vwap,
                "vwap_drift": vwap_drift,
                "session_high_running_t0": session_high_running_t0,
                "session_low_running_t0": session_low_running_t0,
                "range_position": range_position,
                "dist_day_high": dist_day_high,
                "dist_day_low": dist_day_low,
                "midrange": midrange,
                "midrange_position": midrange_position,
                "distance_from_session_midrange": distance_from_session_midrange,
                "z_range": z_range,
                "opening_range_high": opening_range_high,
                "opening_range_low": opening_range_low,
                "opening_range_position": opening_range_position,
                "prior_day_rth_high": prior_day_rth_high,
                "prior_day_rth_low": prior_day_rth_low,
                "distance_from_prior_day_high": distance_from_prior_day_high,
                "distance_from_prior_day_low": distance_from_prior_day_low,
                "overnight_high": overnight_high,
                "overnight_low": overnight_low,
                "distance_from_overnight_high": distance_from_overnight_high,
                "distance_from_overnight_low": distance_from_overnight_low,
            })
            rows_out.append(r)

        if idx_day % 25 == 0:
            print(f"...processed {idx_day} event days")

    if not rows_out:
        print("No enriched rows produced.")
        return

    df_out = pl.from_dicts(rows_out).sort(["date_ymd", "ts", "band_k", "side"])

    # light casting
    float_cols = [
        "entry_price", "vwap0", "sigma0", "distance0",
        "cooldown_seconds", "since_touch_any_seconds",
        "time_to_vwap_s", "rv_pre_pts2", "trade_rate_pre_per_s",
        "sigma_lvl", "dsigma_dt", "vwap_slope_pts_per_s", "abs_vwap_slope_pts_per_s",
        "vwap0_minus_open_pts", "vwap_open", "distance_from_session_vwap_anchor",
        "z_anchor_vwap", "vwap_drift", "session_high_running_t0", "session_low_running_t0",
        "range_position", "dist_day_high", "dist_day_low", "midrange", "midrange_position",
        "distance_from_session_midrange", "z_range",
        "opening_range_high", "opening_range_low", "opening_range_position",
        "prior_day_rth_high", "prior_day_rth_low",
        "distance_from_prior_day_high", "distance_from_prior_day_low",
        "overnight_high", "overnight_low",
        "distance_from_overnight_high", "distance_from_overnight_low",
        "seconds_since_open", "day_open", "day_close",
    ]
    utf8_cols = [
        "event_id", "date_ymd", "side", "session_bucket", "day_trend",
        "cooldown_bucket", "since_touch_any_bucket",
    ]
    bool_cols = ["is_weekend"]

    cast_exprs = []
    for c in float_cols:
        if c in df_out.columns:
            cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    for c in utf8_cols:
        if c in df_out.columns:
            cast_exprs.append(pl.col(c).cast(pl.Utf8, strict=False))
    for c in bool_cols:
        if c in df_out.columns:
            cast_exprs.append(pl.col(c).cast(pl.Boolean, strict=False))
    if "ts" in df_out.columns:
        cast_exprs.append(pl.col("ts").cast(pl.Datetime, strict=False))

    df_out = df_out.with_columns(cast_exprs)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(OUT_FILE)

    print("\nDone.")
    print(f"Wrote: {OUT_FILE}")
    print(f"Rows: {df_out.height:,}  Cols: {len(df_out.columns)}")

    print("\nQuick sanity:")
    print(df_out.select([
        pl.len().alias("n"),
        pl.col("distance_from_session_vwap_anchor").null_count().alias("anchor_nulls"),
        pl.col("range_position").null_count().alias("range_pos_nulls"),
        pl.col("opening_range_position").null_count().alias("or_pos_nulls"),
        pl.col("distance_from_prior_day_high").null_count().alias("prior_high_nulls"),
        pl.col("distance_from_overnight_high").null_count().alias("overnight_high_nulls"),
    ]))

    if missing_trade_days:
        print(f"\n[warn] Missing trade parquet for {len(missing_trade_days)} event days")
        print("Example:", missing_trade_days[:10])


if __name__ == "__main__":
    main()
# build_event_features_v4_orderflow_impact.py
from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import math
import polars as pl


# -----------------------------
# Paths
# -----------------------------
BASE_EVENTS_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v3.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUT_FILE = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v4.parquet"
)


# -----------------------------
# Constants
# -----------------------------
TICK = 0.25
NEAR_LEVELS = [1, 2, 3]

BASE_FLOW_WINDOW_S = 10
SHORT_WINDOW_S = 5
MID_WINDOW_S = 30
LONG_WINDOW_S = 60


# -----------------------------
# Helpers
# -----------------------------
def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def mean_or_none(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std_or_none(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    n = len(vals)
    if n < 2:
        return None
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var)


def slope_per_second_from_series(
    ts_us: List[int],
    xs: List[Optional[float]],
) -> Optional[float]:
    pairs = [(t, float(x)) for t, x in zip(ts_us, xs) if x is not None]
    if len(pairs) < 2:
        return None
    t0, x0 = pairs[0]
    t1, x1 = pairs[-1]
    dt_s = (t1 - t0) / 1_000_000.0
    if dt_s <= 0:
        return None
    return (x1 - x0) / dt_s


def get_level_sum(arr: List[Optional[float]], levels: List[int]) -> float:
    s = 0.0
    for lvl in levels:
        v = arr[lvl]
        if v is not None:
            s += float(v)
    return s


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


def prefix_sum(xs: List[float]) -> List[float]:
    out = [0.0]
    s = 0.0
    for x in xs:
        s += float(x)
        out.append(s)
    return out


def prefix_sum_int(xs: List[int]) -> List[int]:
    out = [0]
    s = 0
    for x in xs:
        s += int(x)
        out.append(s)
    return out


def range_sum(pref: List[float], i0: int, i1: int) -> float:
    i0 = max(0, i0)
    i1 = max(i0, i1)
    i1 = min(i1, len(pref) - 1)
    return pref[i1] - pref[i0]


def range_sum_int(pref: List[int], i0: int, i1: int) -> int:
    i0 = max(0, i0)
    i1 = max(i0, i1)
    i1 = min(i1, len(pref) - 1)
    return pref[i1] - pref[i0]


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


def compute_mid(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    if best_ask <= best_bid:
        return None
    return 0.5 * (best_bid + best_ask)


def compute_spread(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    if best_ask <= best_bid:
        return None
    return best_ask - best_bid


def compute_microprice(
    best_bid: Optional[float],
    best_ask: Optional[float],
    best_bid_queue: Optional[float],
    best_ask_queue: Optional[float],
) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    if best_ask <= best_bid:
        return None
    if best_bid_queue is None or best_ask_queue is None:
        return None
    den = best_bid_queue + best_ask_queue
    if den == 0:
        return None
    return (best_ask * best_bid_queue + best_bid * best_ask_queue) / den


def lag1_autocorr(signs: List[int]) -> Optional[float]:
    vals = [float(x) for x in signs if x != 0]
    n = len(vals)
    if n < 3:
        return None
    x = vals[:-1]
    y = vals[1:]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x)
    deny = sum((b - my) ** 2 for b in y)
    den = math.sqrt(denx * deny)
    if den == 0:
        return None
    return num / den


# -----------------------------
# Loaders
# -----------------------------
def load_l2_day(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)

    required = {"ts", "level", "type", "depth", "price", "volume", "action"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Combined parquet missing required L2 cols: {missing}")

    return (
        df
        .filter(
            (pl.col("level") == 2) &
            (pl.col("type").is_in([0, 1])) &
            pl.col("depth").is_not_null() &
            pl.col("action").is_not_null() &
            pl.col("price").is_not_null() &
            pl.col("volume").is_not_null()
        )
        .with_columns(
            pl.when(pl.col("type") == 0).then(pl.lit("bid"))
            .when(pl.col("type") == 1).then(pl.lit("ask"))
            .otherwise(pl.lit(None))
            .alias("side")
        )
        .select(["ts", "side", "depth", "price", "volume", "action"])
        .sort("ts")
    )


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
def add_null_v4_columns(r: Dict[str, Any]) -> None:
    r.update({
        "trades_per_second": None,
        "volume_per_second": None,
        "avg_trade_size": None,

        "trade_sign_autocorr_10s": None,
        "trade_sign_autocorr_30s": None,
        "trade_sign_autocorr_60s": None,

        "aggressive_buy_volume": None,
        "aggressive_sell_volume": None,
        "aggressive_volume_ratio": None,

        "OFI": None,
        "OFI_5s": None,
        "OFI_30s": None,
        "OFI_60s": None,

        "CVD": None,
        "cvd_slope": None,
        "cvd_acceleration": None,

        "flow_burst_ratio": None,
        "volume_burst_ratio": None,
        "OFI_burst_ratio": None,

        "signed_volume": None,

        "trade_imbalance": None,
        "trade_imbalance_5s": None,
        "trade_imbalance_30s": None,
        "trade_imbalance_60s": None,

        "queue_imbalance_mean_5s": None,
        "queue_imbalance_std_5s": None,
        "queue_imbalance_slope_5s": None,
        "queue_imbalance_positive_ratio": None,

        "impact_efficiency_3s": None,
        "impact_efficiency_10s": None,
        "impact_efficiency_30s": None,

        "impact_decay_3s": None,
        "impact_decay_5s": None,

        "buy_impact_efficiency_10s": None,
        "sell_impact_efficiency_10s": None,
        "impact_asymmetry_10s": None,

        "absorption_ratio": None,
        "flow_impact": None,

        "flow_toxicity": None,
        "flow_toxicity_10s": None,
        "flow_toxicity_30s": None,
        "toxicity_ratio": None,
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
            add_null_v4_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    full_path = Path(full_path)
    if not full_path.exists():
        for r in df_day_events.to_dicts():
            add_null_v4_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    # -------------------------
    # L2 reconstruction
    # -------------------------
    df_l2 = load_l2_day(full_path)
    if df_l2.height == 0:
        for r in df_day_events.to_dicts():
            add_null_v4_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    l2_ts_col = df_l2["ts"].to_list()
    l2_side_col = df_l2["side"].to_list()
    l2_depth_col = df_l2["depth"].to_list()
    l2_price_col = df_l2["price"].to_list()
    l2_volume_col = df_l2["volume"].to_list()
    l2_action_col = df_l2["action"].to_list()

    bid_px: List[Optional[float]] = [None] * 11
    bid_sz: List[Optional[float]] = [None] * 11
    ask_px: List[Optional[float]] = [None] * 11
    ask_sz: List[Optional[float]] = [None] * 11

    l2_us: List[int] = []

    # action metadata
    event_side_code: List[int] = []   # 0 bid, 1 ask
    event_action: List[int] = []

    # snapshot series
    snap_best_bid: List[Optional[float]] = []
    snap_best_ask: List[Optional[float]] = []
    snap_best_bid_queue: List[Optional[float]] = []
    snap_best_ask_queue: List[Optional[float]] = []
    snap_near_bid_total: List[float] = []
    snap_near_ask_total: List[float] = []
    snap_spread: List[Optional[float]] = []
    snap_mid: List[Optional[float]] = []
    snap_microprice: List[Optional[float]] = []
    snap_queue_imbalance: List[Optional[float]] = []

    ofi_delta_series: List[float] = []

    prev_near_bid_total: Optional[float] = None
    prev_near_ask_total: Optional[float] = None

    for ts, side, depth_raw, price_raw, volume_raw, action_raw in zip(
        l2_ts_col, l2_side_col, l2_depth_col, l2_price_col, l2_volume_col, l2_action_col
    ):
        if depth_raw is None or action_raw is None:
            continue

        depth = int(depth_raw)
        price = None if price_raw is None else float(price_raw)
        volume = None if volume_raw is None else float(volume_raw)
        action = int(action_raw)

        if depth < 1 or depth > 10:
            continue

        if side == "bid":
            if action == 2 or volume is None or volume <= 0:
                bid_px[depth] = None
                bid_sz[depth] = None
            else:
                bid_px[depth] = price
                bid_sz[depth] = volume
            scode = 0
        else:
            if action == 2 or volume is None or volume <= 0:
                ask_px[depth] = None
                ask_sz[depth] = None
            else:
                ask_px[depth] = price
                ask_sz[depth] = volume
            scode = 1

        t_us = dt_to_us(ts)

        best_bid = bid_px[1]
        best_ask = ask_px[1]
        best_bid_queue = bid_sz[1]
        best_ask_queue = ask_sz[1]
        spread = compute_spread(best_bid, best_ask)
        mid = compute_mid(best_bid, best_ask)
        microprice = compute_microprice(best_bid, best_ask, best_bid_queue, best_ask_queue)

        near_bid_total = get_level_sum(bid_sz, NEAR_LEVELS)
        near_ask_total = get_level_sum(ask_sz, NEAR_LEVELS)

        # near-book OFI increment
        if prev_near_bid_total is None or prev_near_ask_total is None:
            ofi_delta = 0.0
        else:
            ofi_delta = (near_bid_total - prev_near_bid_total) - (near_ask_total - prev_near_ask_total)

        prev_near_bid_total = near_bid_total
        prev_near_ask_total = near_ask_total

        queue_imbalance = None
        if best_bid_queue is not None and best_ask_queue is not None:
            den = best_bid_queue + best_ask_queue
            queue_imbalance = safe_div(best_bid_queue - best_ask_queue, den)

        l2_us.append(t_us)
        event_side_code.append(scode)
        event_action.append(action)

        snap_best_bid.append(best_bid)
        snap_best_ask.append(best_ask)
        snap_best_bid_queue.append(best_bid_queue)
        snap_best_ask_queue.append(best_ask_queue)
        snap_near_bid_total.append(near_bid_total)
        snap_near_ask_total.append(near_ask_total)
        snap_spread.append(spread)
        snap_mid.append(mid)
        snap_microprice.append(microprice)
        snap_queue_imbalance.append(queue_imbalance)
        ofi_delta_series.append(ofi_delta)

    if not l2_us:
        for r in df_day_events.to_dicts():
            add_null_v4_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    ofi_pref = prefix_sum(ofi_delta_series)

    # -------------------------
    # Trades + aggressor classification
    # -------------------------
    df_tr = load_trade_day(full_path)
    if df_tr.height == 0:
        for r in df_day_events.to_dicts():
            add_null_v4_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    tr_ts = df_tr["ts"].to_list()
    tr_px = [float(x) for x in df_tr["price"].to_list()]
    tr_vol = [float(x) for x in df_tr["volume"].to_list()]
    tr_us = [dt_to_us(x) for x in tr_ts]

    trade_sign: List[int] = []
    last_nonzero_sign = 0

    for i, (t_us, px) in enumerate(zip(tr_us, tr_px)):
        j = bisect_right(l2_us, t_us) - 1

        sign = 0
        if j >= 0:
            bb = snap_best_bid[j]
            ba = snap_best_ask[j]

            if bb is not None and ba is not None and ba > bb:
                if px >= ba - 1e-12:
                    sign = 1
                elif px <= bb + 1e-12:
                    sign = -1

        # tick-rule fallback
        if sign == 0:
            if i > 0:
                if px > tr_px[i - 1]:
                    sign = 1
                elif px < tr_px[i - 1]:
                    sign = -1
                else:
                    sign = last_nonzero_sign

        if sign != 0:
            last_nonzero_sign = sign

        trade_sign.append(sign)

    buy_vol_series = [v if s > 0 else 0.0 for v, s in zip(tr_vol, trade_sign)]
    sell_vol_series = [v if s < 0 else 0.0 for v, s in zip(tr_vol, trade_sign)]
    signed_vol_series = [v * s for v, s in zip(tr_vol, trade_sign)]
    buy_cnt_series = [1 if s > 0 else 0 for s in trade_sign]
    sell_cnt_series = [1 if s < 0 else 0 for s in trade_sign]

    tr_vol_pref = prefix_sum(tr_vol)
    buy_vol_pref = prefix_sum(buy_vol_series)
    sell_vol_pref = prefix_sum(sell_vol_series)
    signed_vol_pref = prefix_sum(signed_vol_series)
    buy_cnt_pref = prefix_sum_int(buy_cnt_series)
    sell_cnt_pref = prefix_sum_int(sell_cnt_series)

    # session CVD at each trade index
    cvd_series: List[float] = []
    s = 0.0
    for x in signed_vol_series:
        s += x
        cvd_series.append(s)

    # -------------------------
    # Event processing
    # -------------------------
    for r in df_day_events.to_dicts():
        t0: datetime = r["ts"]
        t0_us = dt_to_us(t0)

        # latest L2 snapshot <= t
        j = bisect_right(l2_us, t0_us) - 1
        if j < 0:
            add_null_v4_columns(r)
            rows_out_day.append(r)
            continue

        # trade windows
        tr_i0_5, tr_i1_5 = window_bounds(tr_us, t0_us, SHORT_WINDOW_S)
        tr_i0_10, tr_i1_10 = window_bounds(tr_us, t0_us, BASE_FLOW_WINDOW_S)
        tr_i0_30, tr_i1_30 = window_bounds(tr_us, t0_us, MID_WINDOW_S)
        tr_i0_60, tr_i1_60 = window_bounds(tr_us, t0_us, LONG_WINDOW_S)

        # l2 windows
        l2_i0_5, l2_i1_5 = window_bounds(l2_us, t0_us, SHORT_WINDOW_S)
        l2_i0_30, l2_i1_30 = window_bounds(l2_us, t0_us, MID_WINDOW_S)
        l2_i0_60, l2_i1_60 = window_bounds(l2_us, t0_us, LONG_WINDOW_S)

        # -------------------------
        # Base trade activity (10s)
        # -------------------------
        n_tr_10 = tr_i1_10 - tr_i0_10
        vol_10 = range_sum(tr_vol_pref, tr_i0_10, tr_i1_10)
        buy_vol_10 = range_sum(buy_vol_pref, tr_i0_10, tr_i1_10)
        sell_vol_10 = range_sum(sell_vol_pref, tr_i0_10, tr_i1_10)
        signed_vol_10 = range_sum(signed_vol_pref, tr_i0_10, tr_i1_10)

        trades_per_second = n_tr_10 / BASE_FLOW_WINDOW_S
        volume_per_second = vol_10 / BASE_FLOW_WINDOW_S
        avg_trade_size = safe_div(vol_10, n_tr_10)

        aggressive_buy_volume = buy_vol_10
        aggressive_sell_volume = sell_vol_10
        aggressive_volume_ratio = safe_div(buy_vol_10, sell_vol_10)

        # -------------------------
        # Trade sign autocorr
        # -------------------------
        trade_sign_autocorr_10s = lag1_autocorr(trade_sign[tr_i0_10:tr_i1_10])
        trade_sign_autocorr_30s = lag1_autocorr(trade_sign[tr_i0_30:tr_i1_30])
        trade_sign_autocorr_60s = lag1_autocorr(trade_sign[tr_i0_60:tr_i1_60])

        # -------------------------
        # OFI (near-book)
        # -------------------------
        OFI_5s = range_sum(ofi_pref, l2_i0_5, l2_i1_5)
        OFI_30s = range_sum(ofi_pref, l2_i0_30, l2_i1_30)
        OFI_60s = range_sum(ofi_pref, l2_i0_60, l2_i1_60)
        OFI = range_sum(ofi_pref, *window_bounds(l2_us, t0_us, BASE_FLOW_WINDOW_S))

        # -------------------------
        # CVD
        # -------------------------
        tr_j = bisect_right(tr_us, t0_us) - 1
        CVD = None if tr_j < 0 else cvd_series[tr_j]

        signed_vol_5 = range_sum(signed_vol_pref, tr_i0_5, tr_i1_5)
        signed_vol_30 = range_sum(signed_vol_pref, tr_i0_30, tr_i1_30)

        cvd_slope = safe_div(signed_vol_10, BASE_FLOW_WINDOW_S)
        cvd_acceleration = None
        short_rate = safe_div(signed_vol_5, SHORT_WINDOW_S)
        long_rate = safe_div(signed_vol_30, MID_WINDOW_S)
        if short_rate is not None and long_rate is not None:
            cvd_acceleration = short_rate - long_rate

        # -------------------------
        # Burst ratios
        # -------------------------
        trades_per_second_60 = (tr_i1_60 - tr_i0_60) / LONG_WINDOW_S
        volume_per_second_60 = range_sum(tr_vol_pref, tr_i0_60, tr_i1_60) / LONG_WINDOW_S
        ofi_rate_5 = safe_div(abs(OFI_5s), SHORT_WINDOW_S)
        ofi_rate_60 = safe_div(abs(OFI_60s), LONG_WINDOW_S)

        flow_burst_ratio = safe_div(trades_per_second, trades_per_second_60)
        volume_burst_ratio = safe_div(volume_per_second, volume_per_second_60)
        OFI_burst_ratio = safe_div(ofi_rate_5, ofi_rate_60)

        # -------------------------
        # Signed volume / trade imbalance
        # -------------------------
        signed_volume = signed_vol_10

        def trade_imbalance_window(i0: int, i1: int) -> Optional[float]:
            buy_n = range_sum_int(buy_cnt_pref, i0, i1)
            sell_n = range_sum_int(sell_cnt_pref, i0, i1)
            den = buy_n + sell_n
            if den == 0:
                return None
            return (buy_n - sell_n) / den

        trade_imbalance = trade_imbalance_window(tr_i0_10, tr_i1_10)
        trade_imbalance_5s = trade_imbalance_window(tr_i0_5, tr_i1_5)
        trade_imbalance_30s = trade_imbalance_window(tr_i0_30, tr_i1_30)
        trade_imbalance_60s = trade_imbalance_window(tr_i0_60, tr_i1_60)

        # -------------------------
        # Queue imbalance 5s stats
        # -------------------------
        q_i0, q_i1 = l2_i0_5, l2_i1_5
        q_ts = l2_us[q_i0:q_i1]
        q_vals = snap_queue_imbalance[q_i0:q_i1]

        queue_imbalance_mean_5s = mean_or_none(q_vals)
        queue_imbalance_std_5s = std_or_none(q_vals)
        queue_imbalance_slope_5s = slope_per_second_from_series(q_ts, q_vals)

        q_nonnull = [float(x) for x in q_vals if x is not None]
        if q_nonnull:
            queue_imbalance_positive_ratio = sum(1 for x in q_nonnull if x > 0) / len(q_nonnull)
        else:
            queue_imbalance_positive_ratio = None

        # -------------------------
        # Impact / absorption
        # -------------------------
        cur_mid = snap_mid[j]

        mid_3s_ago = lagged_value_at_or_before(l2_us, snap_mid, t0_us, 3)
        mid_5s_ago = lagged_value_at_or_before(l2_us, snap_mid, t0_us, 5)
        mid_10s_ago = lagged_value_at_or_before(l2_us, snap_mid, t0_us, 10)
        mid_30s_ago = lagged_value_at_or_before(l2_us, snap_mid, t0_us, 30)

        def abs_mid_move(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
            if cur is None or prev is None:
                return None
            return abs(cur - prev)

        move_3 = abs_mid_move(cur_mid, mid_3s_ago)
        move_10 = abs_mid_move(cur_mid, mid_10s_ago)
        move_30 = abs_mid_move(cur_mid, mid_30s_ago)

        signed_move_10 = None
        if cur_mid is not None and mid_10s_ago is not None:
            signed_move_10 = cur_mid - mid_10s_ago

        # signed vol windows for impact
        tr_i0_3, tr_i1_3 = window_bounds(tr_us, t0_us, 3)
        vol_signed_3 = range_sum(signed_vol_pref, tr_i0_3, tr_i1_3)
        vol_signed_10 = signed_vol_10
        vol_signed_30 = signed_vol_30

        impact_efficiency_3s = safe_div(move_3, abs(vol_signed_3))
        impact_efficiency_10s = safe_div(move_10, abs(vol_signed_10))
        impact_efficiency_30s = safe_div(move_30, abs(vol_signed_30))

        # decay = recent impact efficiency / prior-window impact efficiency
        def prior_efficiency(window_s: int) -> Optional[float]:
            end_us = t0_us - int(window_s * 1_000_000)
            start_us = end_us - int(window_s * 1_000_000)

            tr_p0 = bisect_left(tr_us, start_us)
            tr_p1 = bisect_right(tr_us, end_us)

            mid_end = lagged_value_at_or_before(l2_us, snap_mid, t0_us, window_s)
            mid_start = lagged_value_at_or_before(l2_us, snap_mid, t0_us, 2 * window_s)
            if mid_end is None or mid_start is None:
                return None

            move = abs(mid_end - mid_start)
            signed_vol = range_sum(signed_vol_pref, tr_p0, tr_p1)
            return safe_div(move, abs(signed_vol))

        prev_eff_3 = prior_efficiency(3)
        prev_eff_5 = prior_efficiency(5)

        current_eff_3 = impact_efficiency_3s

        tr_i0_5b, tr_i1_5b = window_bounds(tr_us, t0_us, 5)
        mid_5_move = abs_mid_move(cur_mid, mid_5s_ago)
        signed_vol_5b = range_sum(signed_vol_pref, tr_i0_5b, tr_i1_5b)
        current_eff_5 = safe_div(mid_5_move, abs(signed_vol_5b))

        impact_decay_3s = safe_div(current_eff_3, prev_eff_3)
        impact_decay_5s = safe_div(current_eff_5, prev_eff_5)

        buy_impact_efficiency_10s = None
        sell_impact_efficiency_10s = None
        if signed_move_10 is not None:
            if signed_move_10 > 0:
                buy_impact_efficiency_10s = safe_div(signed_move_10, buy_vol_10)
                sell_impact_efficiency_10s = safe_div(0.0, sell_vol_10) if sell_vol_10 > 0 else 0.0
            elif signed_move_10 < 0:
                buy_impact_efficiency_10s = safe_div(0.0, buy_vol_10) if buy_vol_10 > 0 else 0.0
                sell_impact_efficiency_10s = safe_div(abs(signed_move_10), sell_vol_10)
            else:
                buy_impact_efficiency_10s = 0.0
                sell_impact_efficiency_10s = 0.0

        impact_asymmetry_10s = None
        if buy_impact_efficiency_10s is not None and sell_impact_efficiency_10s is not None:
            impact_asymmetry_10s = buy_impact_efficiency_10s - sell_impact_efficiency_10s

        aggressive_total_10 = buy_vol_10 + sell_vol_10
        move_10_ticks = None if move_10 is None else move_10 / TICK
        absorption_ratio = safe_div(aggressive_total_10, move_10_ticks)

        flow_impact = safe_div(signed_move_10, signed_vol_10)

        # -------------------------
        # Toxicity
        # -------------------------
        rolling_volume_mean_10 = volume_per_second_60 * BASE_FLOW_WINDOW_S if volume_per_second_60 is not None else None
        flow_toxicity_10s = safe_div(signed_vol_10, rolling_volume_mean_10)

        vol_30 = range_sum(tr_vol_pref, tr_i0_30, tr_i1_30)
        rolling_volume_mean_30 = volume_per_second_60 * MID_WINDOW_S if volume_per_second_60 is not None else None
        flow_toxicity_30s = safe_div(signed_vol_30, rolling_volume_mean_30)

        flow_toxicity = flow_toxicity_10s
        toxicity_ratio = safe_div(abs(signed_vol_10), aggressive_total_10)

        r.update({
            "trades_per_second": trades_per_second,
            "volume_per_second": volume_per_second,
            "avg_trade_size": avg_trade_size,

            "trade_sign_autocorr_10s": trade_sign_autocorr_10s,
            "trade_sign_autocorr_30s": trade_sign_autocorr_30s,
            "trade_sign_autocorr_60s": trade_sign_autocorr_60s,

            "aggressive_buy_volume": aggressive_buy_volume,
            "aggressive_sell_volume": aggressive_sell_volume,
            "aggressive_volume_ratio": aggressive_volume_ratio,

            "OFI": OFI,
            "OFI_5s": OFI_5s,
            "OFI_30s": OFI_30s,
            "OFI_60s": OFI_60s,

            "CVD": CVD,
            "cvd_slope": cvd_slope,
            "cvd_acceleration": cvd_acceleration,

            "flow_burst_ratio": flow_burst_ratio,
            "volume_burst_ratio": volume_burst_ratio,
            "OFI_burst_ratio": OFI_burst_ratio,

            "signed_volume": signed_volume,

            "trade_imbalance": trade_imbalance,
            "trade_imbalance_5s": trade_imbalance_5s,
            "trade_imbalance_30s": trade_imbalance_30s,
            "trade_imbalance_60s": trade_imbalance_60s,

            "queue_imbalance_mean_5s": queue_imbalance_mean_5s,
            "queue_imbalance_std_5s": queue_imbalance_std_5s,
            "queue_imbalance_slope_5s": queue_imbalance_slope_5s,
            "queue_imbalance_positive_ratio": queue_imbalance_positive_ratio,

            "impact_efficiency_3s": impact_efficiency_3s,
            "impact_efficiency_10s": impact_efficiency_10s,
            "impact_efficiency_30s": impact_efficiency_30s,

            "impact_decay_3s": impact_decay_3s,
            "impact_decay_5s": impact_decay_5s,

            "buy_impact_efficiency_10s": buy_impact_efficiency_10s,
            "sell_impact_efficiency_10s": sell_impact_efficiency_10s,
            "impact_asymmetry_10s": impact_asymmetry_10s,

            "absorption_ratio": absorption_ratio,
            "flow_impact": flow_impact,

            "flow_toxicity": flow_toxicity,
            "flow_toxicity_10s": flow_toxicity_10s,
            "flow_toxicity_30s": flow_toxicity_30s,
            "toxicity_ratio": toxicity_ratio,
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
        "trades_per_second",
        "trade_sign_autocorr_10s",
        "aggressive_buy_volume",
        "OFI_5s",
        "CVD",
        "trade_imbalance_5s",
        "queue_imbalance_mean_5s",
        "impact_efficiency_10s",
        "impact_decay_3s",
        "flow_toxicity_10s",
        "toxicity_ratio",
    ] if c in df_out.columns]

    exprs = [pl.len().alias("n")] + [
        pl.col(c).null_count().alias(f"{c}_nulls") for c in sanity_cols
    ]
    print(df_out.select(exprs))


if __name__ == "__main__":
    main()
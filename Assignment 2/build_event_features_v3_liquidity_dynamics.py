# build_event_features_v3_liquidity_dynamics.py
from __future__ import annotations

from bisect import bisect_left, bisect_right
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
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v2.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUT_FILE = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v3.parquet"
)


# -----------------------------
# Constants
# -----------------------------
TICK = 0.25
DYNAMICS_WINDOW_S = 5
NEAR_LEVELS = [1, 2, 3]


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


def sum_positive_deltas(xs: List[Optional[float]]) -> Optional[float]:
    vals = [None if x is None else float(x) for x in xs]
    total = 0.0
    seen = False
    for i in range(1, len(vals)):
        a = vals[i - 1]
        b = vals[i]
        if a is None or b is None:
            continue
        seen = True
        d = b - a
        if d > 0:
            total += d
    return total if seen else None


def sum_negative_deltas_abs(xs: List[Optional[float]]) -> Optional[float]:
    vals = [None if x is None else float(x) for x in xs]
    total = 0.0
    seen = False
    for i in range(1, len(vals)):
        a = vals[i - 1]
        b = vals[i]
        if a is None or b is None:
            continue
        seen = True
        d = b - a
        if d < 0:
            total += abs(d)
    return total if seen else None


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


def load_l2_day(path: Path) -> pl.DataFrame:
    """
    Load L2 bid/ask book updates from combined MarketTick parquet.

    MarketTick:
      level == 2  -> L2 row
      type == 0   -> bid
      type == 1   -> ask
    """
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


def compute_mid(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    return 0.5 * (best_bid + best_ask)


def compute_spread(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
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
    if best_bid_queue is None or best_ask_queue is None:
        return None
    den = best_bid_queue + best_ask_queue
    if den == 0:
        return None
    return (best_ask * best_bid_queue + best_bid * best_ask_queue) / den


def add_null_v3_columns(r: Dict[str, Any]) -> None:
    r.update({
        "microprice_deviation": None,
        "microprice_pressure": None,

        "spread_change_rate": None,
        "spread_volatility": None,
        "spread_expansion_ratio": None,

        "cancel_rate_bid": None,
        "cancel_rate_ask": None,
        "cancel_ratio": None,

        "add_rate_bid": None,
        "add_rate_ask": None,

        "remove_rate_near_bid": None,
        "remove_rate_near_ask": None,

        "update_rate_bid": None,
        "update_rate_ask": None,

        "bid_liquidity_shock": None,
        "ask_liquidity_shock": None,
        "net_liquidity_shock": None,

        "replenishment_rate_near_5s": None,

        "best_bid_queue_norm": None,
        "best_ask_queue_norm": None,
        "queue_length_total_norm": None,

        "bid_queue_change_rate": None,
        "ask_queue_change_rate": None,

        "bid_queue_depletion_rate": None,
        "ask_queue_depletion_rate": None,

        "queue_imbalance": None,

        "microprice": None,
        "microprice_minus_mid": None,
        "spread": None,
        "spread_mid_ratio": None,
        "best_bid_queue_ratio": None,
        "best_ask_queue_ratio": None,
        "queue_length_ratio": None,
    })


def process_event_day(
    day_ymd: str,
    full_path: Optional[str],
    df_day_events: pl.DataFrame,
) -> List[Dict[str, Any]]:
    rows_out_day: List[Dict[str, Any]] = []

    if full_path is None:
        for r in df_day_events.to_dicts():
            add_null_v3_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    full_path = Path(full_path)
    if not full_path.exists():
        for r in df_day_events.to_dicts():
            add_null_v3_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    df_l2 = load_l2_day(full_path)
    if df_l2.height == 0:
        for r in df_day_events.to_dicts():
            add_null_v3_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    l2_ts_col = df_l2["ts"].to_list()
    l2_side_col = df_l2["side"].to_list()
    l2_depth_col = df_l2["depth"].to_list()
    l2_price_col = df_l2["price"].to_list()
    l2_volume_col = df_l2["volume"].to_list()
    l2_action_col = df_l2["action"].to_list()

    # Reconstruct book state across L2 updates
    bid_px: List[Optional[float]] = [None] * 11
    bid_sz: List[Optional[float]] = [None] * 11
    ask_px: List[Optional[float]] = [None] * 11
    ask_sz: List[Optional[float]] = [None] * 11

    l2_us: List[int] = []

    # per-update metadata for action rates
    event_side_code: List[int] = []   # 0 bid, 1 ask
    event_depth: List[int] = []
    event_action: List[int] = []

    # precomputed snapshot series
    snap_best_bid: List[Optional[float]] = []
    snap_best_ask: List[Optional[float]] = []
    snap_best_bid_queue: List[Optional[float]] = []
    snap_best_ask_queue: List[Optional[float]] = []
    snap_near_bid_total: List[float] = []
    snap_near_ask_total: List[float] = []
    snap_spread: List[Optional[float]] = []
    snap_mid: List[Optional[float]] = []
    snap_microprice: List[Optional[float]] = []

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

        if (
            best_bid is None or
            best_ask is None or
            best_ask <= best_bid
        ):
            spread = None
            mid = None
            microprice = None
        else:
            spread = best_ask - best_bid
            mid = 0.5 * (best_bid + best_ask)
            microprice = compute_microprice(
                best_bid,
                best_ask,
                best_bid_queue,
                best_ask_queue,
            )

        l2_us.append(t_us)
        event_side_code.append(scode)
        event_depth.append(depth)
        event_action.append(action)

        snap_best_bid.append(best_bid)
        snap_best_ask.append(best_ask)
        snap_best_bid_queue.append(best_bid_queue)
        snap_best_ask_queue.append(best_ask_queue)
        snap_near_bid_total.append(get_level_sum(bid_sz, NEAR_LEVELS))
        snap_near_ask_total.append(get_level_sum(ask_sz, NEAR_LEVELS))
        snap_spread.append(spread)
        snap_mid.append(mid)
        snap_microprice.append(microprice)

    if not l2_us:
        for r in df_day_events.to_dicts():
            add_null_v3_columns(r)
            rows_out_day.append(r)
        return rows_out_day

    for r in df_day_events.to_dicts():
        t0: datetime = r["ts"]
        t0_us = dt_to_us(t0)

        j = bisect_right(l2_us, t0_us) - 1
        if j < 0:
            add_null_v3_columns(r)
            rows_out_day.append(r)
            continue

        start_us = t0_us - int(DYNAMICS_WINDOW_S * 1_000_000)
        i0 = bisect_left(l2_us, start_us)
        i1 = bisect_right(l2_us, t0_us)

        if i1 <= i0:
            add_null_v3_columns(r)
            rows_out_day.append(r)
            continue

        win_ts = l2_us[i0:i1]
        win_best_bid = snap_best_bid[i0:i1]
        win_best_ask = snap_best_ask[i0:i1]
        win_best_bid_queue = snap_best_bid_queue[i0:i1]
        win_best_ask_queue = snap_best_ask_queue[i0:i1]
        win_near_bid_total = snap_near_bid_total[i0:i1]
        win_near_ask_total = snap_near_ask_total[i0:i1]
        win_spread = snap_spread[i0:i1]
        win_mid = snap_mid[i0:i1]
        win_microprice = snap_microprice[i0:i1]

        cur_best_bid = snap_best_bid[j]
        cur_best_ask = snap_best_ask[j]
        cur_best_bid_queue = snap_best_bid_queue[j]
        cur_best_ask_queue = snap_best_ask_queue[j]
        cur_near_bid_total = snap_near_bid_total[j]
        cur_near_ask_total = snap_near_ask_total[j]
        cur_spread = snap_spread[j]
        cur_mid = snap_mid[j]
        cur_microprice = snap_microprice[j]
    
        # -------------------------
        # Raw book quantities
        # -------------------------
        microprice_minus_mid = None
        if cur_microprice is not None and cur_mid is not None:
            microprice_minus_mid = cur_microprice - cur_mid

        spread_mid_ratio = safe_div(cur_spread, cur_mid)

        best_bid_queue_ratio = None
        best_ask_queue_ratio = None
        queue_imbalance = None
        if cur_best_bid_queue is not None and cur_best_ask_queue is not None:
            den_q = cur_best_bid_queue + cur_best_ask_queue
            best_bid_queue_ratio = safe_div(cur_best_bid_queue, den_q)
            best_ask_queue_ratio = safe_div(cur_best_ask_queue, den_q)
            queue_imbalance = safe_div(cur_best_bid_queue - cur_best_ask_queue, den_q)

        queue_length_ratio = safe_div(cur_best_bid_queue, cur_best_ask_queue)

        # -------------------------
        # Microprice dynamics
        # -------------------------
        mp_mean = mean_or_none(win_microprice)
        mp_std = std_or_none(win_microprice)

        microprice_deviation = None
        if cur_microprice is not None and mp_mean is not None:
            # z-score if std exists, otherwise raw centered deviation
            if mp_std is not None and mp_std > 0:
                microprice_deviation = (cur_microprice - mp_mean) / mp_std
            else:
                microprice_deviation = cur_microprice - mp_mean

        microprice_pressure = safe_div(microprice_minus_mid, cur_spread)

        # -------------------------
        # Spread dynamics
        # -------------------------
        spread_change_rate = slope_per_second_from_series(win_ts, win_spread)
        spread_volatility = std_or_none(win_spread)
        spread_mean = mean_or_none(win_spread)
        spread_expansion_ratio = safe_div(cur_spread, spread_mean)

        # -------------------------
        # Action-count rates in window
        # -------------------------
        bid_cancel_n = 0
        ask_cancel_n = 0
        bid_add_n = 0
        ask_add_n = 0
        bid_update_n = 0
        ask_update_n = 0
        total_cancel_n = 0
        total_event_n = 0

        for q in range(i0, i1):
            sc = event_side_code[q]
            ac = event_action[q]
            total_event_n += 1

            if ac == 2:
                total_cancel_n += 1
                if sc == 0:
                    bid_cancel_n += 1
                else:
                    ask_cancel_n += 1
            elif ac == 0:
                if sc == 0:
                    bid_add_n += 1
                else:
                    ask_add_n += 1
            elif ac == 1:
                if sc == 0:
                    bid_update_n += 1
                else:
                    ask_update_n += 1

        W = float(DYNAMICS_WINDOW_S)
        cancel_rate_bid = bid_cancel_n / W
        cancel_rate_ask = ask_cancel_n / W
        add_rate_bid = bid_add_n / W
        add_rate_ask = ask_add_n / W
        update_rate_bid = bid_update_n / W
        update_rate_ask = ask_update_n / W
        cancel_ratio = safe_div(total_cancel_n, total_event_n)

        # -------------------------
        # Volume-change / removal rates
        # remove_rate_* = aggregate displayed-size decreases per second
        # -------------------------
        remove_bid = sum_negative_deltas_abs(win_near_bid_total)
        remove_ask = sum_negative_deltas_abs(win_near_ask_total)
        remove_rate_near_bid = safe_div(remove_bid, W)
        remove_rate_near_ask = safe_div(remove_ask, W)

        # -------------------------
        # Liquidity shocks
        # shock = z-score of current near-side liquidity vs trailing window
        # -------------------------
        bid_mean = mean_or_none(win_near_bid_total)
        ask_mean = mean_or_none(win_near_ask_total)
        bid_std = std_or_none(win_near_bid_total)
        ask_std = std_or_none(win_near_ask_total)

        bid_liquidity_shock = None
        ask_liquidity_shock = None

        if bid_mean is not None:
            if bid_std is not None and bid_std > 0:
                bid_liquidity_shock = (cur_near_bid_total - bid_mean) / bid_std
            else:
                bid_liquidity_shock = cur_near_bid_total - bid_mean

        if ask_mean is not None:
            if ask_std is not None and ask_std > 0:
                ask_liquidity_shock = (cur_near_ask_total - ask_mean) / ask_std
            else:
                ask_liquidity_shock = cur_near_ask_total - ask_mean

        net_liquidity_shock = None
        if bid_liquidity_shock is not None and ask_liquidity_shock is not None:
            net_liquidity_shock = bid_liquidity_shock - ask_liquidity_shock

        # -------------------------
        # Replenishment
        # replenishment_rate_near_5s = aggregate positive increases in near depth per second
        # across both sides within W
        # -------------------------
        repl_bid = sum_positive_deltas(win_near_bid_total)
        repl_ask = sum_positive_deltas(win_near_ask_total)
        repl_total = None
        if repl_bid is not None and repl_ask is not None:
            repl_total = repl_bid + repl_ask
        elif repl_bid is not None:
            repl_total = repl_bid
        elif repl_ask is not None:
            repl_total = repl_ask
        replenishment_rate_near_5s = safe_div(repl_total, W)

        # -------------------------
        # Queue normalization
        # normalized against trailing window mean
        # -------------------------
        bid_q_mean = mean_or_none(win_best_bid_queue)
        ask_q_mean = mean_or_none(win_best_ask_queue)

        total_q_series: List[Optional[float]] = []
        for bq, aq in zip(win_best_bid_queue, win_best_ask_queue):
            if bq is None or aq is None:
                total_q_series.append(None)
            else:
                total_q_series.append(float(bq) + float(aq))

        cur_total_q = None
        if cur_best_bid_queue is not None and cur_best_ask_queue is not None:
            cur_total_q = cur_best_bid_queue + cur_best_ask_queue

        total_q_mean = mean_or_none(total_q_series)

        best_bid_queue_norm = safe_div(cur_best_bid_queue, bid_q_mean)
        best_ask_queue_norm = safe_div(cur_best_ask_queue, ask_q_mean)
        queue_length_total_norm = safe_div(cur_total_q, total_q_mean)

        # -------------------------
        # Queue rates
        # -------------------------
        bid_queue_change_rate = slope_per_second_from_series(win_ts, win_best_bid_queue)
        ask_queue_change_rate = slope_per_second_from_series(win_ts, win_best_ask_queue)

        bid_q_depl = sum_negative_deltas_abs(win_best_bid_queue)
        ask_q_depl = sum_negative_deltas_abs(win_best_ask_queue)

        bid_queue_depletion_rate = safe_div(bid_q_depl, W)
        ask_queue_depletion_rate = safe_div(ask_q_depl, W)

        r.update({
            "microprice_deviation": microprice_deviation,
            "microprice_pressure": microprice_pressure,

            "spread_change_rate": spread_change_rate,
            "spread_volatility": spread_volatility,
            "spread_expansion_ratio": spread_expansion_ratio,

            "cancel_rate_bid": cancel_rate_bid,
            "cancel_rate_ask": cancel_rate_ask,
            "cancel_ratio": cancel_ratio,

            "add_rate_bid": add_rate_bid,
            "add_rate_ask": add_rate_ask,

            "remove_rate_near_bid": remove_rate_near_bid,
            "remove_rate_near_ask": remove_rate_near_ask,

            "update_rate_bid": update_rate_bid,
            "update_rate_ask": update_rate_ask,

            "bid_liquidity_shock": bid_liquidity_shock,
            "ask_liquidity_shock": ask_liquidity_shock,
            "net_liquidity_shock": net_liquidity_shock,

            "replenishment_rate_near_5s": replenishment_rate_near_5s,

            "best_bid_queue_norm": best_bid_queue_norm,
            "best_ask_queue_norm": best_ask_queue_norm,
            "queue_length_total_norm": queue_length_total_norm,

            "bid_queue_change_rate": bid_queue_change_rate,
            "ask_queue_change_rate": ask_queue_change_rate,

            "bid_queue_depletion_rate": bid_queue_depletion_rate,
            "ask_queue_depletion_rate": ask_queue_depletion_rate,

            "queue_imbalance": queue_imbalance,

            "microprice": cur_microprice,
            "microprice_minus_mid": microprice_minus_mid,
            "spread": cur_spread,
            "spread_mid_ratio": spread_mid_ratio,
            "best_bid_queue_ratio": best_bid_queue_ratio,
            "best_ask_queue_ratio": best_ask_queue_ratio,
            "queue_length_ratio": queue_length_ratio,
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
        day_jobs.append(
            (
                day_ymd,
                None if full_path is None else str(full_path),
                df_day_events,
            )
        )

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
        "microprice_deviation",
        "microprice_pressure",
        "spread_change_rate",
        "spread_volatility",
        "cancel_rate_bid",
        "remove_rate_near_bid",
        "bid_liquidity_shock",
        "best_bid_queue_norm",
        "queue_imbalance",
        "microprice",
        "spread",
    ] if c in df_out.columns]

    exprs = [pl.len().alias("n")] + [
        pl.col(c).null_count().alias(f"{c}_nulls") for c in sanity_cols
    ]
    print(df_out.select(exprs))


if __name__ == "__main__":
    main()
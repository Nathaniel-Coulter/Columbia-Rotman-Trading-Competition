# BACKUP build_event_features_v2_l2 (no parallelizing)
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import math
import polars as pl


# -----------------------------
# Paths
# -----------------------------
BASE_EVENTS_PATH = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v1.parquet"
)

FULL_TICK_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_parquet"
)

OUT_FILE = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_touch_events\event_features_v2.parquet"
)


# -----------------------------
# Constants
# -----------------------------
TICK = 0.25

RTH_OPEN = time(14, 30)
RTH_CLOSE = time(21, 0)

# walls
WALL_KS = [3, 5]
WALL_AVG_WINDOWS_S = [2, 5]

# value area
VALUE_AREA_PCT = 0.70

# depth bucket convention
NEAR_LEVELS = [1, 2, 3]
MID_LEVELS = [4, 5, 6]
FAR_LEVELS = [7, 8, 9, 10]

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


def load_trade_day(path: Path) -> pl.DataFrame:
    """
    Load actual trade prints from the combined MarketTick parquet.

    MarketTick type codes:
      0 = Bid
      1 = Ask
      2 = Last (Trade)

    We keep only trade rows (type == 2).
    """
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


def load_l2_day(path: Path) -> pl.DataFrame:
    """
    Load L2 bid/ask book updates from the combined MarketTick parquet.

    MarketTick:
      level == 2  -> L2 row
      type == 0   -> bid
      type == 1   -> ask

    Output columns:
      ts, side, depth, price, volume, action
      where side in {"bid", "ask"}
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

    

def prefix_sum(xs: List[float]) -> List[float]:
    out = [0.0]
    s = 0.0
    for x in xs:
        s += float(x)
        out.append(s)
    return out


def range_sum(pref: List[float], i0: int, i1: int) -> float:
    i0 = max(0, i0)
    i1 = max(i0, i1)
    i1 = min(i1, len(pref) - 1)
    return pref[i1] - pref[i0]


def get_level_sum(arr: List[Optional[float]], levels: List[int]) -> float:
    s = 0.0
    for lvl in levels:
        v = arr[lvl]
        if v is not None:
            s += float(v)
    return s


def side_depths(level_sz: List[Optional[float]], n: int) -> List[float]:
    out = []
    for i in range(1, n + 1):
        v = level_sz[i]
        out.append(0.0 if v is None else float(v))
    return out


def compute_obi(level_sz_bid: List[Optional[float]], level_sz_ask: List[Optional[float]], n: int) -> Optional[float]:
    bid = sum(0.0 if level_sz_bid[i] is None else float(level_sz_bid[i]) for i in range(1, n + 1))
    ask = sum(0.0 if level_sz_ask[i] is None else float(level_sz_ask[i]) for i in range(1, n + 1))
    den = bid + ask
    if den == 0:
        return None
    return (bid - ask) / den


def compute_weighted_obi(
    level_px_bid: List[Optional[float]],
    level_sz_bid: List[Optional[float]],
    level_px_ask: List[Optional[float]],
    level_sz_ask: List[Optional[float]],
    entry_price: float,
    n: int,
) -> Optional[float]:
    wb = 0.0
    wa = 0.0

    for i in range(1, n + 1):
        pb = level_px_bid[i]
        vb = level_sz_bid[i]
        if pb is not None and vb is not None:
            dp = abs(entry_price - float(pb))
            if dp > 0:
                wb += float(vb) / dp

        pa = level_px_ask[i]
        va = level_sz_ask[i]
        if pa is not None and va is not None:
            dp = abs(float(pa) - entry_price)
            if dp > 0:
                wa += float(va) / dp

    den = wb + wa
    if den == 0:
        return None
    return (wb - wa) / den


def linear_slope(y: List[float]) -> Optional[float]:
    n = len(y)
    if n < 2:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    num = 0.0
    den = 0.0
    for i, yi in enumerate(y):
        dx = i - x_mean
        num += dx * (yi - y_mean)
        den += dx * dx
    if den == 0:
        return None
    return num / den


def second_diff_curvature(y: List[float]) -> Optional[float]:
    # simple discrete curvature proxy across levels
    n = len(y)
    if n < 3:
        return None
    vals = []
    for i in range(1, n - 1):
        vals.append(y[i + 1] - 2.0 * y[i] + y[i - 1])
    if not vals:
        return None
    return sum(vals) / len(vals)


def convexity_ratio(y: List[float]) -> Optional[float]:
    if len(y) < 4:
        return None
    near = sum(y[:3])
    far = sum(y[-3:])
    if far == 0:
        return None
    return near / far

def compute_book_snapshot_features_static(
    level_sz_bid: List[Optional[float]],
    level_sz_ask: List[Optional[float]],
) -> Dict[str, Optional[float]]:
    feats: Dict[str, Optional[float]] = {}

    feats["OBI_l1"] = compute_obi(level_sz_bid, level_sz_ask, 1)
    feats["OBI_l3"] = compute_obi(level_sz_bid, level_sz_ask, 3)
    feats["OBI_l5"] = compute_obi(level_sz_bid, level_sz_ask, 5)
    feats["OBI_l10"] = compute_obi(level_sz_bid, level_sz_ask, 10)

    feats["depth_near_bid"] = get_level_sum(level_sz_bid, NEAR_LEVELS)
    feats["depth_near_ask"] = get_level_sum(level_sz_ask, NEAR_LEVELS)
    feats["depth_mid_bid"] = get_level_sum(level_sz_bid, MID_LEVELS)
    feats["depth_mid_ask"] = get_level_sum(level_sz_ask, MID_LEVELS)
    feats["depth_far_bid"] = get_level_sum(level_sz_bid, FAR_LEVELS)
    feats["depth_far_ask"] = get_level_sum(level_sz_ask, FAR_LEVELS)

    bid_y = side_depths(level_sz_bid, 10)
    ask_y = side_depths(level_sz_ask, 10)

    feats["liquidity_slope_bid"] = linear_slope(bid_y)
    feats["liquidity_slope_ask"] = linear_slope(ask_y)

    bid_conv = convexity_ratio(bid_y)
    ask_conv = convexity_ratio(ask_y)
    feats["bid_convexity"] = bid_conv
    feats["ask_convexity"] = ask_conv
    feats["convexity_ratio_bid"] = bid_conv
    feats["convexity_ratio_ask"] = ask_conv
    feats["convexity_imbalance"] = None if (bid_conv is None or ask_conv is None) else (bid_conv - ask_conv)

    near_tot = (feats["depth_near_bid"] or 0.0) + (feats["depth_near_ask"] or 0.0)
    far_tot = (feats["depth_far_bid"] or 0.0) + (feats["depth_far_ask"] or 0.0)
    feats["liquidity_compression"] = safe_div(near_tot, far_tot)

    feats["bid_depth_curvature"] = second_diff_curvature(bid_y)
    feats["ask_depth_curvature"] = second_diff_curvature(ask_y)

    if feats["bid_depth_curvature"] is not None and feats["ask_depth_curvature"] is not None:
        feats["depth_curvature_pressure"] = feats["bid_depth_curvature"] - feats["ask_depth_curvature"]
    else:
        feats["depth_curvature_pressure"] = None

    bid_y5 = bid_y[:5]
    ask_y5 = ask_y[:5]
    feats["bid_curvature_l5"] = second_diff_curvature(bid_y5)
    feats["ask_curvature_l5"] = second_diff_curvature(ask_y5)

    if feats["bid_curvature_l5"] is not None and feats["ask_curvature_l5"] is not None:
        feats["curvature_pressure_l5"] = feats["bid_curvature_l5"] - feats["ask_curvature_l5"]
    else:
        feats["curvature_pressure_l5"] = None

    return feats

def compute_book_snapshot_features(
    entry_price: float,
    level_px_bid: List[Optional[float]],
    level_sz_bid: List[Optional[float]],
    level_px_ask: List[Optional[float]],
    level_sz_ask: List[Optional[float]],
) -> Dict[str, Optional[float]]:
    feats: Dict[str, Optional[float]] = {}

    # OBI
    feats["OBI_l1"] = compute_obi(level_sz_bid, level_sz_ask, 1)
    feats["OBI_l3"] = compute_obi(level_sz_bid, level_sz_ask, 3)
    feats["OBI_l5"] = compute_obi(level_sz_bid, level_sz_ask, 5)
    feats["OBI_l10"] = compute_obi(level_sz_bid, level_sz_ask, 10)

    # weighted OBI
    feats["weighted_OBI_l3"] = compute_weighted_obi(level_px_bid, level_sz_bid, level_px_ask, level_sz_ask, entry_price, 3)
    feats["weighted_OBI_l5"] = compute_weighted_obi(level_px_bid, level_sz_bid, level_px_ask, level_sz_ask, entry_price, 5)
    feats["weighted_OBI_l10"] = compute_weighted_obi(level_px_bid, level_sz_bid, level_px_ask, level_sz_ask, entry_price, 10)

    # bucket depths
    feats["depth_near_bid"] = get_level_sum(level_sz_bid, NEAR_LEVELS)
    feats["depth_near_ask"] = get_level_sum(level_sz_ask, NEAR_LEVELS)
    feats["depth_mid_bid"] = get_level_sum(level_sz_bid, MID_LEVELS)
    feats["depth_mid_ask"] = get_level_sum(level_sz_ask, MID_LEVELS)
    feats["depth_far_bid"] = get_level_sum(level_sz_bid, FAR_LEVELS)
    feats["depth_far_ask"] = get_level_sum(level_sz_ask, FAR_LEVELS)

    # liquidity slopes
    bid_y = side_depths(level_sz_bid, 10)
    ask_y = side_depths(level_sz_ask, 10)

    feats["liquidity_slope_bid"] = linear_slope(bid_y)
    feats["liquidity_slope_ask"] = linear_slope(ask_y)

    # convexity
    bid_conv = convexity_ratio(bid_y)
    ask_conv = convexity_ratio(ask_y)
    feats["bid_convexity"] = bid_conv
    feats["ask_convexity"] = ask_conv
    feats["convexity_ratio_bid"] = bid_conv
    feats["convexity_ratio_ask"] = ask_conv
    feats["convexity_imbalance"] = None if (bid_conv is None or ask_conv is None) else (bid_conv - ask_conv)

    # liquidity compression
    near_tot = (feats["depth_near_bid"] or 0.0) + (feats["depth_near_ask"] or 0.0)
    far_tot = (feats["depth_far_bid"] or 0.0) + (feats["depth_far_ask"] or 0.0)
    feats["liquidity_compression"] = safe_div(near_tot, far_tot)

    # curvature
    feats["bid_depth_curvature"] = second_diff_curvature(bid_y)
    feats["ask_depth_curvature"] = second_diff_curvature(ask_y)

    if feats["bid_depth_curvature"] is not None and feats["ask_depth_curvature"] is not None:
        feats["depth_curvature_pressure"] = feats["bid_depth_curvature"] - feats["ask_depth_curvature"]
    else:
        feats["depth_curvature_pressure"] = None

    bid_y5 = bid_y[:5]
    ask_y5 = ask_y[:5]
    feats["bid_curvature_l5"] = second_diff_curvature(bid_y5)
    feats["ask_curvature_l5"] = second_diff_curvature(ask_y5)
    if feats["bid_curvature_l5"] is not None and feats["ask_curvature_l5"] is not None:
        feats["curvature_pressure_l5"] = feats["bid_curvature_l5"] - feats["ask_curvature_l5"]
    else:
        feats["curvature_pressure_l5"] = None

    return feats


def wall_strength_opposite(
    side: str,
    level_px_bid: List[Optional[float]],
    level_sz_bid: List[Optional[float]],
    level_px_ask: List[Optional[float]],
    level_sz_ask: List[Optional[float]],
    entry_price: float,
    k_ticks: int,
) -> float:
    """
    Upper event -> ask wall above price
    Lower event -> bid wall below price
    """
    max_dist = k_ticks * TICK
    s = 0.0

    if side == "upper":
        for i in range(1, 11):
            px = level_px_ask[i]
            sz = level_sz_ask[i]
            if px is None or sz is None:
                continue
            d = float(px) - entry_price
            if 0.0 <= d <= max_dist:
                s += float(sz)
    else:
        for i in range(1, 11):
            px = level_px_bid[i]
            sz = level_sz_bid[i]
            if px is None or sz is None:
                continue
            d = entry_price - float(px)
            if 0.0 <= d <= max_dist:
                s += float(sz)

    return s


def compute_vpoc_and_value_area(
    vol_by_price: Dict[float, float],
    value_area_pct: float = VALUE_AREA_PCT,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns:
      vpoc, vah, val
    """
    if not vol_by_price:
        return None, None, None

    items = sorted(vol_by_price.items())
    prices = [p for p, _ in items]
    vols = [v for _, v in items]

    total_vol = sum(vols)
    if total_vol <= 0:
        return None, None, None

    vpoc_idx = max(range(len(vols)), key=lambda i: vols[i])
    vpoc = prices[vpoc_idx]

    target = value_area_pct * total_vol
    covered = vols[vpoc_idx]

    lo = hi = vpoc_idx

    while covered < target:
        left_vol = vols[lo - 1] if lo - 1 >= 0 else -1.0
        right_vol = vols[hi + 1] if hi + 1 < len(vols) else -1.0

        if left_vol < 0 and right_vol < 0:
            break
        elif right_vol > left_vol:
            hi += 1
            covered += vols[hi]
        else:
            lo -= 1
            covered += vols[lo]

    val = prices[lo]
    vah = prices[hi]
    return vpoc, vah, val


def add_null_v2_columns(r: Dict[str, Any]) -> None:
    r.update({
        "OBI_l1": None,
        "OBI_l3": None,
        "OBI_l5": None,
        "OBI_l10": None,
        "weighted_OBI_l3": None,
        "weighted_OBI_l5": None,
        "weighted_OBI_l10": None,
        "depth_near_bid": None,
        "depth_near_ask": None,
        "depth_mid_bid": None,
        "depth_mid_ask": None,
        "depth_far_bid": None,
        "depth_far_ask": None,
        "liquidity_slope_bid": None,
        "liquidity_slope_ask": None,
        "bid_convexity": None,
        "ask_convexity": None,
        "convexity_ratio_bid": None,
        "convexity_ratio_ask": None,
        "convexity_imbalance": None,
        "liquidity_compression": None,
        "bid_depth_curvature": None,
        "ask_depth_curvature": None,
        "depth_curvature_pressure": None,
        "bid_curvature_l5": None,
        "ask_curvature_l5": None,
        "curvature_pressure_l5": None,
        "wall_strength_opposite_3": None,
        "wall_strength_opposite_5": None,
        "wall_strength_opposite_3_2s_avg": None,
        "wall_strength_opposite_3_5s_avg": None,
        "wall_strength_opposite_5_2s_avg": None,
        "wall_strength_opposite_5_5s_avg": None,
        "wall_persistence_2s": None,
        "vpoc": None,
        "vah": None,
        "val": None,
        "distance_from_vpoc": None,
        "distance_from_value_area_high": None,
        "distance_from_value_area_low": None,
        "inside_value_area_flag": None,
    })


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

    required = {"event_id", "date_ymd", "ts", "entry_price", "side"}
    missing = [c for c in required if c not in df_events.columns]
    if missing:
        raise ValueError(f"Base events parquet missing required columns: {missing}")

    rows_out: List[Dict[str, Any]] = []
    event_days = sorted(df_events.select("date_ymd").unique().to_series().to_list())

    for idx_day, day_ymd in enumerate(event_days, start=1):
        df_day_events = (
            df_events
            .filter(pl.col("date_ymd") == day_ymd)
            .sort("ts")
        )

        full_path = day_to_full.get(day_ymd)

        if full_path is None or not full_path.exists():
            for r in df_day_events.to_dicts():
                add_null_v2_columns(r)
                rows_out.append(r)
            continue

        # trades
        df_tr = load_trade_day(full_path)
        tr_ts = df_tr["ts"].to_list()
        tr_us = [dt_to_us(x) for x in tr_ts]
        tr_px = [float(x) for x in df_tr["price"].to_list()]
        tr_vol = [float(x) for x in df_tr["volume"].to_list()]

        # l2
        df_l2 = load_l2_day(full_path)
        l2_ts_col = df_l2["ts"].to_list()
        l2_side_col = df_l2["side"].to_list()
        l2_depth_col = df_l2["depth"].to_list()
        l2_price_col = df_l2["price"].to_list()
        l2_volume_col = df_l2["volume"].to_list()
        l2_action_col = df_l2["action"].to_list()

        if df_l2.height == 0:
            for r in df_day_events.to_dicts():
                add_null_v2_columns(r)
                rows_out.append(r)
            continue

        if df_tr.height == 0:
            for r in df_day_events.to_dicts():
                add_null_v2_columns(r)
                rows_out.append(r)
            continue

        if idx_day <= 2:
            print(f"\nDEBUG DAY {day_ymd}")
            print("trade rows:", df_tr.height)
            print("l2 rows:", df_l2.height)

        # Build cumulative session-to-event volume profile on trades
        # We advance this pointer as events move forward through time.
        vol_by_price: Dict[float, float] = defaultdict(float)
        trade_ptr = 0

        # Reconstruct book state across L2 updates
        # depth indexed 1..10 for convenience
        bid_px: List[Optional[float]] = [None] * 11
        bid_sz: List[Optional[float]] = [None] * 11
        ask_px: List[Optional[float]] = [None] * 11
        ask_sz: List[Optional[float]] = [None] * 11

        l2_ts: List[datetime] = []
        l2_us: List[int] = []

        # snapshot caches after each update
        snapshots_bid_px: List[List[Optional[float]]] = []
        snapshots_bid_sz: List[List[Optional[float]]] = []
        snapshots_ask_px: List[List[Optional[float]]] = []
        snapshots_ask_sz: List[List[Optional[float]]] = []
        snapshot_static_feats: List[Dict[str, Optional[float]]] = []

        # Need a reference price for walls during book history construction.
        # Use mid if available, else best available side.
        def approx_ref_price() -> Optional[float]:
            bb = bid_px[1]
            ba = ask_px[1]
            if bb is not None and ba is not None:
                return 0.5 * (float(bb) + float(ba))
            if bb is not None:
                return float(bb)
            if ba is not None:
                return float(ba)
            return None

        for ts, side, depth_raw, price_raw, volume_raw, action_raw in zip(
            l2_ts_col, l2_side_col, l2_depth_col, l2_price_col, l2_volume_col, l2_action_col
        ):
            t_us = dt_to_us(ts)

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
            elif side == "ask":
                if action == 2 or volume is None or volume <= 0:
                    ask_px[depth] = None
                    ask_sz[depth] = None
                else:
                    ask_px[depth] = price
                    ask_sz[depth] = volume

            ref = approx_ref_price()
            if ref is None:
                w3 = 0.0
                w5 = 0.0
            else:
                # upper-style reference for generic book history; event-specific averaging will use these as generic resisting-continuation proxies around price.
                # We store ask-above-price generic wall for 3/5, then compute event-specific averages later from event-specific function if needed.
                w3 = 0.0
                w5 = 0.0
                for i in range(1, 11):
                    pxa = ask_px[i]
                    sza = ask_sz[i]
                    if pxa is not None and sza is not None:
                        d = float(pxa) - ref
                        if 0.0 <= d <= 3 * TICK:
                            w3 += float(sza)
                        if 0.0 <= d <= 5 * TICK:
                            w5 += float(sza)

            l2_ts.append(ts)
            l2_us.append(t_us)
            snapshots_bid_px.append(bid_px.copy())
            snapshots_bid_sz.append(bid_sz.copy())
            snapshots_ask_px.append(ask_px.copy())
            snapshots_ask_sz.append(ask_sz.copy())

            snapshot_static_feats.append(
                compute_book_snapshot_features_static(
                    level_sz_bid=bid_sz,
                    level_sz_ask=ask_sz,
                )
            )

        # For event-level processing, walk events in time order
        for r in df_day_events.to_dicts():
            t0: datetime = r["ts"]
            t0_us = dt_to_us(t0)
            entry_price = float(r["entry_price"])
            ev_side = str(r["side"])

            # advance session profile only with information available by event time
            while trade_ptr < len(tr_us) and tr_us[trade_ptr] <= t0_us:
                px = tr_px[trade_ptr]
                vol = tr_vol[trade_ptr]
                if RTH_OPEN <= tr_ts[trade_ptr].time() <= RTH_CLOSE:
                    vol_by_price[px] += vol
                trade_ptr += 1

            # nearest L2 snapshot at or before t0
            j = bisect_right(l2_us, t0_us) - 1
            if j < 0:
                add_null_v2_columns(r)
                rows_out.append(r)
                continue

            bpx = snapshots_bid_px[j]
            bsz = snapshots_bid_sz[j]
            apx = snapshots_ask_px[j]
            asz = snapshots_ask_sz[j]



            # geometry snapshot features
            snap_feats = dict(snapshot_static_feats[j])
            snap_feats["weighted_OBI_l3"] = compute_weighted_obi(bpx, bsz, apx, asz, entry_price, 3)
            snap_feats["weighted_OBI_l5"] = compute_weighted_obi(bpx, bsz, apx, asz, entry_price, 5)
            snap_feats["weighted_OBI_l10"] = compute_weighted_obi(bpx, bsz, apx, asz, entry_price, 10)

            # event-specific walls at snapshot
            wall3 = wall_strength_opposite(ev_side, bpx, bsz, apx, asz, entry_price, 3)
            wall5 = wall_strength_opposite(ev_side, bpx, bsz, apx, asz, entry_price, 5)

            # short window averages from L2 history
            # For wall averages, use event-time re-evaluation across snapshots in window.
            # Since that's more expensive, keep it correct rather than over-clever.
            def avg_wall_over_window(window_s: int, k_ticks: int) -> Optional[float]:
                start_us = t0_us - int(window_s * 1_000_000)
                i0 = bisect_left(l2_us, start_us)
                i1 = bisect_right(l2_us, t0_us)
                if i1 <= i0:
                    return None
                vals = []
                for q in range(i0, i1):
                    vals.append(
                        wall_strength_opposite(
                            ev_side,
                            snapshots_bid_px[q],
                            snapshots_bid_sz[q],
                            snapshots_ask_px[q],
                            snapshots_ask_sz[q],
                            entry_price,
                            k_ticks,
                        )
                    )
                if not vals:
                    return None
                return sum(vals) / len(vals)

            wall3_2s = avg_wall_over_window(2, 3)
            wall3_5s = avg_wall_over_window(5, 3)
            wall5_2s = avg_wall_over_window(2, 5)
            wall5_5s = avg_wall_over_window(5, 5)

            # wall persistence over last 2s using 3-tick wall vs local average baseline
            wall_persistence_2s = None
            start_us = t0_us - int(2 * 1_000_000)
            i0 = bisect_left(l2_us, start_us)
            i1 = bisect_right(l2_us, t0_us)
            if i1 > i0:
                vals = []
                for q in range(i0, i1):
                    vals.append(
                        wall_strength_opposite(
                            ev_side,
                            snapshots_bid_px[q],
                            snapshots_bid_sz[q],
                            snapshots_ask_px[q],
                            snapshots_ask_sz[q],
                            entry_price,
                            3,
                        )
                    )
                if vals:
                    baseline = sum(vals) / len(vals)
                    thr = 1.25 * baseline
                    if baseline > 0:
                        wall_persistence_2s = sum(1 for v in vals if v > thr) / len(vals)
                    else:
                        wall_persistence_2s = 0.0

            # session-to-event profile features
            vpoc, vah, val = compute_vpoc_and_value_area(vol_by_price)

            distance_from_vpoc = None if vpoc is None else entry_price - vpoc
            distance_from_value_area_high = None if vah is None else entry_price - vah
            distance_from_value_area_low = None if val is None else entry_price - val
            inside_value_area_flag = None
            if vah is not None and val is not None:
                inside_value_area_flag = 1 if (val <= entry_price <= vah) else 0

            r.update(snap_feats)
            r.update({
                "wall_strength_opposite_3": wall3,
                "wall_strength_opposite_5": wall5,
                "wall_strength_opposite_3_2s_avg": wall3_2s,
                "wall_strength_opposite_3_5s_avg": wall3_5s,
                "wall_strength_opposite_5_2s_avg": wall5_2s,
                "wall_strength_opposite_5_5s_avg": wall5_5s,
                "wall_persistence_2s": wall_persistence_2s,
                "vpoc": vpoc,
                "vah": vah,
                "val": val,
                "distance_from_vpoc": distance_from_vpoc,
                "distance_from_value_area_high": distance_from_value_area_high,
                "distance_from_value_area_low": distance_from_value_area_low,
                "inside_value_area_flag": inside_value_area_flag,
            })
            rows_out.append(r)

        if idx_day % 25 == 0:
            print(f"...processed {idx_day} event days")

    if not rows_out:
        print("No rows produced.")
        return

    df_out = pl.from_dicts(rows_out).sort(["date_ymd", "ts", "band_k", "side"])

    float_cols = [
        "OBI_l1", "OBI_l3", "OBI_l5", "OBI_l10",
        "weighted_OBI_l3", "weighted_OBI_l5", "weighted_OBI_l10",
        "depth_near_bid", "depth_near_ask", "depth_mid_bid", "depth_mid_ask",
        "depth_far_bid", "depth_far_ask",
        "liquidity_slope_bid", "liquidity_slope_ask",
        "bid_convexity", "ask_convexity",
        "convexity_ratio_bid", "convexity_ratio_ask", "convexity_imbalance",
        "liquidity_compression",
        "bid_depth_curvature", "ask_depth_curvature",
        "depth_curvature_pressure",
        "bid_curvature_l5", "ask_curvature_l5", "curvature_pressure_l5",
        "wall_strength_opposite_3", "wall_strength_opposite_5",
        "wall_strength_opposite_3_2s_avg", "wall_strength_opposite_3_5s_avg",
        "wall_strength_opposite_5_2s_avg", "wall_strength_opposite_5_5s_avg",
        "wall_persistence_2s",
        "vpoc", "vah", "val",
        "distance_from_vpoc",
        "distance_from_value_area_high",
        "distance_from_value_area_low",
        "inside_value_area_flag",
    ]

    cast_exprs = []
    for c in float_cols:
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
        "OBI_l1",
        "weighted_OBI_l3",
        "wall_strength_opposite_3",
        "wall_strength_opposite_3_2s_avg",
        "distance_from_vpoc",
        "distance_from_value_area_high",
        "inside_value_area_flag",
    ] if c in df_out.columns]

    exprs = [pl.len().alias("n")] + [
        pl.col(c).null_count().alias(f"{c}_nulls") for c in sanity_cols
    ]
    print(df_out.select(exprs))


if __name__ == "__main__":
    main()
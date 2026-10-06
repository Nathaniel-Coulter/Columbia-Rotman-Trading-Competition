# build_touch_events_reversion_targets.py
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import timedelta, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import polars as pl

# -------------------------
# INPUTS
# -------------------------
TRADES_ROOT = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\es_t2_trades_parquet"
)

# -------------------------
# OUTPUTS
# -------------------------
OUT_DIR = Path(
    r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\futures_research\data\processed\MR_es_t2_touch_events"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "touch_reversion_targets.parquet"

# -------------------------
# TOUCH / FEATURE SETTINGS
# -------------------------
BANDS = [1.0, 2.0, 3.0]

COOLDOWN_MINUTES = 15  # must be >= this long since last ACCEPTED event for that (k,side)
REARM_SECONDS = 0      # require continuous "inside" time before allowing next touch; set 0 to disable

ROLLING_WINDOW_TRADES = 5000
SIGMA_MIN_SAMPLES = 100  # Polars renamed min_periods -> min_samples

TIME_HORIZONS_S = [1,3,5,10,15,30,60,120,180,300,600,900]
TRADE_HORIZONS_N = [10, 25, 50, 100, 250, 500, 1000, 2500]


# -------------------------
# HELPERS
# -------------------------
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


def compute_session_vwap_and_sigma(df: pl.DataFrame) -> pl.DataFrame:
    """
    Session-anchored VWAP (cum pv / cum v) + rolling sigma.
    Uses residual sigma (price - vwap) primarily, falls back to sigma_price.
    """
    df = df.with_columns([
        (pl.col("price") * pl.col("volume")).alias("pv"),
        pl.col("volume").alias("v"),
    ]).with_columns([
        pl.col("pv").cum_sum().alias("cum_pv"),
        pl.col("v").cum_sum().alias("cum_v"),
    ]).with_columns([
        (pl.col("cum_pv") / pl.col("cum_v")).alias("vwap"),
    ])

    df = df.with_columns([
        pl.col("price").rolling_std(
            window_size=ROLLING_WINDOW_TRADES,
            min_samples=SIGMA_MIN_SAMPLES
        ).alias("sigma_price"),
    ])

    # Create resid in its own call so it can be referenced safely
    df = df.with_columns([
        (pl.col("price") - pl.col("vwap")).alias("resid"),
    ]).with_columns([
        pl.col("resid").rolling_std(
            window_size=ROLLING_WINDOW_TRADES,
            min_samples=SIGMA_MIN_SAMPLES
        ).alias("sigma_resid"),
    ])

    df = df.with_columns([
        pl.when(pl.col("sigma_resid").is_not_null() & (pl.col("sigma_resid") > 0))
          .then(pl.col("sigma_resid"))
          .otherwise(pl.col("sigma_price"))
          .alias("sigma")
    ])

    return df


def dt_to_us(dt) -> int:
    # dt is python datetime
    return int(dt.timestamp() * 1_000_000)


@dataclass
class BandState:
    # transition detection
    in_touch: bool = False

    # accepted-event clock (cross-day)
    last_event_us: Optional[int] = None

    # touch-any clock (cross-day; updates on any touch, even if not accepted)
    last_touch_any_us: Optional[int] = None

    # optional inside tracking if REARM_SECONDS > 0
    inside_start_us: Optional[int] = None

    # global monotonic event counter (for IDs)
    event_idx: int = 0


def init_global_state() -> Dict[Tuple[float, str], BandState]:
    state: Dict[Tuple[float, str], BandState] = {}
    for k in BANDS:
        state[(k, "upper")] = BandState()
        state[(k, "lower")] = BandState()
    return state


def build_cooldown_touch_events_day(
    ts: List,          # datetime
    ts_us: List[int],  # microseconds
    price: List[float],
    vwap: List[float],
    sigma: List[float],
    year: int,
    month: int,
    day: int,
    state: Dict[Tuple[float, str], BandState],
) -> pl.DataFrame:
    """
    Build events for ONE day, but uses/updates GLOBAL state dict so:
      - cooldown_seconds is cross-day
      - since_touch_any_seconds is cross-day
      - transition detection (in_touch) is cross-day
    VWAP/sigma themselves are session/day anchored (because arrays are per-day).
    """
    n = len(ts)
    if n == 0:
        return pl.DataFrame([])

    cooldown_us = COOLDOWN_MINUTES * 60 * 1_000_000
    rearm_us = REARM_SECONDS * 1_000_000

    out: List[Dict[str, Any]] = []

    rearm_enabled = REARM_SECONDS > 0

    for i in range(n):
        t_us = ts_us[i]
        p = price[i]
        vw = vwap[i]
        sg = sigma[i]

        for k in BANDS:
            up_thr = vw + k * sg
            lo_thr = vw - k * sg

            # -------------------
            # UPPER SIDE
            # -------------------
            st = state[(k, "upper")]
            is_touch = p >= up_thr
            is_inside = p < up_thr

            # update last_touch_any on ANY touch (even if we reject the event)
            if is_touch:
                st.last_touch_any_us = t_us

            # inside tracking only if rearm enabled
            if rearm_enabled:
                if is_inside:
                    if st.inside_start_us is None:
                        st.inside_start_us = t_us
                    st.in_touch = False
                else:
                    had_rearm_run = (
                        st.inside_start_us is not None
                        and (t_us - st.inside_start_us) >= rearm_us
                    )
                    entering_touch = is_touch and (not st.in_touch)

                    if entering_touch:
                        can_cooldown = (
                            st.last_event_us is None
                            or (t_us - st.last_event_us) >= cooldown_us
                        )
                        can_rearm = had_rearm_run
                        if can_cooldown and can_rearm:
                            prev_event = st.last_event_us
                            cooldown_seconds = None if prev_event is None else (t_us - prev_event) / 1_000_000.0

                            prev_touch_any = st.last_touch_any_us
                            # NOTE: we updated last_touch_any_us above if is_touch; for "since touch-any"
                            # we want the *previous* touch-any before this touch.
                            # So compute from a saved value:
                            # We'll handle by reading prev before overwriting on touch.
                        # but we already overwrote if is_touch; so do touch-any timing carefully:
                        # (see optimized approach below: compute prev_touch_any BEFORE overwrite)
                        # We'll handle by restructuring in a second pass below.
                    st.in_touch = bool(is_touch)
                    st.inside_start_us = None
            else:
                # rearm disabled: simplest transition/cooldown logic
                if is_inside:
                    st.in_touch = False
                else:
                    entering_touch = is_touch and (not st.in_touch)
                    if entering_touch:
                        can_cooldown = (
                            st.last_event_us is None
                            or (t_us - st.last_event_us) >= cooldown_us
                        )
                        if can_cooldown:
                            prev_event = st.last_event_us
                            cooldown_seconds = None if prev_event is None else (t_us - prev_event) / 1_000_000.0

                            # for touch-any, we need prev value BEFORE we set last_touch_any_us this tick
                            # Since we already updated it above, we must reconstruct:
                            # We'll compute prev_touch_any by using a temp variable set before overwrite.
                            # (handled below by rewriting upper/lower logic with prev vars)
                            pass
                    st.in_touch = bool(is_touch)

            # -------------------
            # LOWER SIDE
            # -------------------
            # (Same issue: touch-any "prev" should be measured before overwrite)
            # We'll replace the above block with a clean implementation right after this loop.
            # To keep this file easy to paste/run, we implement the final clean version below.
            #
            # This placeholder prevents partial logic from running.
            #
            raise RuntimeError(
                "Internal placeholder reached. Please use the final implementation below."
            )

    # unreachable
    # return pl.from_dicts(out).sort(["ts", "band_k", "side"])


# ---- FINAL (clean) implementation of event builder with correct prev-touch tracking ----
def build_cooldown_touch_events_day(
    ts: List,          # datetime
    ts_us: List[int],  # microseconds
    price: List[float],
    vwap: List[float],
    sigma: List[float],
    year: int,
    month: int,
    day: int,
    state: Dict[Tuple[float, str], BandState],
) -> pl.DataFrame:
    n = len(ts)
    if n == 0:
        return pl.DataFrame([])

    cooldown_us = COOLDOWN_MINUTES * 60 * 1_000_000
    rearm_us = REARM_SECONDS * 1_000_000
    rearm_enabled = REARM_SECONDS > 0

    out: List[Dict[str, Any]] = []

    for i in range(n):
        t = ts[i]
        t_us = ts_us[i]
        p = price[i]
        vw = vwap[i]
        sg = sigma[i]

        for k in BANDS:
            up_thr = vw + k * sg
            lo_thr = vw - k * sg

            # ===================
            # UPPER
            # ===================
            st_u = state[(k, "upper")]
            is_touch_u = p >= up_thr
            is_inside_u = p < up_thr

            prev_touch_any_u = st_u.last_touch_any_us  # capture BEFORE overwrite
            if is_touch_u:
                st_u.last_touch_any_us = t_us

            if rearm_enabled:
                if is_inside_u:
                    if st_u.inside_start_us is None:
                        st_u.inside_start_us = t_us
                    st_u.in_touch = False
                else:
                    had_rearm_run = (
                        st_u.inside_start_us is not None
                        and (t_us - st_u.inside_start_us) >= rearm_us
                    )
                    entering_touch = is_touch_u and (not st_u.in_touch)
                    if entering_touch:
                        can_cooldown = (
                            st_u.last_event_us is None
                            or (t_us - st_u.last_event_us) >= cooldown_us
                        )
                        can_rearm = had_rearm_run
                        if can_cooldown and can_rearm:
                            prev_event = st_u.last_event_us
                            cooldown_seconds = None if prev_event is None else (t_us - prev_event) / 1_000_000.0
                            since_touch_any_seconds = None if prev_touch_any_u is None else (t_us - prev_touch_any_u) / 1_000_000.0

                            st_u.event_idx += 1
                            st_u.last_event_us = t_us

                            out.append({
                                "ts": t,
                                "year": year, "month": month, "day": day,
                                "band_k": float(k),
                                "side": "upper",
                                "event_n": int(st_u.event_idx),
                                "entry_price": float(p),
                                "vwap0": float(vw),
                                "sigma0": float(sg),
                                "distance0": float(p - vw),
                                "cooldown_seconds": None if cooldown_seconds is None else float(cooldown_seconds),
                                "since_touch_any_seconds": None if since_touch_any_seconds is None else float(since_touch_any_seconds),
                            })
                    st_u.in_touch = bool(is_touch_u)
                    st_u.inside_start_us = None
            else:
                # no rearm
                if is_inside_u:
                    st_u.in_touch = False
                else:
                    entering_touch = is_touch_u and (not st_u.in_touch)
                    if entering_touch:
                        can_cooldown = (
                            st_u.last_event_us is None
                            or (t_us - st_u.last_event_us) >= cooldown_us
                        )
                        if can_cooldown:
                            prev_event = st_u.last_event_us
                            cooldown_seconds = None if prev_event is None else (t_us - prev_event) / 1_000_000.0
                            since_touch_any_seconds = None if prev_touch_any_u is None else (t_us - prev_touch_any_u) / 1_000_000.0

                            st_u.event_idx += 1
                            st_u.last_event_us = t_us

                            out.append({
                                "ts": t,
                                "year": year, "month": month, "day": day,
                                "band_k": float(k),
                                "side": "upper",
                                "event_n": int(st_u.event_idx),
                                "entry_price": float(p),
                                "vwap0": float(vw),
                                "sigma0": float(sg),
                                "distance0": float(p - vw),
                                "cooldown_seconds": None if cooldown_seconds is None else float(cooldown_seconds),
                                "since_touch_any_seconds": None if since_touch_any_seconds is None else float(since_touch_any_seconds),
                            })
                    st_u.in_touch = bool(is_touch_u)

            # ===================
            # LOWER
            # ===================
            st_l = state[(k, "lower")]
            is_touch_l = p <= lo_thr
            is_inside_l = p > lo_thr

            prev_touch_any_l = st_l.last_touch_any_us  # capture BEFORE overwrite
            if is_touch_l:
                st_l.last_touch_any_us = t_us

            if rearm_enabled:
                if is_inside_l:
                    if st_l.inside_start_us is None:
                        st_l.inside_start_us = t_us
                    st_l.in_touch = False
                else:
                    had_rearm_run = (
                        st_l.inside_start_us is not None
                        and (t_us - st_l.inside_start_us) >= rearm_us
                    )
                    entering_touch = is_touch_l and (not st_l.in_touch)
                    if entering_touch:
                        can_cooldown = (
                            st_l.last_event_us is None
                            or (t_us - st_l.last_event_us) >= cooldown_us
                        )
                        can_rearm = had_rearm_run
                        if can_cooldown and can_rearm:
                            prev_event = st_l.last_event_us
                            cooldown_seconds = None if prev_event is None else (t_us - prev_event) / 1_000_000.0
                            since_touch_any_seconds = None if prev_touch_any_l is None else (t_us - prev_touch_any_l) / 1_000_000.0

                            st_l.event_idx += 1
                            st_l.last_event_us = t_us

                            out.append({
                                "ts": t,
                                "year": year, "month": month, "day": day,
                                "band_k": float(k),
                                "side": "lower",
                                "event_n": int(st_l.event_idx),
                                "entry_price": float(p),
                                "vwap0": float(vw),
                                "sigma0": float(sg),
                                "distance0": float(p - vw),
                                "cooldown_seconds": None if cooldown_seconds is None else float(cooldown_seconds),
                                "since_touch_any_seconds": None if since_touch_any_seconds is None else float(since_touch_any_seconds),
                            })
                    st_l.in_touch = bool(is_touch_l)
                    st_l.inside_start_us = None
            else:
                if is_inside_l:
                    st_l.in_touch = False
                else:
                    entering_touch = is_touch_l and (not st_l.in_touch)
                    if entering_touch:
                        can_cooldown = (
                            st_l.last_event_us is None
                            or (t_us - st_l.last_event_us) >= cooldown_us
                        )
                        if can_cooldown:
                            prev_event = st_l.last_event_us
                            cooldown_seconds = None if prev_event is None else (t_us - prev_event) / 1_000_000.0
                            since_touch_any_seconds = None if prev_touch_any_l is None else (t_us - prev_touch_any_l) / 1_000_000.0

                            st_l.event_idx += 1
                            st_l.last_event_us = t_us

                            out.append({
                                "ts": t,
                                "year": year, "month": month, "day": day,
                                "band_k": float(k),
                                "side": "lower",
                                "event_n": int(st_l.event_idx),
                                "entry_price": float(p),
                                "vwap0": float(vw),
                                "sigma0": float(sg),
                                "distance0": float(p - vw),
                                "cooldown_seconds": None if cooldown_seconds is None else float(cooldown_seconds),
                                "since_touch_any_seconds": None if since_touch_any_seconds is None else float(since_touch_any_seconds),
                            })
                    st_l.in_touch = bool(is_touch_l)

    if not out:
        return pl.DataFrame([])

    return pl.from_dicts(out).sort(["ts", "band_k", "side"])


# -------------------------
# OUTCOMES (FAST, EXACT)
# -------------------------
def compute_outcomes_for_event_fast(
    ts_list: List,            # datetime
    price_list: List[float],
    vwap_list: List[float],
    i0: int,
    side: str,
) -> Dict[str, Any]:
    """
    Exact forward mean-reversion outcomes using python arrays.

    Definitions:
      upper touch -> reversion = move DOWN from entry toward VWAP
      lower touch -> reversion = move UP from entry toward VWAP

    Outputs:
      - exact first VWAP hit time/print after entry (unbounded)
      - mr_pts_t{H}: max favorable move toward the mean within time horizon H
      - hit_vwap_t{H}: whether VWAP was hit within time horizon H
      - mr_pts_n{N}: max favorable move toward the mean within next N prints
      - hit_vwap_n{N}: whether VWAP was hit within next N prints
    """
    n = len(ts_list)
    out: Dict[str, Any] = {}

    entry_price = float(price_list[i0])
    t0 = ts_list[i0]

    # -------------------------
    # Horizon end indices
    # -------------------------
    time_end_idx: Dict[int, int] = {}
    max_i1_time = i0 + 1
    for H in TIME_HORIZONS_S:
        t_end = t0 + timedelta(seconds=H)
        i1 = bisect_left(ts_list, t_end)
        if i1 < i0 + 1:
            i1 = i0 + 1
        if i1 > n:
            i1 = n
        time_end_idx[H] = i1
        if i1 > max_i1_time:
            max_i1_time = i1

    trade_end_idx: Dict[int, int] = {}
    max_i1_trade = i0 + 1
    for N in TRADE_HORIZONS_N:
        i1 = i0 + N
        if i1 > n:
            i1 = n
        if i1 < i0 + 1:
            i1 = i0 + 1
        trade_end_idx[N] = i1
        if i1 > max_i1_trade:
            max_i1_trade = i1

    # -------------------------
    # Exact first VWAP hit (unbounded)
    # -------------------------
    hit_idx = None
    if side == "upper":
        for j in range(i0, n):
            if price_list[j] <= vwap_list[j]:
                hit_idx = j
                break
    else:
        for j in range(i0, n):
            if price_list[j] >= vwap_list[j]:
                hit_idx = j
                break

    if hit_idx is None:
        out["time_to_vwap_s"] = None
        out["trades_to_vwap"] = None
    else:
        out["time_to_vwap_s"] = float((ts_list[hit_idx] - t0).total_seconds())
        out["trades_to_vwap"] = int(hit_idx - i0)

    # -------------------------
    # TIME horizons
    # -------------------------
    running_min = entry_price
    running_max = entry_price

    hit_vwap_time: Dict[int, bool] = {H: False for H in TIME_HORIZONS_S}

    end_to_H: Dict[int, List[int]] = {}
    for H, i1 in time_end_idx.items():
        end_to_H.setdefault(i1, []).append(H)

    for j in range(i0, max_i1_time):
        pj = float(price_list[j])
        running_min = pj if pj < running_min else running_min
        running_max = pj if pj > running_max else running_max

        if side == "upper":
            if pj <= float(vwap_list[j]):
                for H in TIME_HORIZONS_S:
                    hit_vwap_time[H] = True
        else:
            if pj >= float(vwap_list[j]):
                for H in TIME_HORIZONS_S:
                    hit_vwap_time[H] = True

        boundary = j + 1
        if boundary in end_to_H:
            for H in end_to_H[boundary]:
                if boundary - i0 <= 1:
                    out[f"mr_pts_t{H}"] = None
                    out[f"hit_vwap_t{H}"] = None
                    continue

                if side == "upper":
                    mr = entry_price - running_min
                else:
                    mr = running_max - entry_price

                out[f"mr_pts_t{H}"] = float(mr)
                out[f"hit_vwap_t{H}"] = bool(hit_vwap_time[H])

    # -------------------------
    # TRADE horizons
    # -------------------------
    running_min = entry_price
    running_max = entry_price
    hit_vwap_trade: Dict[int, bool] = {N: False for N in TRADE_HORIZONS_N}

    end_to_N: Dict[int, List[int]] = {}
    for N, i1 in trade_end_idx.items():
        end_to_N.setdefault(i1, []).append(N)

    for j in range(i0, max_i1_trade):
        pj = float(price_list[j])
        running_min = pj if pj < running_min else running_min
        running_max = pj if pj > running_max else running_max

        if side == "upper":
            if pj <= float(vwap_list[j]):
                for N in TRADE_HORIZONS_N:
                    hit_vwap_trade[N] = True
        else:
            if pj >= float(vwap_list[j]):
                for N in TRADE_HORIZONS_N:
                    hit_vwap_trade[N] = True

        boundary = j + 1
        if boundary in end_to_N:
            for N in end_to_N[boundary]:
                if boundary - i0 <= 1:
                    out[f"mr_pts_n{N}"] = None
                    out[f"hit_vwap_n{N}"] = None
                    continue

                if side == "upper":
                    mr = entry_price - running_min
                else:
                    mr = running_max - entry_price

                out[f"mr_pts_n{N}"] = float(mr)
                out[f"hit_vwap_n{N}"] = bool(hit_vwap_trade[N])

    return out

# -------------------------
# MAIN
# -------------------------
def main():
    day_files = list(iter_day_trade_parquets(TRADES_ROOT))
    print(f"Found {len(day_files)} trades day partitions under {TRADES_ROOT}")

    state = init_global_state()

    all_rows: List[Dict[str, Any]] = []
    days_done = 0

    # “Processed” here means: we actually attempted event/outcome generation for that day
    # (i.e., not skipped for small/no-sigma).
    days_with_zero_events = 0

    skipped_small: List[str] = []
    skipped_no_sigma: List[str] = []
    skipped_sigma_nonpos: List[str] = []

    for year, month, day, p in day_files:
        df = pl.read_parquet(p)

        is_weekend = date(year, month, day).weekday() >= 5  # Sat/Sun True

        if df.height < 500:
            skipped_small.append(f"{year:04d}-{month:02d}-{day:02d}")
            continue

        # ensure sorted
        df = df.sort("ts")

        df_feat = compute_session_vwap_and_sigma(df.select(["ts", "price", "volume"]))

        # any non-null sigma at all?
        if df_feat.select(pl.col("sigma").is_not_null().any()).item() is False:
            skipped_no_sigma.append(f"{year:04d}-{month:02d}-{day:02d}")
            continue

        # any strictly-positive sigma at all?
        if df_feat.select((pl.col("sigma") > 0).any()).item() is False:
            skipped_sigma_nonpos.append(f"{year:04d}-{month:02d}-{day:02d}")
            continue

        # filter rows where sigma is defined/positive
        df2 = (
            df_feat
            .filter(pl.col("sigma").is_not_null() & (pl.col("sigma") > 0))
            .select(["ts", "price", "vwap", "sigma"])
        )

        # extremely rare safety net (should be redundant given the "any sigma>0" check)
        if df2.height == 0:
            skipped_sigma_nonpos.append(f"{year:04d}-{month:02d}-{day:02d}")
            continue

        # convert ONCE per day (no repeated to_list anywhere)
        ts_list = df2["ts"].to_list()
        ts_us_list = [dt_to_us(t) for t in ts_list]
        price_list = [float(x) for x in df2["price"].to_list()]
        vwap_list = [float(x) for x in df2["vwap"].to_list()]
        sigma_list = [float(x) for x in df2["sigma"].to_list()]

        # build events for the day (cross-day stateful)
        df_events = build_cooldown_touch_events_day(
            ts=ts_list,
            ts_us=ts_us_list,
            price=price_list,
            vwap=vwap_list,
            sigma=sigma_list,
            year=year,
            month=month,
            day=day,
            state=state,
        )

        if df_events.height == 0:
            days_with_zero_events += 1
        else:
            # outcomes: use fast arrays and bisect (exact)
            # events are built on df2 so indices align with ts_list
            for ev in df_events.iter_rows(named=True):
                t0 = ev["ts"]
                i0 = bisect_left(ts_list, t0)
                if i0 >= len(ts_list):
                    continue

                outcomes = compute_outcomes_for_event_fast(
                    ts_list=ts_list,
                    price_list=price_list,
                    vwap_list=vwap_list,
                    i0=i0,
                    side=ev["side"],
                )

                row = dict(ev)
                row.update(outcomes)

                row["date_ymd"] = f"{year:04d}-{month:02d}-{day:02d}"
                row["is_weekend"] = is_weekend
                row["event_id"] = (
                    f"{year:04d}{month:02d}{day:02d}_{ev['side']}_k{ev['band_k']}_"
                    f"{ev['event_n']}_{t0.isoformat()}"
                )
                all_rows.append(row)

        days_done += 1
        if days_done % 25 == 0:
            print(f"...processed {days_done} days")

    if not all_rows:
        print("No outcomes produced (no events).")
        return

    # -------------------------
    # BUILD OUTPUT DF (KEEP ALL KEYS)
    # -------------------------
    df_out = pl.from_dicts(all_rows)  # keep ALL keys (including mfe/mae/etc)

    # Ensure all horizon columns exist (only fills missing cols, does NOT overwrite real ones)
    expected_cols: List[str] = []
    expected_cols += [f"mr_pts_t{H}" for H in TIME_HORIZONS_S]
    expected_cols += [f"hit_vwap_t{H}" for H in TIME_HORIZONS_S]
    expected_cols += [f"mr_pts_n{N}" for N in TRADE_HORIZONS_N]
    expected_cols += [f"hit_vwap_n{N}" for N in TRADE_HORIZONS_N]

    missing = [c for c in expected_cols if c not in df_out.columns]
    if missing:
        df_out = df_out.with_columns([pl.lit(None).alias(c) for c in missing])

    # Cast core columns (do AFTER building so we don’t drop outcome keys)
    core_casts = [
        ("event_id", pl.Utf8),
        ("date_ymd", pl.Utf8),
        ("ts", pl.Datetime),
        ("year", pl.Int32),
        ("month", pl.Int32),
        ("day", pl.Int32),
        ("is_weekend", pl.Boolean),
        ("band_k", pl.Float64),
        ("side", pl.Utf8),
        ("event_n", pl.Int32),
        ("entry_price", pl.Float64),
        ("vwap0", pl.Float64),
        ("sigma0", pl.Float64),
        ("distance0", pl.Float64),
        ("cooldown_seconds", pl.Float64),
        ("since_touch_any_seconds", pl.Float64),
        ("time_to_vwap_s", pl.Float64),
        ("trades_to_vwap", pl.Int32),
    ]

    df_out = df_out.with_columns([
        pl.col(name).cast(dtype, strict=False)
        for name, dtype in core_casts
        if name in df_out.columns
    ])

    # Column order
    core_cols = [
        "event_id", "date_ymd", "is_weekend",
        "ts", "year", "month", "day",
        "band_k", "side", "event_n",
        "entry_price", "vwap0", "sigma0", "distance0",
        "cooldown_seconds", "since_touch_any_seconds",
        "time_to_vwap_s", "trades_to_vwap",
    ]
    other_cols = [c for c in df_out.columns if c not in core_cols]
    df_out = df_out.select([c for c in core_cols if c in df_out.columns] + sorted(other_cols))

    df_out.write_parquet(OUT_FILE)

    print("\nDone.")
    print(f"Wrote outcomes: {OUT_FILE}")
    print(f"Rows: {df_out.height:,}  Cols: {len(df_out.columns)}")

    print("\nCoverage summary:")
    print(f"  total partitions found: {len(day_files):,}")
    print(f"  days processed:         {days_done:,}")
    print(f"  days w/ 0 events:       {days_with_zero_events:,}")
    print(f"  skipped (<500 trades):  {len(skipped_small):,}")
    print(f"  skipped (no sigma):     {len(skipped_no_sigma):,}")
    print(f"  skipped (sigma <= 0):   {len(skipped_sigma_nonpos):,}")
    print(f"  accounted total:        {(days_done + len(skipped_small) + len(skipped_no_sigma) + len(skipped_sigma_nonpos)):,}")

    if skipped_small:
        print("\nPreview skipped (<500 trades):")
        print(skipped_small[:10])

    if skipped_no_sigma:
        print("\nPreview skipped (no sigma):")
        print(skipped_no_sigma[:10])

    if skipped_sigma_nonpos:
        print("\nPreview skipped (sigma <= 0):")
        print(skipped_sigma_nonpos[:10])

    # Quick diagnostics
    if "cooldown_seconds" in df_out.columns:
        q = (
            df_out
            .filter(pl.col("cooldown_seconds").is_not_null())
            .group_by(["band_k", "side"])
            .agg([
                pl.len().alias("n"),
                pl.col("cooldown_seconds").quantile(0.10).alias("cooldown_q10"),
                pl.col("cooldown_seconds").quantile(0.50).alias("cooldown_q50"),
                pl.col("cooldown_seconds").quantile(0.90).alias("cooldown_q90"),
                pl.col("cooldown_seconds").min().alias("cooldown_min"),
                pl.col("cooldown_seconds").max().alias("cooldown_max"),
            ])
            .sort(["band_k", "side"])
        )
        print("\nCooldown_seconds quantiles by band/side (non-null):")
        print(q)

    if "since_touch_any_seconds" in df_out.columns:
        q2 = (
            df_out
            .filter(pl.col("since_touch_any_seconds").is_not_null())
            .group_by(["band_k", "side"])
            .agg([
                pl.len().alias("n"),
                pl.col("since_touch_any_seconds").quantile(0.10).alias("touch_any_q10"),
                pl.col("since_touch_any_seconds").quantile(0.50).alias("touch_any_q50"),
                pl.col("since_touch_any_seconds").quantile(0.90).alias("touch_any_q90"),
                pl.col("since_touch_any_seconds").min().alias("touch_any_min"),
                pl.col("since_touch_any_seconds").max().alias("touch_any_max"),
            ])
            .sort(["band_k", "side"])
        )
        print("\nSince_touch_any_seconds quantiles by band/side (non-null):")
        print(q2)

if __name__ == "__main__":
    main()
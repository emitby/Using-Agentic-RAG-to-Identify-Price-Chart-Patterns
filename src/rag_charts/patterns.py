"""Deterministic chart-pattern detection.

Everything numeric on the final chart comes from here: swing pivots, support/resistance
levels, trendlines, classic reversal patterns, the current price formation, and the
moving-average trend regime. The agents never invent prices; they review these candidates
against the website definitions and decide which to show and how to describe them.

Method overview
---------------
1. **Zigzag pivots.** A swing high/low is confirmed once price reverses by at least
   `zigzag_pct` from it. This filters noise in proportion to the move size and yields an
   alternating High/Low sequence, which every pattern test below reads.
2. **Support/resistance.** Pivot prices are clustered; a cluster touched two or more times
   is a level (ChartSchool: support/resistance are where prior turns cluster). Levels below
   the last close are support, above are resistance.
3. **Trendlines.** For pairs of rising pivot lows (support) or falling pivot highs
   (resistance), the line through them must not be violated by closes between the two
   anchors (a valid trendline is not crossed). More touches and later anchors score higher.
4. **Reversal patterns** read consecutive pivots: double/triple tops and bottoms, head and
   shoulders (and inverse). Status is "confirmed" once price closes through the trough /
   peak / neckline, "pending" before that, and "failed" (busted) if price instead exceeds
   the pattern's extreme.
5. **Current formation.** Lines fitted through the last few pivot highs and lows are
   classified by slope and convergence into triangles, wedges, channels, rectangles or
   broadening formations.
6. **Trend regime.** Fast/slow simple moving averages (50/200 days, or 10/40 weeks, the
   weekly equivalent) give the trend direction and the latest golden/death cross.

Thresholds per bar size are in `PARAMS` and are deliberately simple and documented so the
Verifier agent can check them against the definitions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Detection thresholds. Weekly bars are noisier per bar but cover more time, so swings
# must be larger before they count, and levels may be wider.
PARAMS = {
    "daily": {
        "zigzag_pct": 0.035,     # reversal needed to confirm a swing pivot
        "level_tol": 0.012,      # pivots within 1.2% belong to the same S/R level
        "equal_tol": 0.025,      # "roughly equal" peaks/troughs (double/triple, shoulders)
        "head_min": 0.02,        # head must exceed both shoulders by 2%
        "min_sep_bars": 10,      # minimum bars between twin peaks/troughs
        "flat_change": 0.025,    # a formation side moving <2.5% end-to-end is "flat"
        "sma": (50, 200),
    },
    "weekly": {
        "zigzag_pct": 0.06,
        "level_tol": 0.02,
        "equal_tol": 0.03,
        "head_min": 0.03,
        "min_sep_bars": 4,
        "flat_change": 0.04,
        "sma": (10, 40),         # 10/40 weeks ~= 50/200 days
    },
}

MAX_PER_PATTERN_TYPE = 2   # keep the most recent N of each reversal pattern type
MAX_LEVELS_EACH_SIDE = 3   # support levels below / resistance levels above the price


@dataclass
class Pivot:
    """A confirmed (or, for the last one, provisional) swing high or low.

    Attributes:
        pos: Integer bar position in the frame.
        date: Bar date.
        price: Closing price at the pivot.
        kind: "H" for swing high, "L" for swing low.
        confirmed: False only for the final running extreme, which could still extend.
    """
    pos: int
    date: pd.Timestamp
    price: float
    kind: str
    confirmed: bool = True


def _pt(pivot: Pivot, label: str = "") -> dict:
    """Serialize a pivot as a labeled chart point."""
    return {"label": label, "date": pivot.date.strftime("%Y-%m-%d"),
            "price": round(pivot.price, 2)}


def _date(bars: pd.DataFrame, pos: int) -> str:
    """Date string of a bar position."""
    return bars.index[pos].strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 1. Zigzag pivots
# ---------------------------------------------------------------------------
def zigzag(close: pd.Series, pct: float) -> list[Pivot]:
    """Find alternating swing highs and lows that are separated by at least `pct` moves.

    Walk the closes keeping the running extreme of the current leg. When price retraces
    `pct` from that extreme, the extreme becomes a confirmed pivot and a new leg starts in
    the other direction. The final running extreme is appended as an unconfirmed pivot.

    Args:
        close: Closing prices indexed by date.
        pct: Minimum reversal as a fraction (0.05 = 5%).

    Returns:
        Pivots in date order, alternating H and L.
    """
    values = close.to_numpy(dtype=float)
    dates = close.index
    if len(values) < 3:
        return []
    pivots: list[Pivot] = []
    direction = 0                 # +1 rising leg, -1 falling leg, 0 undecided
    hi_pos = lo_pos = 0
    for i in range(1, len(values)):
        if values[i] > values[hi_pos]:
            hi_pos = i
        if values[i] < values[lo_pos]:
            lo_pos = i
        if direction >= 0 and values[i] <= values[hi_pos] * (1 - pct) and hi_pos > 0:
            if direction == 0 and lo_pos < hi_pos:
                pivots.append(Pivot(lo_pos, dates[lo_pos], values[lo_pos], "L"))
            pivots.append(Pivot(hi_pos, dates[hi_pos], values[hi_pos], "H"))
            direction, lo_pos = -1, i
        elif direction <= 0 and values[i] >= values[lo_pos] * (1 + pct) and lo_pos > 0:
            if direction == 0 and hi_pos < lo_pos:
                pivots.append(Pivot(hi_pos, dates[hi_pos], values[hi_pos], "H"))
            pivots.append(Pivot(lo_pos, dates[lo_pos], values[lo_pos], "L"))
            direction, hi_pos = 1, i
    # Provisional final extreme of the leg in progress.
    if direction == 1:
        pivots.append(Pivot(hi_pos, dates[hi_pos], values[hi_pos], "H", confirmed=False))
    elif direction == -1:
        pivots.append(Pivot(lo_pos, dates[lo_pos], values[lo_pos], "L", confirmed=False))
    return pivots


# ---------------------------------------------------------------------------
# 2. Support and resistance
# ---------------------------------------------------------------------------
def support_resistance(pivots: list[Pivot], last_close: float, tol: float,
                       bars: pd.DataFrame) -> list[dict]:
    """Cluster pivot prices into horizontal support/resistance levels.

    Greedy clustering over pivots sorted by price: a pivot joins the current cluster if it
    is within `tol` of the cluster's mean. Clusters with 2+ touches become levels. Some
    single-touch pivots are always offered too, because ChartSchool treats them as support
    or resistance on their own: the window's highest and lowest pivots (range extremes,
    e.g. an all-time high) and the most recent confirmed swing high and swing low (the
    nearest prior turns, which on a long uptrend are far closer than any multi-touch zone).

    Args:
        pivots: Output of `zigzag`.
        last_close: Latest close; decides support (below) vs resistance (above).
        tol: Clustering tolerance as a fraction of price.
        bars: The OHLCV frame (for dates).

    Returns:
        Candidate dicts of type "support" or "resistance", nearest levels first.
    """
    if not pivots:
        return []
    ordered = sorted(pivots, key=lambda p: p.price)
    clusters: list[list[Pivot]] = [[ordered[0]]]
    for pivot in ordered[1:]:
        mean = np.mean([p.price for p in clusters[-1]])
        if abs(pivot.price - mean) / mean <= tol:
            clusters[-1].append(pivot)
        else:
            clusters.append([pivot])

    # Single-touch pivots that still qualify as levels, with the reason why.
    key_pivots = {max(pivots, key=lambda p: p.price).pos: "range extreme",
                  min(pivots, key=lambda p: p.price).pos: "range extreme"}
    for kind in ("H", "L"):
        recent = [p for p in pivots if p.kind == kind and p.confirmed]
        if recent:
            key_pivots.setdefault(recent[-1].pos,
                                  "most recent swing high" if kind == "H" else "most recent swing low")
    levels = []
    for cluster in clusters:
        reasons = sorted({key_pivots[p.pos] for p in cluster if p.pos in key_pivots})
        is_extreme = bool(reasons)
        if len(cluster) < 2 and not is_extreme:
            continue
        price = float(np.mean([p.price for p in cluster]))
        side = "support" if price < last_close else "resistance"
        touches = sorted(cluster, key=lambda p: p.pos)
        levels.append({
            "type": side,
            "family": "level",
            "implied_bias": "neutral",
            "status": "active",
            "start": touches[0].date.strftime("%Y-%m-%d"),
            "end": touches[-1].date.strftime("%Y-%m-%d"),
            "points": [_pt(p, f"touch {i + 1} ({'high' if p.kind == 'H' else 'low'})")
                       for i, p in enumerate(touches)],
            "shapes": [{"kind": "level", "price": round(price, 2),
                        "band": round(price * tol, 2)}],
            "metrics": {
                "level": round(price, 2),
                "touches": len(cluster),
                "touch_kinds": "".join(p.kind for p in touches),
                "distance_from_close_pct": round((price / last_close - 1) * 100, 2),
                "key_pivot": ", ".join(reasons),
                "last_touch": touches[-1].date.strftime("%Y-%m-%d"),
                "role_reversal": len({p.kind for p in cluster}) == 2,
            },
            "summary": (f"{side.title()} near ${price:,.2f}: {len(cluster)} pivot touch(es) "
                        f"from {touches[0].date:%Y-%m-%d} to {touches[-1].date:%Y-%m-%d}"
                        + ("; acted as both support and resistance (role reversal)"
                           if len({p.kind for p in cluster}) == 2 else "")
                        + (f"; {', '.join(reasons)}" if reasons else "") + "."),
        })
    below = sorted([lv for lv in levels if lv["type"] == "support"],
                   key=lambda lv: -lv["metrics"]["level"])[:MAX_LEVELS_EACH_SIDE]
    above = sorted([lv for lv in levels if lv["type"] == "resistance"],
                   key=lambda lv: lv["metrics"]["level"])[:MAX_LEVELS_EACH_SIDE]
    return below + above


# ---------------------------------------------------------------------------
# 3. Trendlines
# ---------------------------------------------------------------------------
def _line_value(a: Pivot, b: Pivot, pos: int) -> float:
    """Price of the straight line through pivots a and b at bar position `pos`."""
    slope = (b.price - a.price) / (b.pos - a.pos)
    return a.price + slope * (pos - a.pos)


def trendlines(pivots: list[Pivot], bars: pd.DataFrame, tol: float,
               min_change: float) -> list[dict]:
    """Find the best rising support trendline and the best falling resistance trendline.

    For each pair of recent same-kind pivots (rising lows for support, falling highs for
    resistance) the line through them is tested. The anchors must differ by at least
    `min_change` (a nearly flat line is horizontal support/resistance, not a rising or
    falling trendline; the Chart Analyst flagged a -0.003%/bar "falling" line in an eval).
    No close between the two anchors may
    cross it by more than `tol`. Valid lines are scored by touches (pivots within `tol`
    of the line), recency of the second anchor, and span. After the second anchor the
    line is extended to the last bar and checked for a break.

    Args:
        pivots: Output of `zigzag`.
        bars: OHLCV frame.
        tol: Touch/violation tolerance as a fraction of price.
        min_change: Minimum relative price change between the two anchors.

    Returns:
        Up to two candidate dicts ("rising_support_trendline",
        "falling_resistance_trendline").
    """
    close = bars["close"].to_numpy(dtype=float)
    last = len(close) - 1
    found = []
    for kind, want_rising, name, below in (("L", True, "rising_support_trendline", True),
                                           ("H", False, "falling_resistance_trendline", False)):
        anchors = [p for p in pivots if p.kind == kind and p.confirmed][-8:]
        best, best_score = None, -1.0
        for i, a in enumerate(anchors):
            for b in anchors[i + 1:]:
                if (b.price > a.price) != want_rising or abs(b.price / a.price - 1) < min_change:
                    continue
                line = np.array([_line_value(a, b, k) for k in range(a.pos, b.pos + 1)])
                segment = close[a.pos:b.pos + 1]
                # Support may not be closed below; resistance may not be closed above.
                crossed = (segment < line * (1 - tol)) if below else (segment > line * (1 + tol))
                if crossed.any():
                    continue
                touches = sum(abs(p.price - _line_value(a, b, p.pos)) <= p.price * tol
                              for p in pivots if p.kind == kind and p.pos >= a.pos)
                score = touches * 2 + b.pos / last + (b.pos - a.pos) / last
                if score > best_score:
                    best, best_score = (a, b, touches), score
        if not best:
            continue
        a, b, touches = best
        # Look for the first close through the extended line after the second anchor.
        broken_pos = None
        for k in range(b.pos + 1, last + 1):
            value = _line_value(a, b, k)
            if (close[k] < value * (1 - tol)) if below else (close[k] > value * (1 + tol)):
                broken_pos = k
                break
        end_pos = broken_pos if broken_pos is not None else last
        slope_pct = (b.price / a.price - 1) / (b.pos - a.pos) * 100
        status = "broken" if broken_pos is not None else "intact"
        found.append({
            "type": name,
            "family": "trendline",
            "implied_bias": ("bullish" if below else "bearish") if status == "intact"
            else ("bearish" if below else "bullish"),
            "status": status,
            "start": a.date.strftime("%Y-%m-%d"),
            "end": _date(bars, end_pos),
            "points": [_pt(a, "anchor 1"), _pt(b, "anchor 2")],
            "shapes": [{"kind": "line", "points": [
                _pt(a), {"label": "", "date": _date(bars, end_pos),
                         "price": round(_line_value(a, b, end_pos), 2)}]}],
            "metrics": {
                "touches": int(touches),
                "slope_pct_per_bar": round(slope_pct, 4),
                "span_bars": int(end_pos - a.pos),
                "broken_on": _date(bars, broken_pos) if broken_pos is not None else "",
                "line_value_at_last_bar": round(_line_value(a, b, last), 2),
            },
            "summary": (f"{'Rising support' if below else 'Falling resistance'} trendline from "
                        f"{a.date:%Y-%m-%d} (${a.price:,.2f}) through {b.date:%Y-%m-%d} "
                        f"(${b.price:,.2f}), {touches} touches, "
                        + (f"broken on {_date(bars, broken_pos)}." if broken_pos is not None
                           else "still intact.")),
        })
    return found


# ---------------------------------------------------------------------------
# 4. Reversal patterns
# ---------------------------------------------------------------------------
def _first_close_beyond(close: np.ndarray, start: int, level: float, below: bool) -> int | None:
    """First bar at/after `start` whose close is below (or above) `level`, else None."""
    for k in range(start, len(close)):
        if (close[k] < level) if below else (close[k] > level):
            return k
    return None


def _status(close, after: int, confirm_level: float, fail_level: float,
            bearish: bool) -> tuple[str, int | None]:
    """Decide a reversal pattern's status from what price did after its last pivot.

    Confirmation (close through `confirm_level` in the pattern's direction) and failure
    (close beyond the pattern's extreme the other way) are both searched for; whichever
    happened first wins. A pattern can be confirmed and later fail ("busted").

    Returns:
        (status, bar position of the deciding event or None).
    """
    confirm = _first_close_beyond(close, after, confirm_level, below=bearish)
    fail = _first_close_beyond(close, after, fail_level, below=not bearish)
    if confirm is not None and (fail is None or confirm < fail):
        bust = _first_close_beyond(close, confirm, fail_level, below=not bearish)
        return ("confirmed_then_failed", bust) if bust is not None else ("confirmed", confirm)
    if fail is not None:
        return "failed", fail
    return "pending", None


def reversal_patterns(pivots: list[Pivot], bars: pd.DataFrame, p: dict) -> list[dict]:
    """Detect double/triple tops and bottoms and head-and-shoulders (and inverse).

    Rules (matching the ChartSchool / Bulkowski descriptions at the level a close-only
    chart can test):
      * Double top: peak, trough, peak; peaks within `equal_tol`; at least `min_sep_bars`
        apart. Confirmed by a close below the trough. Mirror for double bottom.
      * Triple top: three peaks within `equal_tol` with two troughs. Confirmed by a close
        below the lower trough. Mirror for triple bottom.
      * Head and shoulders top: three peaks, middle one at least `head_min` above both
        shoulders, shoulders within 2 x `equal_tol` of each other. Neckline through the two
        troughs; confirmed by a close below the neckline. Mirror for the inverse pattern.

    Args:
        pivots: Output of `zigzag` (only confirmed pivots form patterns).
        bars: OHLCV frame.
        p: Threshold dict from `PARAMS`.

    Returns:
        Candidate dicts, the most recent `MAX_PER_PATTERN_TYPE` per type.
    """
    close = bars["close"].to_numpy(dtype=float)
    confirmed = [pv for pv in pivots if pv.confirmed]
    eq, out = p["equal_tol"], {}

    def near(a: float, b: float, tol: float) -> bool:
        return abs(a - b) / max(a, b) <= tol

    def add(kind: str, record: dict) -> None:
        out.setdefault(kind, []).append(record)

    for i in range(len(confirmed) - 2):
        a, m, b = confirmed[i:i + 3]
        # Double top / bottom: X-Y-X with equal X's.
        if a.kind == b.kind and near(a.price, b.price, eq) and b.pos - a.pos >= p["min_sep_bars"]:
            top = a.kind == "H"
            extreme = max(a.price, b.price) if top else min(a.price, b.price)
            status, at = _status(close, b.pos + 1, m.price, extreme * (1 + eq if top else 1 - eq), top)
            depth = abs(extreme - m.price) / extreme
            kind = "double_top" if top else "double_bottom"
            add(kind, {
                "type": kind, "family": "reversal",
                "implied_bias": "bearish" if top else "bullish",
                "status": status, "start": _date(bars, a.pos),
                "end": _date(bars, at if at is not None else len(close) - 1),
                "points": [_pt(a, "peak 1" if top else "trough 1"),
                           _pt(m, "trough" if top else "peak"),
                           _pt(b, "peak 2" if top else "trough 2")],
                "shapes": [
                    {"kind": "polyline", "points": [_pt(a), _pt(m), _pt(b)]},
                    {"kind": "line", "style": "secondary", "points": [
                        {"label": "", "date": _date(bars, a.pos), "price": round(m.price, 2)},
                        {"label": "", "date": _date(bars, at if at is not None else len(close) - 1),
                         "price": round(m.price, 2)}]},
                ],
                "metrics": {
                    "peak_difference_pct": round(abs(a.price - b.price) / extreme * 100, 2),
                    "depth_pct": round(depth * 100, 2),
                    "bars_between": int(b.pos - a.pos),
                    "confirmation_level": round(m.price, 2),
                    "event_date": _date(bars, at) if at is not None else "",
                },
                "summary": (f"{'Double top' if top else 'Double bottom'}: "
                            f"{'peaks' if top else 'troughs'} ${a.price:,.2f} ({a.date:%Y-%m-%d}) "
                            f"and ${b.price:,.2f} ({b.date:%Y-%m-%d}), "
                            f"{'trough' if top else 'peak'} ${m.price:,.2f}; status {status}"
                            + (f" on {_date(bars, at)}" if at is not None else "") + "."),
            })

    for i in range(len(confirmed) - 4):
        s1, t1, h, t2, s2 = confirmed[i:i + 5]
        if not (s1.kind == h.kind == s2.kind):
            continue
        top = h.kind == "H"
        # Triple top / bottom: three roughly equal extremes.
        if near(s1.price, h.price, eq) and near(h.price, s2.price, eq) and near(s1.price, s2.price, eq):
            extreme = max(s1.price, h.price, s2.price) if top else min(s1.price, h.price, s2.price)
            confirm_level = min(t1.price, t2.price) if top else max(t1.price, t2.price)
            status, at = _status(close, s2.pos + 1, confirm_level,
                                 extreme * (1 + eq if top else 1 - eq), top)
            kind = "triple_top" if top else "triple_bottom"
            add(kind, {
                "type": kind, "family": "reversal",
                "implied_bias": "bearish" if top else "bullish",
                "status": status, "start": _date(bars, s1.pos),
                "end": _date(bars, at if at is not None else len(close) - 1),
                "points": [_pt(x, f"{'peak' if top else 'trough'} {n}")
                           for n, x in enumerate((s1, h, s2), 1)],
                "shapes": [{"kind": "polyline", "points": [_pt(x) for x in (s1, t1, h, t2, s2)]},
                           {"kind": "line", "style": "secondary", "points": [
                               {"label": "", "date": _date(bars, s1.pos), "price": round(confirm_level, 2)},
                               {"label": "", "date": _date(bars, at if at is not None else len(close) - 1),
                                "price": round(confirm_level, 2)}]}],
                "metrics": {"confirmation_level": round(confirm_level, 2),
                            "spread_pct": round((max(s1.price, h.price, s2.price)
                                                 / min(s1.price, h.price, s2.price) - 1) * 100, 2),
                            "event_date": _date(bars, at) if at is not None else ""},
                "summary": (f"{'Triple top' if top else 'Triple bottom'} at about ${extreme:,.2f} "
                            f"({s1.date:%Y-%m-%d} to {s2.date:%Y-%m-%d}); status {status}."),
            })
            continue
        # Head and shoulders: middle extreme beyond both shoulders by head_min.
        head_ok = (h.price > s1.price * (1 + p["head_min"]) and h.price > s2.price * (1 + p["head_min"])) \
            if top else (h.price < s1.price * (1 - p["head_min"]) and h.price < s2.price * (1 - p["head_min"]))
        if not head_ok or not near(s1.price, s2.price, 2 * eq):
            continue
        neckline_at = lambda pos: _line_value(t1, t2, pos)  # noqa: E731
        confirm, fail = None, None
        for k in range(s2.pos + 1, len(close)):
            beyond_neck = close[k] < neckline_at(k) if top else close[k] > neckline_at(k)
            beyond_head = close[k] > h.price if top else close[k] < h.price
            if beyond_neck and confirm is None and fail is None:
                confirm = k
            if beyond_head and fail is None:
                fail = k
        if confirm is not None and (fail is None or confirm < fail):
            status, at = ("confirmed_then_failed", fail) if fail is not None else ("confirmed", confirm)
        elif fail is not None:
            status, at = "failed", fail
        else:
            status, at = "pending", None
        end_pos = at if at is not None else len(close) - 1
        kind = "head_and_shoulders_top" if top else "inverse_head_and_shoulders"
        add(kind, {
            "type": kind, "family": "reversal",
            "implied_bias": "bearish" if top else "bullish",
            "status": status, "start": _date(bars, s1.pos), "end": _date(bars, end_pos),
            "points": [_pt(s1, "left shoulder"), _pt(t1, "neckline 1"), _pt(h, "head"),
                       _pt(t2, "neckline 2"), _pt(s2, "right shoulder")],
            "shapes": [{"kind": "polyline", "points": [_pt(x) for x in (s1, t1, h, t2, s2)]},
                       {"kind": "line", "style": "secondary", "points": [
                           _pt(t1), {"label": "", "date": _date(bars, end_pos),
                                     "price": round(neckline_at(end_pos), 2)}]}],
            "metrics": {
                "head_over_shoulders_pct": round(abs(h.price / max(s1.price, s2.price) - 1) * 100, 2)
                if top else round(abs(1 - h.price / min(s1.price, s2.price)) * 100, 2),
                "shoulder_difference_pct": round(abs(s1.price - s2.price) / max(s1.price, s2.price) * 100, 2),
                "neckline_slope_pct": round((t2.price / t1.price - 1) * 100, 2),
                "event_date": _date(bars, at) if at is not None else "",
            },
            "summary": (f"{'Head-and-shoulders top' if top else 'Inverse head-and-shoulders'}: "
                        f"shoulders ${s1.price:,.2f} / ${s2.price:,.2f}, head ${h.price:,.2f} "
                        f"({h.date:%Y-%m-%d}); status {status}"
                        + (f" on {_date(bars, at)}" if at is not None else "") + "."),
        })

    results = []
    for records in out.values():
        results.extend(records[-MAX_PER_PATTERN_TYPE:])
    return results


# ---------------------------------------------------------------------------
# 5. Current formation (triangles, wedges, channels, rectangles, broadening)
# ---------------------------------------------------------------------------
def current_formation(pivots: list[Pivot], bars: pd.DataFrame, p: dict) -> dict | None:
    """Classify the shape traced by the most recent pivots.

    Least-squares lines are fitted through the last 3 confirmed pivot highs and last 3
    confirmed pivot lows (2 each if that is all there is). The provisional final pivot is
    left out: it is just the running extreme of a leg still in progress, so treating it
    as a peak would, for example, turn a rally toward flat resistance into a false "lower
    high" and misread an ascending triangle as symmetrical. Each line is
    "flat" if it changes less than `flat_change` over the formation, else rising/falling.
    The gap between the lines at the start vs. the end decides converging (<70%),
    diverging (>130%) or parallel.

    Classification (ChartSchool / Bulkowski definitions):
      flat highs + rising lows ............ ascending triangle (bullish)
      falling highs + flat lows ........... descending triangle (bearish)
      falling highs + rising lows ......... symmetrical triangle (neutral, continuation)
      both rising, converging ............. rising wedge (bearish)
      both falling, converging ............ falling wedge (bullish)
      both rising/falling, parallel ....... rising/falling channel
      both flat ........................... rectangle
      diverging ........................... broadening formation

    Args:
        pivots: Output of `zigzag`.
        bars: OHLCV frame.
        p: Threshold dict from `PARAMS`.

    Returns:
        A candidate dict, or None if there are too few pivots or no clean fit.
    """
    all_highs = [pv for pv in pivots if pv.kind == "H" and pv.confirmed]
    all_lows = [pv for pv in pivots if pv.kind == "L" and pv.confirmed]
    end = len(bars) - 1

    # Try the widest fit first, then drop the oldest pivot on either side. A fit is only
    # accepted if every pivot lies within `flat_change` of its line and the two lines do
    # not cross between the formation start and the last bar. Otherwise an old pivot
    # from a different phase (e.g. the low a rally started from) would drag a line
    # through the pattern, as the Chart Analyst caught in the ascending-triangle eval.
    for n_hi, n_lo in ((3, 3), (3, 2), (2, 3), (2, 2)):
        highs, lows = all_highs[-n_hi:], all_lows[-n_lo:]
        if len(highs) < n_hi or len(lows) < n_lo:
            continue
        start = min(highs[0].pos, lows[0].pos)
        fit_hi = np.polyfit([x.pos for x in highs], [x.price for x in highs], 1)
        fit_lo = np.polyfit([x.pos for x in lows], [x.price for x in lows], 1)
        residual = max(abs(x.price / np.polyval(fit, x.pos) - 1)
                       for fit, group in ((fit_hi, highs), (fit_lo, lows)) for x in group)
        hi_s, hi_e = np.polyval(fit_hi, start), np.polyval(fit_hi, end)
        lo_s, lo_e = np.polyval(fit_lo, start), np.polyval(fit_lo, end)
        if residual <= p["flat_change"] and hi_s > lo_s and hi_e > lo_e:
            break
    else:
        return None

    def direction(a: float, b: float) -> str:
        change = b / a - 1
        return "flat" if abs(change) < p["flat_change"] else ("rising" if change > 0 else "falling")

    upper, lower = direction(hi_s, hi_e), direction(lo_s, lo_e)
    width_ratio = (hi_e - lo_e) / (hi_s - lo_s)  # both widths are positive after the fit loop
    shape = ("converging" if width_ratio < 0.7 else
             "diverging" if width_ratio > 1.3 else "parallel")

    table = {
        ("flat", "rising"): ("ascending_triangle", "bullish"),
        ("falling", "flat"): ("descending_triangle", "bearish"),
        ("falling", "rising"): ("symmetrical_triangle", "neutral"),
        ("flat", "flat"): ("rectangle", "neutral"),
    }
    if (upper, lower) in table:
        kind, bias = table[(upper, lower)]
    elif upper == lower == "rising":
        kind, bias = (("rising_wedge", "bearish") if shape == "converging"
                      else ("rising_channel", "bullish") if shape == "parallel"
                      else ("broadening_formation", "neutral"))
    elif upper == lower == "falling":
        kind, bias = (("falling_wedge", "bullish") if shape == "converging"
                      else ("falling_channel", "bearish") if shape == "parallel"
                      else ("broadening_formation", "neutral"))
    else:
        kind, bias = "broadening_formation", "neutral"

    last_close = float(bars["close"].iloc[-1])
    position = ("above upper line" if last_close > hi_e * 1.005 else
                "below lower line" if last_close < lo_e * 0.995 else "inside")
    used = sorted(highs + lows, key=lambda x: x.pos)
    return {
        "type": kind, "family": "formation", "implied_bias": bias,
        "status": {"inside": "forming", "above upper line": "breakout_up",
                   "below lower line": "breakout_down"}[position],
        "start": _date(bars, start), "end": _date(bars, end),
        "points": [_pt(x, f"pivot {x.kind}") for x in used],
        # Each line is drawn from its own first pivot (not the formation start) so it is
        # never extrapolated backward away from the prices it was fitted to.
        "shapes": [
            {"kind": "line", "points": [
                {"label": "", "date": _date(bars, highs[0].pos),
                 "price": round(float(np.polyval(fit_hi, highs[0].pos)), 2)},
                {"label": "", "date": _date(bars, end), "price": round(hi_e, 2)}]},
            {"kind": "line", "points": [
                {"label": "", "date": _date(bars, lows[0].pos),
                 "price": round(float(np.polyval(fit_lo, lows[0].pos)), 2)},
                {"label": "", "date": _date(bars, end), "price": round(lo_e, 2)}]},
        ],
        "metrics": {
            "upper_line": upper, "lower_line": lower, "shape": shape,
            "width_ratio_end_vs_start": round(float(width_ratio), 2),
            "upper_line_now": round(float(hi_e), 2), "lower_line_now": round(float(lo_e), 2),
            "pivot_highs": len(highs), "pivot_lows": len(lows),
            "price_position": position,
        },
        "summary": (f"Recent pivots form a {kind.replace('_', ' ')} candidate: upper line {upper}, "
                    f"lower line {lower}, {shape} (width ratio {width_ratio:.2f}); "
                    f"last close is {position} (upper ${hi_e:,.2f}, lower ${lo_e:,.2f})."),
    }


# ---------------------------------------------------------------------------
# 6. Trend regime
# ---------------------------------------------------------------------------
def trend_regime(bars: pd.DataFrame, fast: int, slow: int) -> dict:
    """Summarize trend direction from price vs. fast/slow simple moving averages.

    Score (each condition +1 if true, -1 if false): close above fast SMA, close above slow
    SMA, fast above slow, slow SMA rising over the last 20 bars. +3/+4 = uptrend,
    -3/-4 = downtrend, otherwise mixed. Also finds the latest golden/death cross.

    Args:
        bars: OHLCV frame.
        fast: Fast SMA length in bars.
        slow: Slow SMA length in bars.

    Returns:
        Dict with the regime, score, SMA values, cross info, and the SMA series (for the
        chart; stripped before anything is sent to a model).
    """
    close = bars["close"]
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean()
    last = float(close.iloc[-1])
    checks = {
        "close_above_fast_sma": bool(last > sma_f.iloc[-1]),
        "close_above_slow_sma": bool(last > sma_s.iloc[-1]),
        "fast_above_slow": bool(sma_f.iloc[-1] > sma_s.iloc[-1]),
        "slow_sma_rising": bool(sma_s.iloc[-1] > sma_s.iloc[-21]) if len(sma_s.dropna()) > 21 else False,
    }
    score = sum(1 if v else -1 for v in checks.values())
    regime = "uptrend" if score >= 3 else "downtrend" if score <= -3 else "mixed"
    above = (sma_f > sma_s).astype(int).diff()
    crosses = above[above != 0].dropna()
    last_cross = None
    if not crosses.empty:
        when = crosses.index[-1]
        last_cross = {"type": "golden_cross" if crosses.iloc[-1] > 0 else "death_cross",
                      "date": when.strftime("%Y-%m-%d"), "price": round(float(close.loc[when]), 2)}
    window_high = float(close.max())
    return {
        "regime": regime, "score": score, "checks": checks,
        "fast_len": fast, "slow_len": slow,
        "fast_sma": round(float(sma_f.iloc[-1]), 2), "slow_sma": round(float(sma_s.iloc[-1]), 2),
        "last_close": round(last, 2),
        "drawdown_from_window_high_pct": round((last / window_high - 1) * 100, 2),
        "last_cross": last_cross,
        "sma_fast_series": sma_f, "sma_slow_series": sma_s,
    }


def cross_candidate(trend: dict) -> dict | None:
    """Turn the latest moving-average cross into a marker candidate (None if no cross)."""
    cross = trend.get("last_cross")
    if not cross:
        return None
    golden = cross["type"] == "golden_cross"
    return {
        "type": cross["type"], "family": "moving_average",
        "implied_bias": "bullish" if golden else "bearish", "status": "occurred",
        "start": cross["date"], "end": cross["date"],
        "points": [{"label": cross["type"].replace("_", " "), "date": cross["date"], "price": cross["price"]}],
        "shapes": [{"kind": "marker", "points": [{"label": "", "date": cross["date"], "price": cross["price"]}]}],
        "metrics": {"fast_len": trend["fast_len"], "slow_len": trend["slow_len"]},
        "summary": (f"{'Golden' if golden else 'Death'} cross: {trend['fast_len']}-bar SMA crossed "
                    f"{'above' if golden else 'below'} the {trend['slow_len']}-bar SMA on {cross['date']}."),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def detect(bars: pd.DataFrame, bar_size: str) -> dict:
    """Run every detector on one window and return a detection report.

    Args:
        bars: Date-indexed OHLCV frame (daily or weekly).
        bar_size: "daily" or "weekly" (selects `PARAMS`).

    Returns:
        Report dict: window info, `trend` (with SMA series), `pivots`, and `candidates`,
        each candidate given a short id like "C01".
    """
    p = PARAMS[bar_size]
    close = bars["close"]
    pivots = zigzag(close, p["zigzag_pct"])
    trend = trend_regime(bars, *p["sma"])
    last_close = float(close.iloc[-1])

    candidates = []
    candidates += support_resistance(pivots, last_close, p["level_tol"], bars)
    candidates += trendlines(pivots, bars, p["level_tol"], p["flat_change"])
    candidates += reversal_patterns(pivots, bars, p)
    formation = current_formation(pivots, bars, p)
    if formation:
        candidates.append(formation)
    cross = cross_candidate(trend)
    if cross:
        candidates.append(cross)
    for n, cand in enumerate(candidates, start=1):
        cand["id"] = f"C{n:02d}"

    return {
        "bar_size": bar_size, "params": p,
        "first_date": bars.index[0].strftime("%Y-%m-%d"),
        "last_date": bars.index[-1].strftime("%Y-%m-%d"),
        "bars": len(bars), "last_close": round(last_close, 2),
        "window_high": round(float(close.max()), 2), "window_low": round(float(close.min()), 2),
        "trend": trend,
        "pivots": [_pt(pv, f"{pv.kind}{'' if pv.confirmed else '?'}") for pv in pivots],
        "candidates": candidates,
    }


def report_for_model(report: dict) -> dict:
    """Copy of a detection report without pandas objects, safe to JSON-encode for a model.

    Drops the SMA series and the shapes (drawing geometry the model does not need), keeping
    the labeled points, metrics and summary of each candidate.
    """
    trend = {k: v for k, v in report["trend"].items() if not k.endswith("_series")}
    candidates = [{k: v for k, v in c.items() if k != "shapes"} for c in report["candidates"]]
    return {**{k: v for k, v in report.items() if k not in ("trend", "candidates")},
            "trend": trend, "candidates": candidates}

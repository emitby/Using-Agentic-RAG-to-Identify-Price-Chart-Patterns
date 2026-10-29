"""Tests for the deterministic detector on synthetic series with planted patterns.

Each series is built from straight-line legs so the expected pivots are known exactly.
"""

from helpers import path_series

from rag_charts.patterns import PARAMS, detect, trend_regime, zigzag


def types_of(report: dict) -> dict:
    """Map candidate type -> list of candidates."""
    out = {}
    for cand in report["candidates"]:
        out.setdefault(cand["type"], []).append(cand)
    return out


def test_zigzag_alternates_and_respects_threshold():
    bars = path_series([(0, 100), (40, 130), (70, 110), (110, 140), (150, 120)])
    pivots = zigzag(bars["close"], 0.05)
    kinds = [p.kind for p in pivots]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    # The starting low counts as the first pivot once the rise from it exceeds 5%.
    assert [round(p.price) for p in pivots if p.confirmed] == [100, 130, 110, 140]
    assert not pivots[-1].confirmed and round(pivots[-1].price) == 120


def test_double_top_confirmed():
    # Rise to 120, dip to 105, back to 120, then fall through 105.
    bars = path_series([(0, 90), (60, 120), (90, 105), (120, 120), (170, 95)])
    found = types_of(detect(bars, "daily"))
    assert "double_top" in found
    top = found["double_top"][-1]
    assert top["status"] == "confirmed"
    assert top["implied_bias"] == "bearish"
    assert top["metrics"]["confirmation_level"] == 105.0


def test_double_top_pending_when_trough_not_broken():
    bars = path_series([(0, 90), (60, 120), (90, 105), (120, 119.5), (140, 110)])
    tops = types_of(detect(bars, "daily")).get("double_top", [])
    assert tops and tops[-1]["status"] == "pending"


def test_double_bottom_confirmed():
    bars = path_series([(0, 130), (60, 100), (90, 115), (120, 100.5), (170, 125)])
    bottoms = types_of(detect(bars, "daily")).get("double_bottom", [])
    assert bottoms and bottoms[-1]["status"] == "confirmed"
    assert bottoms[-1]["implied_bias"] == "bullish"


def test_head_and_shoulders_top_confirmed():
    bars = path_series([(0, 90), (40, 110), (60, 100), (90, 125), (120, 100),
                        (150, 111), (200, 88)])
    hs = types_of(detect(bars, "daily")).get("head_and_shoulders_top", [])
    assert hs, "head and shoulders not detected"
    assert hs[-1]["status"] == "confirmed"
    assert [p["label"] for p in hs[-1]["points"]][2] == "head"


def test_inverse_head_and_shoulders():
    bars = path_series([(0, 130), (40, 110), (60, 120), (90, 95), (120, 120),
                        (150, 109), (200, 135)])
    assert "inverse_head_and_shoulders" in types_of(detect(bars, "daily"))


def test_ascending_triangle_formation():
    # Flat highs near 120, rising lows 100 -> 106 -> 112, price still inside.
    bars = path_series([(0, 95), (30, 120), (50, 100), (80, 120.3), (100, 106),
                        (130, 119.8), (150, 112), (160, 116)])
    formation = [c for c in detect(bars, "daily")["candidates"] if c["family"] == "formation"]
    assert formation and formation[0]["type"] == "ascending_triangle"
    assert formation[0]["implied_bias"] == "bullish"


def test_triangle_fit_ignores_old_starting_low():
    # Regression for the eval case the Chart Analyst rejected: the series' $90 starting
    # low must not be fitted into the triangle's lower line (that made the lines cross).
    bars = path_series([(0, 90), (60, 120), (80, 108), (120, 120.3), (140, 112),
                        (180, 119.8), (200, 115.5), (215, 118)], noise=0.003, seed=7)
    formation = [c for c in detect(bars, "daily")["candidates"] if c["family"] == "formation"]
    assert formation and formation[0]["type"] == "ascending_triangle"
    m = formation[0]["metrics"]
    assert m["upper_line_now"] > m["lower_line_now"] and m["width_ratio_end_vs_start"] > 0
    assert formation[0]["status"] == "forming"


def test_flat_line_is_not_a_trendline():
    # Peaks at 120.0 and 119.9 are a horizontal level, not a falling resistance trendline.
    bars = path_series([(0, 100), (40, 120), (70, 108), (110, 119.9), (140, 109), (160, 114)])
    types = {c["type"] for c in detect(bars, "daily")["candidates"]}
    assert "falling_resistance_trendline" not in types


def test_clean_uptrend_has_no_reversal_patterns_and_reads_uptrend():
    bars = path_series([(0, 100), (400, 200)], noise=0.002, seed=3)
    report = detect(bars, "daily")
    assert not [c for c in report["candidates"] if c["family"] == "reversal"]
    assert report["trend"]["regime"] == "uptrend"


def test_trend_regime_downtrend_and_death_cross():
    bars = path_series([(0, 100), (250, 160), (500, 90)])
    trend = trend_regime(bars, *PARAMS["daily"]["sma"])
    assert trend["regime"] == "downtrend"
    assert trend["last_cross"]["type"] == "death_cross"


def test_support_and_resistance_sides():
    bars = path_series([(0, 100), (30, 120), (60, 105), (90, 120.5), (120, 104.5), (150, 112)])
    report = detect(bars, "daily")
    levels = [c for c in report["candidates"] if c["family"] == "level"]
    for lv in levels:
        expected = "support" if lv["metrics"]["level"] < report["last_close"] else "resistance"
        assert lv["type"] == expected
    assert any(lv["metrics"]["touches"] >= 2 for lv in levels)


def test_every_candidate_has_id_shapes_and_summary():
    bars = path_series([(0, 90), (60, 120), (90, 105), (120, 120), (170, 95)])
    for cand in detect(bars, "daily")["candidates"]:
        assert cand["id"].startswith("C") and cand["shapes"] and cand["summary"]

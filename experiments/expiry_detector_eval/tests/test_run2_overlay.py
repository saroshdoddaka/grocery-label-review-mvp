from pathlib import Path

import pytest

from experiments.expiry_detector_eval.run2_overlay import (
    OCRLine,
    box_iou,
    clip_box,
    detection_overlay_coverage,
    group_overlay_lines,
    locate_overlay,
    overlay_location,
    score_overlay_line,
)


def test_overlay_text_signals_recognize_timestamp_and_address():
    score, reasons = score_overlay_line("Jul 13, 2026 at 11:44:16 AM")
    assert score >= 7
    assert {"month_day_year", "at", "clock_time", "am_pm"}.issubset(reasons)
    address_score, address_reasons = score_overlay_line("5178 McGinnis Ferry Rd")
    assert address_score >= 2
    assert "street_address" in address_reasons
    state_score, state_reasons = score_overlay_line("Alpharetta GA 30005")
    assert state_score >= 2
    assert "state_zip" in state_reasons
    assert "united_states" in score_overlay_line("Unlted States")[1]


def test_overlay_lines_group_when_stacked_and_support_any_corner():
    lines = [
        OCRLine("Jul 13, 2026 at 11:44:16 AM", 0.9, (900, 40, 1200, 80)),
        OCRLine("5178 McGinnis Ferry Rd", 0.9, (900, 85, 1200, 120)),
        OCRLine("Alpharetta GA 30005", 0.9, (900, 125, 1200, 160)),
        OCRLine("United States", 0.9, (900, 165, 1200, 200)),
    ]
    located = locate_overlay(lines, grouping_distance=60, score_threshold=4, partial_threshold=2)
    assert located["status"] == "found"
    assert len(located["lines"]) == 4
    assert overlay_location(located["bbox"], 1280, 720) == "top-right"
    assert len(group_overlay_lines(lines, 60)) == 1


def test_partial_overlay_and_fail_safe_not_found():
    partial = locate_overlay([OCRLine("Jul 13, 2026 at 11:44:16 AM", None, (0, 0, 250, 40))], 50, 10, 2)
    assert partial["status"] == "partial"
    missing = locate_overlay([OCRLine("Ingredients: water, fruit", None, (0, 0, 300, 40))], 50, 4, 2)
    assert missing["status"] == "not_found"
    assert missing["filter_bbox"] is None


def test_coordinate_clipping_and_overlay_overlap_metrics():
    assert clip_box((-10, 5, 120, 90), 100, 80) == (0, 5, 100, 80)
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(1 / 7)
    assert detection_overlay_coverage((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)

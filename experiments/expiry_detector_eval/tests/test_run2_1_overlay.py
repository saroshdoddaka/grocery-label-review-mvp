from PIL import Image

from experiments.expiry_detector_eval.run2_1_overlay import (
    _corner_boxes,
    _dedupe_lines,
    _variants,
    OCRLine,
)


def test_corner_search_covers_all_four_overlapping_zones():
    zones = _corner_boxes(1000, 800, 0.55, 0.45, 0.15)
    assert set(zones) == {"top-left", "top-right", "bottom-left", "bottom-right"}
    assert zones["top-left"][2] > 550
    assert zones["top-right"][0] < 450
    assert zones["bottom-left"][1] < 440
    assert zones["bottom-right"][1] < 440


def test_variants_include_requested_multiscale_and_preprocessing_families():
    variants = dict((name, (image, scale)) for name, image, scale in _variants(Image.new("RGB", (100, 80), "white")))
    assert {"original_2x", "original_3x", "grayscale", "contrast_grayscale", "sharpened", "clahe", "white_text_emphasis", "adaptive_threshold", "inverted_adaptive_threshold", "edge_enhanced"} == set(variants)
    assert variants["original_2x"][1] == 2.0
    assert variants["original_3x"][0].size == (300, 240)


def test_ocr_line_deduplication_removes_duplicate_variant_boxes():
    lines = [
        OCRLine("Jul 13, 2026 at 11:44:16 AM", 0.8, (10, 10, 200, 40)),
        OCRLine("Jul 13, 2026 at 11:44:16 AM", 0.9, (11, 11, 201, 41)),
        OCRLine("5178 McGinnis Ferry Rd", 0.7, (10, 50, 200, 80)),
    ]
    deduped = _dedupe_lines(lines)
    assert len(deduped) == 2
    assert max(line.confidence for line in deduped if "Jul" in line.text) == 0.9

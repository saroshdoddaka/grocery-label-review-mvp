"""Generate a baseline versus Run 2A versus Run 2B comparison after review."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def _yes(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() == "yes"


def _rate(rows: list[dict[str, str]], key: str, denominator_key: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if denominator_key is None or _yes(row, denominator_key)]
    return {"numerator": sum(_yes(row, key) for row in selected), "denominator": len(selected), "rate": sum(_yes(row, key) for row in selected) / len(selected) if selected else None}


def build_comparison(baseline: dict[str, dict[str, str]], run2: dict[str, dict[str, str]], overlay: dict[str, dict[str, str]], run2_summary: dict[str, dict[str, str]]) -> dict[str, Any]:
    names = sorted(set(baseline) & set(run2))
    joined = [{"filename": name, "baseline": baseline[name], "run2": run2[name]} for name in names]
    baseline_visible = [item["baseline"] for item in joined if _yes(item["baseline"], "date_visible")]
    run2_visible = [item["run2"] for item in joined if _yes(item["run2"], "product_date_visible")]
    result = {
        "images_compared": len(joined),
        "baseline_full_date_capture": _rate(baseline_visible, "full_date_inside_crop"),
        "run_2a_full_date_capture": _rate(run2_visible, "run_2a_full_date_inside_crop"),
        "run_2b_full_date_capture": _rate(run2_visible, "run_2b_full_date_inside_crop"),
        "baseline_images_with_false_positives": sum(_yes(item["baseline"], "false_positive") for item in joined),
        "run_2a_images_with_false_positives": sum(_yes(item["run2"], "run_2a_false_positive") for item in joined),
        "run_2b_images_with_false_positives": sum(_yes(item["run2"], "run_2b_false_positive") for item in joined),
        "timestamp_related_detections_removed": None,
        "legitimate_product_date_detections_accidentally_removed_image_level_proxy": sum(_yes(item["baseline"], "correct_region_found") and not _yes(item["run2"], "run_2a_correct_region_found") for item in joined),
        "masking_newly_detected_correct_product_date_image_level_proxy": sum(not _yes(item["baseline"], "correct_region_found") and _yes(item["run2"], "run_2b_correct_region_found") for item in joined),
        "masking_caused_previous_correct_detection_to_disappear_image_level_proxy": sum(_yes(item["baseline"], "correct_region_found") and not _yes(item["run2"], "run_2b_correct_region_found") for item in joined),
        "overlay_localization_success": {"numerator": sum(overlay.get(name, {}).get("overlay_found", "").lower() == "yes" for name in names), "denominator": len(names)},
        "complete_overlay_coverage": {"numerator": sum(overlay.get(name, {}).get("full_overlay_covered", "").lower() == "yes" for name in names), "denominator": len(names)},
        "product_date_accidental_mask_rate": {"numerator": sum(overlay.get(name, {}).get("product_date_accidentally_covered", "").lower() == "yes" for name in names), "denominator": len(names)},
    }
    overlay_success = result["overlay_localization_success"]
    result["overlay_localization_success"]["rate"] = overlay_success["numerator"] / overlay_success["denominator"] if overlay_success["denominator"] else None
    complete = result["complete_overlay_coverage"]
    result["complete_overlay_coverage"]["rate"] = complete["numerator"] / complete["denominator"] if complete["denominator"] else None
    accidental = result["product_date_accidental_mask_rate"]
    result["product_date_accidental_mask_rate"]["rate"] = accidental["numerator"] / accidental["denominator"] if accidental["denominator"] else None
    times = []
    for name, row in run2_summary.items():
        try:
            times.append(float(row.get("overlay_processing_time", 0)) + float(row.get("masked_inference_time", 0)) - float(row.get("raw_inference_time", 0)))
        except (TypeError, ValueError):
            pass
    result["average_added_processing_time"] = statistics.mean(times) if times else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-review", required=True)
    parser.add_argument("--run2-review", required=True)
    parser.add_argument("--overlay-review", required=True)
    parser.add_argument("--run2-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    comparison = build_comparison(_read(Path(args.baseline_review)), _read(Path(args.run2_review)), _read(Path(args.overlay_review)), _read(Path(args.run2_summary)))
    Path(args.output).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Comparison written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

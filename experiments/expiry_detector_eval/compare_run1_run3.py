"""Compare reviewed Run 1 and Run 3 output directories without treating blanks as failures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable


COMPARISON_WARNING = (
    "Run 1 and Run 3 use different image samples unless the same products were deliberately "
    "photographed both with and without the overlay. Differences therefore cannot automatically "
    "be attributed only to removing the overlay. If paired clean/timestamped versions are unavailable, "
    "this is an unpaired baseline comparison rather than a controlled causal test."
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "").strip().lower()
        if value:
            return value
    return ""


def _answer(row: dict[str, str], *names: str) -> bool | None:
    value = _value(row, *names)
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def _rate(rows: list[dict[str, str]], field_names: tuple[str, ...], eligible: Callable[[dict[str, str]], bool]) -> dict[str, Any]:
    answers = [_answer(row, *field_names) for row in rows if eligible(row)]
    reviewed = [answer for answer in answers if answer is not None]
    return {
        "numerator": sum(answer is True for answer in reviewed),
        "denominator": len(reviewed),
        "rate": sum(answer is True for answer in reviewed) / len(reviewed) if reviewed else None,
        "unreviewed_eligible_rows": len(answers) - len(reviewed),
    }


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Calculate review and runtime metrics for a completed evaluator output directory."""
    reviews = _read_csv(run_dir / "manual_review.csv")
    images = _read_csv(run_dir / "image_summary.csv")
    visible = lambda row: _answer(row, "product_date_visible", "date_visible") is True
    inference_times = _float_values(images, "inference_time")
    return {
        "run_directory": str(run_dir),
        "review_rows": len(reviews),
        "correct_region_rate": _rate(reviews, ("correct_region_found",), visible),
        "full_date_capture_rate": _rate(reviews, ("full_date_inside_crop",), visible),
        "false_positive_rate": _rate(reviews, ("false_positive_present", "false_positive"), lambda row: True),
        "detections_per_image": sum(int(row.get("detection_count", 0) or 0) for row in images) / len(images) if images else None,
        "mean_inference_time": statistics.mean(inference_times) if inference_times else None,
        "median_inference_time": statistics.median(inference_times) if inference_times else None,
        "image_count": len(images),
    }


def compare(run1_dir: Path, run3_dir: Path) -> dict[str, Any]:
    return {"comparison_type": "unpaired baseline comparison unless paired products are documented", "important_interpretation_warning": COMPARISON_WARNING, "run_1": summarize_run(run1_dir), "run_3": summarize_run(run3_dir)}


def _write_table(path: Path, comparison: dict[str, Any]) -> None:
    metrics = ["correct_region_rate", "full_date_capture_rate", "false_positive_rate", "detections_per_image", "mean_inference_time", "median_inference_time"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "run_1", "run_3", "run_1_denominator", "run_3_denominator"])
        writer.writeheader()
        for metric in metrics:
            first, third = comparison["run_1"][metric], comparison["run_3"][metric]
            writer.writerow({
                "metric": metric,
                "run_1": first.get("rate") if isinstance(first, dict) else first,
                "run_3": third.get("rate") if isinstance(third, dict) else third,
                "run_1_denominator": first.get("denominator", "") if isinstance(first, dict) else "",
                "run_3_denominator": third.get("denominator", "") if isinstance(third, dict) else "",
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1-dir", required=True, type=Path)
    parser.add_argument("--run3-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        comparison = compare(args.run1_dir, args.run3_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        _write_table(args.output_dir / "comparison.csv", comparison)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Comparison could not run: {exc}")
        return 2
    print(COMPARISON_WARNING)
    print(f"Comparison written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv

from experiments.expiry_detector_eval.compare_run1_run3 import compare


def _write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_directory(tmp_path, name, review_rows, image_rows):
    run_dir = tmp_path / name
    run_dir.mkdir()
    _write(run_dir / "manual_review.csv", list(review_rows[0]), review_rows)
    _write(run_dir / "image_summary.csv", ["filename", "detection_count", "inference_time"], image_rows)
    return run_dir


def test_comparison_uses_only_completed_review_answers(tmp_path):
    run1 = _run_directory(tmp_path, "run1", [
        {"filename": "a.jpg", "date_visible": "Yes", "correct_region_found": "Yes", "full_date_inside_crop": "", "false_positive": "No"},
        {"filename": "b.jpg", "date_visible": "", "correct_region_found": "No", "full_date_inside_crop": "No", "false_positive": ""},
    ], [{"filename": "a.jpg", "detection_count": "2", "inference_time": "0.2"}, {"filename": "b.jpg", "detection_count": "0", "inference_time": "0.4"}])
    run3 = _run_directory(tmp_path, "run3", [
        {"filename": "a.jpg", "product_date_visible": "Yes", "correct_region_found": "Yes", "full_date_inside_crop": "Yes", "false_positive_present": "No", "number_of_real_date_regions": "1", "notes": ""},
    ], [{"filename": "a.jpg", "detection_count": "1", "inference_time": "0.1"}])
    result = compare(run1, run3)
    assert result["run_1"]["correct_region_rate"] == {"numerator": 1, "denominator": 1, "rate": 1.0, "unreviewed_eligible_rows": 0}
    assert result["run_1"]["full_date_capture_rate"]["denominator"] == 0
    assert result["run_1"]["full_date_capture_rate"]["unreviewed_eligible_rows"] == 1
    assert result["run_3"]["detections_per_image"] == 1.0

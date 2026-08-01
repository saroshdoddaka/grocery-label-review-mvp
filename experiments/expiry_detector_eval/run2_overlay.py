"""Run 2: locate the research-photo timestamp overlay and test two suppressors."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

try:
    from experiments.expiry_detector_eval.evaluate import (
        MODEL_FILENAME, MODEL_REPO, _class_name, _open_image, _to_list,
        build_output_paths, discover_images, padded_box, resolve_device,
    )
except ImportError:  # Allows direct execution from the experiment directory.
    from evaluate import (  # type: ignore
        MODEL_FILENAME, MODEL_REPO, _class_name, _open_image, _to_list,
        build_output_paths, discover_images, padded_box, resolve_device,
    )

OVERLAY_REVIEW_FIELDS = ["filename", "overlay_present", "overlay_found", "full_overlay_covered", "product_date_accidentally_covered", "overlay_location", "notes"]
EXPIRY_REVIEW_FIELDS = ["filename", "product_date_visible", "baseline_correct_region_found", "baseline_full_date_inside_crop", "run_2a_correct_region_found", "run_2a_full_date_inside_crop", "run_2a_false_positive", "run_2b_correct_region_found", "run_2b_full_date_inside_crop", "run_2b_false_positive", "notes"]
OVERLAY_CSV_FIELDS = ["filename", "overlay_status", "overlay_score", "ocr_backend", "ocr_text", "ocr_line_details", "ocr_warnings", "overlay_x1", "overlay_y1", "overlay_x2", "overlay_y2", "overlay_line_count", "overlay_location", "overlay_processing_time", "original_path", "ocr_boxes_path", "overlay_preview_path", "mask_path", "masked_image_path"]
DETECTION_CSV_FIELDS = ["filename", "detection_index", "confidence", "class_id", "class_name", "x1", "y1", "x2", "y2", "padded_x1", "padded_y1", "padded_x2", "padded_y2", "iou_with_overlay", "detection_coverage_by_overlay", "kept_after_filter", "removed", "removal_reason", "inference_time", "annotated_image_path", "crop_path", "device_used"]
SUMMARY_FIELDS = ["filename", "overlay_status", "overlay_score", "raw_detection_count", "run_2a_filtered_detection_count", "run_2b_detection_count", "raw_inference_time", "masked_inference_time", "overlay_processing_time", "device", "overlay_preview_path", "mask_path", "masked_image_path"]

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
STREET_SUFFIXES = r"(?:rd|road|st|street|dr|drive|ave|avenue|blvd|boulevard|lane|ln|pkwy|parkway|highway|hwy|court|ct|circle|cir)"
DATE_RE = re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:,|\s)\s*\d{{4}}\b", re.I)
TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b", re.I)
AMPM_RE = re.compile(r"\b(?:a\.?m\.?|p\.?m\.?)\b", re.I)
ADDRESS_RE = re.compile(rf"\b\d{{1,6}}\s+[A-Za-z0-9 .'-]+\s+{STREET_SUFFIXES}\b", re.I)
STATE_ZIP_RE = re.compile(r"\b[A-Za-z .'-]+,?\s+[A-Za-z]{2}\s+\d{5}(?:-\d{4})?\b", re.I)
US_RE = re.compile(r"\bun(?:i|l)ted\s+states\b", re.I)


@dataclass
class OCRLine:
    text: str
    confidence: float | None
    bbox: tuple[float, float, float, float]
    score: float = 0.0
    reasons: tuple[str, ...] = ()


def _bbox_from_polygon(polygon: Any) -> tuple[float, float, float, float] | None:
    try:
        points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, IndexError, ValueError):
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", "I").strip())


def score_overlay_line(text: str) -> tuple[float, tuple[str, ...]]:
    """Score semantic overlay signals without requiring every line to match."""
    normalized = _normalize_text(text)
    reasons: list[str] = []
    score = 0.0
    if DATE_RE.search(normalized):
        score += 3.0
        reasons.append("month_day_year")
    if re.search(r"\bat\b", normalized, re.I):
        score += 1.0
        reasons.append("at")
    if TIME_RE.search(normalized):
        score += 2.0
        reasons.append("clock_time")
    if AMPM_RE.search(normalized):
        score += 1.0
        reasons.append("am_pm")
    if ADDRESS_RE.search(normalized):
        score += 2.0
        reasons.append("street_address")
    if STATE_ZIP_RE.search(normalized):
        score += 2.0
        reasons.append("state_zip")
    if US_RE.search(normalized):
        score += 2.0
        reasons.append("united_states")
    return score, tuple(reasons)


def _tesseract_region(image: Image.Image, offset_x: int, offset_y: int, psm: int) -> list[OCRLine]:
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=Output.DICT)
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for index, text in enumerate(data.get("text", [])):
        text = _normalize_text(text)
        if not text:
            continue
        key = (int(data.get("block_num", [0])[index]), int(data.get("par_num", [0])[index]), int(data.get("line_num", [0])[index]))
        grouped.setdefault(key, []).append({"text": text, "left": data["left"][index], "top": data["top"][index], "width": data["width"][index], "height": data["height"][index], "conf": data.get("conf", [None])[index]})
    lines: list[OCRLine] = []
    for words in grouped.values():
        x1 = min(word["left"] for word in words) + offset_x
        y1 = min(word["top"] for word in words) + offset_y
        x2 = max(word["left"] + word["width"] for word in words) + offset_x
        y2 = max(word["top"] + word["height"] for word in words) + offset_y
        confidences = [float(word["conf"]) for word in words if str(word["conf"]) not in {"", "-1", "None"}]
        lines.append(OCRLine(" ".join(word["text"] for word in words), sum(confidences) / len(confidences) / 100 if confidences else None, (x1, y1, x2, y2)))
    return lines


def _ocr_line_overlap(a: OCRLine, b: OCRLine) -> float:
    return box_iou(a.bbox, b.bbox)


def _run_tesseract(path: Path) -> tuple[list[OCRLine], str]:
    """Run OCR over overlapping tiles so large, outlined overlay text is not lost."""
    image = _open_image(path)
    width, height = image.size
    regions = [(0, 0, width, height)]
    half_width, half_height = width // 2, height // 2
    overlap_x, overlap_y = width // 8, height // 8
    for x1, y1, x2, y2 in (
        (0, 0, half_width + overlap_x, half_height + overlap_y),
        (half_width - overlap_x, 0, width, half_height + overlap_y),
        (0, half_height - overlap_y, half_width + overlap_x, height),
        (half_width - overlap_x, half_height - overlap_y, width, height),
        (0, 0, width, half_height + overlap_y),
        (0, half_height - overlap_y, width, height),
    ):
        regions.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    collected: list[OCRLine] = []
    for x1, y1, x2, y2 in regions:
        collected.extend(_tesseract_region(image.crop((x1, y1, x2, y2)), x1, y1, 12))
    deduplicated: list[OCRLine] = []
    for line in sorted(collected, key=lambda item: (item.confidence if item.confidence is not None else 0), reverse=True):
        if any(_ocr_line_overlap(line, existing) > 0.6 and _normalize_text(line.text).lower() == _normalize_text(existing.text).lower() for existing in deduplicated):
            continue
        deduplicated.append(line)
    return deduplicated, "tesseract"


def _run_paddle(path: Path) -> tuple[list[OCRLine], str]:
    from src.ocr.paddle_engine import run_ocr

    result = run_ocr(str(path))
    lines = []
    for item in result.get("lines", []):
        bbox = _bbox_from_polygon(item.get("bbox"))
        if bbox and item.get("text"):
            lines.append(OCRLine(_normalize_text(str(item["text"])), item.get("confidence"), bbox))
    if not lines and result.get("warnings"):
        raise RuntimeError("; ".join(result["warnings"]))
    return lines, "paddle"


def run_overlay_ocr(path: Path, backend: str) -> tuple[list[OCRLine], str, list[str]]:
    warnings: list[str] = []
    attempts = [backend] if backend != "auto" else ["tesseract", "paddle"]
    for candidate in attempts:
        try:
            return (*(_run_tesseract(path) if candidate == "tesseract" else _run_paddle(path)), warnings)
        except Exception as exc:
            warnings.append(f"{candidate}: {exc}")
    raise RuntimeError("No overlay OCR backend succeeded: " + " | ".join(warnings))


def _union(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else None


def _distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    horizontal = max(a[0] - b[2], b[0] - a[2], 0)
    vertical = max(a[1] - b[3], b[1] - a[3], 0)
    return max(horizontal, vertical)


def group_overlay_lines(lines: list[OCRLine], grouping_distance: float = 120.0) -> list[list[OCRLine]]:
    """Group nearby OCR lines; no corner assumption is used."""
    candidates = [line for line in lines if line.score > 0]
    groups: list[list[OCRLine]] = []
    for line in sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0])):
        placed = False
        for group in groups:
            group_box = _union(item.bbox for item in group)
            if group_box and _distance(group_box, line.bbox) <= grouping_distance:
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])
    changed = True
    while changed:
        changed = False
        for index in range(len(groups) - 1, -1, -1):
            for other in range(index - 1, -1, -1):
                a = _union(line.bbox for line in groups[index])
                b = _union(line.bbox for line in groups[other])
                if a and b and _distance(a, b) <= grouping_distance:
                    groups[other].extend(groups.pop(index))
                    changed = True
                    break
            if changed:
                break
    return groups


def locate_overlay(lines: list[OCRLine], grouping_distance: float, score_threshold: float, partial_threshold: float) -> dict[str, Any]:
    scored: list[OCRLine] = []
    for line in lines:
        line.score, line.reasons = score_overlay_line(line.text)
        scored.append(line)
    groups = group_overlay_lines(scored, grouping_distance)
    candidates = []
    for group in groups:
        box = _union(line.bbox for line in group)
        score = sum(line.score for line in group)
        if box:
            candidates.append((score, box, group))
    if not candidates:
        return {"status": "not_found", "score": 0.0, "bbox": None, "filter_bbox": None, "lines": [], "location": "unknown"}
    score, bbox, group = max(candidates, key=lambda item: item[0])
    status = "found" if score >= score_threshold else "partial" if score >= partial_threshold else "uncertain"
    confident_lines = [line for line in group if line.score >= 1.0]
    filter_bbox = _union(line.bbox for line in confident_lines)
    if status == "uncertain":
        filter_bbox = None
    return {"status": status, "score": round(score, 3), "bbox": bbox, "filter_bbox": filter_bbox, "lines": group, "location": overlay_location(bbox)}


def overlay_location(box: tuple[float, float, float, float] | None, width: int | None = None, height: int | None = None) -> str:
    if not box or not width or not height:
        return "unknown"
    cx = (box[0] + box[2]) / 2 / width
    cy = (box[1] + box[3]) / 2 / height
    return ("top-" if cy < 0.5 else "bottom-") + ("left" if cx < 0.5 else "right")


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    return max(0, min(width, box[0])), max(0, min(height, box[1])), max(0, min(width, box[2])), max(0, min(height, box[3]))


def expanded_box(box: tuple[float, float, float, float], padding_percent: float, width: int, height: int) -> tuple[int, int, int, int]:
    return padded_box(box, padding_percent, width, height)


def make_overlay_mask(image_size: tuple[int, int], lines: list[OCRLine], padding_percent: float) -> Any:
    import numpy as np
    import cv2

    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = expanded_box(line.bbox, padding_percent, width, height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def apply_mask(image: Image.Image, mask: Any, method: str) -> Image.Image:
    import numpy as np
    if not mask.any():
        return image.copy()
    if method == "fill":
        array = np.asarray(image).copy()
        fill = array[max(0, int(mask.shape[0] * 0.05)), max(0, int(mask.shape[1] * 0.05))].tolist()
        array[mask > 0] = fill
        return Image.fromarray(array)
    import cv2

    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def box_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if not a or not b:
        return 0.0
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def detection_overlay_coverage(detection: tuple[float, float, float, float], overlay: tuple[float, float, float, float] | None) -> float:
    if not overlay:
        return 0.0
    ix1, iy1, ix2, iy2 = max(detection[0], overlay[0]), max(detection[1], overlay[1]), min(detection[2], overlay[2]), min(detection[3], overlay[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, detection[2] - detection[0]) * max(0.0, detection[3] - detection[1])
    return intersection / area if area else 0.0


def predict(model: Any, source: Path, conf: float, imgsz: int, device: str) -> tuple[list[dict[str, Any]], Any, float]:
    started = time.perf_counter()
    result = model.predict(source=str(source), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    elapsed = time.perf_counter() - started
    boxes = getattr(result, "boxes", None)
    xyxy = _to_list(boxes.xyxy) if boxes is not None else []
    confidences = _to_list(boxes.conf) if boxes is not None else []
    classes = _to_list(boxes.cls) if boxes is not None else []
    names = getattr(result, "names", getattr(model, "names", {}))
    detections = []
    for index, (box, confidence, class_value) in enumerate(zip(xyxy, confidences, classes), start=1):
        detections.append({"source_index": index, "box": tuple(float(value) for value in box), "confidence": float(confidence), "class_id": int(class_value), "class_name": _class_name(names, int(class_value))})
    return detections, result, elapsed


def _draw_detections(image: Image.Image, detections: list[dict[str, Any]], color: str = "blue") -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        draw.rectangle(box, outline=color, width=max(2, round(min(image.size) / 500)))
        draw.text((box[0], max(0, box[1] - 18)), f"{detection['class_name']} {detection['confidence']:.2f}", fill=color)
    return output


def _save_detection_outputs(image: Image.Image, detections: list[dict[str, Any]], input_folder: Path, run_dir: Path, folder_name: str, relative: Path, padding: float, annotated_name: str | None = None) -> tuple[str, dict[int, str]]:
    annotated = run_dir / folder_name / relative.parent / (annotated_name or f"{relative.stem}.jpg")
    annotated.parent.mkdir(parents=True, exist_ok=True)
    _draw_detections(image, detections).save(annotated)
    crop_paths: dict[int, str] = {}
    for index, detection in enumerate(detections, start=1):
        crop = expanded_box(detection["box"], padding, image.width, image.height)
        path = run_dir / ("run_2a_crops" if folder_name.startswith("run_2a") else "run_2b_crops") / relative.parent / f"{relative.stem}_detection_{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(crop).save(path)
        crop_paths[detection.get("source_index", index)] = str(path.relative_to(run_dir))
    return str(annotated.relative_to(run_dir)), crop_paths


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", default="experiments/expiry_detector_eval/outputs")
    parser.add_argument("--run-name", default="run_2_overlay_suppression")
    parser.add_argument("--overlay-ocr-backend", choices=["auto", "tesseract", "paddle"], default="auto")
    parser.add_argument("--overlay-score-threshold", type=float, default=4.0)
    parser.add_argument("--partial-score-threshold", type=float, default=2.0)
    parser.add_argument("--ocr-grouping-distance", type=float, default=120.0)
    parser.add_argument("--line-mask-padding", type=float, default=12.0)
    parser.add_argument("--detection-overlay-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--detection-overlay-coverage-threshold", type=float, default=0.50)
    parser.add_argument("--masking-method", choices=["inpaint", "fill"], default="inpaint")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--padding", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int)
    return parser


def run(args: argparse.Namespace) -> Path:
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_base = Path(args.output_folder).expanduser().resolve()
    if not input_folder.is_dir():
        raise ValueError(f"Input image folder does not exist: {input_folder}")
    images = discover_images(input_folder)[:args.max_images if args.max_images else None]
    if not images:
        raise ValueError(f"No supported images found in {input_folder}")
    device = resolve_device(args.device)
    run_dir = output_base / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    model = YOLO(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME))
    overlay_rows: list[dict[str, Any]] = []
    run2a_rows: list[dict[str, Any]] = []
    run2b_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for image_path in images:
        relative = image_path.relative_to(input_folder)
        image = _open_image(image_path)
        ocr_started = time.perf_counter()
        try:
            lines, backend, warnings = run_overlay_ocr(image_path, args.overlay_ocr_backend)
            for line in lines:
                line.score, line.reasons = score_overlay_line(line.text)
            overlay = locate_overlay(lines, args.ocr_grouping_distance, args.overlay_score_threshold, args.partial_score_threshold)
        except Exception as exc:
            lines, backend, warnings = [], "failed", [str(exc)]
            overlay = {"status": "not_found", "score": 0.0, "bbox": None, "filter_bbox": None, "lines": [], "location": "unknown"}
        overlay_time = time.perf_counter() - ocr_started
        overlay_bbox = overlay["bbox"]
        overlay["location"] = overlay_location(overlay_bbox, image.width, image.height)

        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        for line in lines:
            draw.rectangle(line.bbox, outline="gray", width=2)
            draw.text((line.bbox[0], line.bbox[1]), f"{line.score:.1f} {'/'.join(line.reasons)}", fill="gray")
        preview_path = run_dir / "overlay_previews" / relative.parent / f"{relative.stem}_ocr.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(preview_path)
        highlighted = image.copy()
        highlight_draw = ImageDraw.Draw(highlighted)
        if overlay_bbox:
            highlight_draw.rectangle(overlay_bbox, outline="lime", width=max(4, round(min(image.size) / 250)))
            highlight_draw.text((overlay_bbox[0], max(0, overlay_bbox[1] - 24)), f"overlay {overlay['status']} {overlay['score']:.1f}", fill="lime")
        highlight_path = run_dir / "overlay_previews" / relative.parent / f"{relative.stem}_overlay.jpg"
        highlighted.save(highlight_path)

        mask = make_overlay_mask(image.size, overlay["lines"] if overlay["status"] in {"found", "partial"} else [], args.line_mask_padding)
        mask_path = run_dir / "masks" / relative.parent / f"{relative.stem}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        masked = apply_mask(image, mask, args.masking_method)
        masked_path = run_dir / "masked_images" / relative.parent / f"{relative.stem}.jpg"
        masked_path.parent.mkdir(parents=True, exist_ok=True)
        masked.save(masked_path)

        raw_detections, _, raw_time = predict(model, image_path, args.confidence, args.imgsz, device)
        filtered: list[dict[str, Any]] = []
        for detection in raw_detections:
            iou = box_iou(detection["box"], overlay["filter_bbox"])
            coverage = detection_overlay_coverage(detection["box"], overlay["filter_bbox"])
            remove = overlay["filter_bbox"] is not None and (iou >= args.detection_overlay_overlap_threshold or coverage >= args.detection_overlay_coverage_threshold)
            detection["iou"] = round(iou, 6)
            detection["coverage"] = round(coverage, 6)
            detection["removed"] = remove
            detection["reason"] = "overlap_with_overlay" if remove else ""
            if not remove:
                filtered.append(detection)
        raw_annotated = _draw_detections(image, raw_detections, "blue")
        raw_preview_path = run_dir / "overlay_previews" / relative.parent / f"{relative.stem}_raw_yolo.jpg"
        raw_annotated.save(raw_preview_path)
        run2a_annotated, run2a_crops = _save_detection_outputs(image, filtered, input_folder, run_dir, "run_2a_filtered_annotated", relative, args.padding)
        masked_detections, _, masked_time = predict(model, masked_path, args.confidence, args.imgsz, device)
        run2b_annotated, run2b_crops = _save_detection_outputs(image, masked_detections, input_folder, run_dir, "run_2b_masked_yolo_annotated", relative, args.padding)
        for index, detection in enumerate(raw_detections, start=1):
            crop = expanded_box(detection["box"], args.padding, image.width, image.height)
            run2a_rows.append({"filename": _relative(image_path, input_folder), "detection_index": index, "confidence": round(detection["confidence"], 6), "class_id": detection["class_id"], "class_name": detection["class_name"], "x1": round(detection["box"][0], 3), "y1": round(detection["box"][1], 3), "x2": round(detection["box"][2], 3), "y2": round(detection["box"][3], 3), "padded_x1": crop[0], "padded_y1": crop[1], "padded_x2": crop[2], "padded_y2": crop[3], "iou_with_overlay": detection.get("iou", 0), "detection_coverage_by_overlay": detection.get("coverage", 0), "kept_after_filter": not detection["removed"], "removed": detection["removed"], "removal_reason": detection["reason"], "inference_time": round(raw_time, 6), "annotated_image_path": run2a_annotated, "crop_path": run2a_crops.get(index, ""), "device_used": device})
        for index, detection in enumerate(masked_detections, start=1):
            crop = expanded_box(detection["box"], args.padding, image.width, image.height)
            run2b_rows.append({"filename": _relative(image_path, input_folder), "detection_index": index, "confidence": round(detection["confidence"], 6), "class_id": detection["class_id"], "class_name": detection["class_name"], "x1": round(detection["box"][0], 3), "y1": round(detection["box"][1], 3), "x2": round(detection["box"][2], 3), "y2": round(detection["box"][3], 3), "padded_x1": crop[0], "padded_y1": crop[1], "padded_x2": crop[2], "padded_y2": crop[3], "iou_with_overlay": "", "detection_coverage_by_overlay": "", "kept_after_filter": "", "removed": "", "removal_reason": "", "inference_time": round(masked_time, 6), "annotated_image_path": run2b_annotated, "crop_path": run2b_crops.get(index, ""), "device_used": device})
        overlay_rows.append({"filename": _relative(image_path, input_folder), "overlay_status": overlay["status"], "overlay_score": overlay["score"], "ocr_backend": backend, "ocr_text": " ".join(line.text for line in overlay["lines"]), "ocr_line_details": json.dumps([{"text": line.text, "score": line.score, "reasons": list(line.reasons), "bbox": line.bbox} for line in overlay["lines"]]), "ocr_warnings": " | ".join(warnings), "overlay_x1": overlay_bbox[0] if overlay_bbox else "", "overlay_y1": overlay_bbox[1] if overlay_bbox else "", "overlay_x2": overlay_bbox[2] if overlay_bbox else "", "overlay_y2": overlay_bbox[3] if overlay_bbox else "", "overlay_line_count": len(overlay["lines"]), "overlay_location": overlay["location"], "overlay_processing_time": round(overlay_time, 6), "original_path": _relative(image_path, input_folder), "ocr_boxes_path": str(preview_path.relative_to(run_dir)), "overlay_preview_path": str(highlight_path.relative_to(run_dir)), "mask_path": str(mask_path.relative_to(run_dir)), "masked_image_path": str(masked_path.relative_to(run_dir))})
        summary_rows.append({"filename": _relative(image_path, input_folder), "overlay_status": overlay["status"], "overlay_score": overlay["score"], "raw_detection_count": len(raw_detections), "run_2a_filtered_detection_count": len(filtered), "run_2b_detection_count": len(masked_detections), "raw_inference_time": round(raw_time, 6), "masked_inference_time": round(masked_time, 6), "overlay_processing_time": round(overlay_time, 6), "device": device, "overlay_preview_path": str(highlight_path.relative_to(run_dir)), "mask_path": str(mask_path.relative_to(run_dir)), "masked_image_path": str(masked_path.relative_to(run_dir))})
    _write_csv(run_dir / "overlay_detection.csv", OVERLAY_CSV_FIELDS, overlay_rows)
    _write_csv(run_dir / "run_2a_detections.csv", DETECTION_CSV_FIELDS, run2a_rows)
    _write_csv(run_dir / "run_2b_detections.csv", DETECTION_CSV_FIELDS, run2b_rows)
    _write_csv(run_dir / "image_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(run_dir / "manual_overlay_review.csv", OVERLAY_REVIEW_FIELDS, [{"filename": row["filename"]} for row in overlay_rows])
    _write_csv(run_dir / "manual_expiry_review.csv", EXPIRY_REVIEW_FIELDS, [{"filename": row["filename"]} for row in overlay_rows])
    report = {"run_name": args.run_name, "input_folder": str(input_folder), "total_images_processed": len(images), "overlay_status_counts": {status: sum(row["overlay_status"] == status for row in overlay_rows) for status in ("found", "partial", "uncertain", "not_found")}, "raw_detections": len(run2a_rows), "run_2a_filtered_detections": sum(row["kept_after_filter"] is True for row in run2a_rows), "run_2a_removed_detections": sum(row["removed"] is True for row in run2a_rows), "run_2b_detections": len(run2b_rows), "device_used": device, "confidence": args.confidence, "imgsz": args.imgsz, "crop_padding": args.padding, "masking_method": args.masking_method, "overlay_ocr_backend": args.overlay_ocr_backend, "settings": vars(args)}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run_dir = run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"Run 2 could not start: {exc}")
        return 2
    print(f"Run 2 complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

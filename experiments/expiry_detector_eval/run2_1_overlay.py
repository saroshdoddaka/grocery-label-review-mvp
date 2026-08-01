"""Run 2.1: validate timestamp-overlay localization before YOLO suppression."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    from experiments.expiry_detector_eval.evaluate import _open_image, discover_images
    from experiments.expiry_detector_eval.run2_overlay import (
        OCRLine, _bbox_from_polygon, _normalize_text, _union, box_iou, expanded_box,
        score_overlay_line,
    )
except ImportError:  # Direct execution from the experiment directory.
    from evaluate import _open_image, discover_images  # type: ignore
    from run2_overlay import (  # type: ignore
        OCRLine, _bbox_from_polygon, _normalize_text, _union, box_iou, expanded_box,
        score_overlay_line,
    )

OVERLAY_REVIEW_FIELDS = ["filename", "overlay_corner_correct", "overlay_status_correct", "complete_overlay_covered", "product_content_accidentally_masked", "acceptable_for_yolo_comparison", "notes"]
OVERLAY_FIELDS = ["filename", "selected_corner", "overlay_status", "overlay_score", "recognized_overlay_text", "matched_pattern_groups", "ocr_variant_count", "overlay_line_count", "overlay_x1", "overlay_y1", "overlay_x2", "overlay_y2", "likely_overlay_text_pixels", "covered_overlay_text_pixels", "estimated_mask_coverage", "ocr_processing_time", "mask_path", "preview_path", "masked_image_path", "rejection_reasons"]
VARIANT_FIELDS = ["filename", "corner", "variant", "ocr_line_count", "recognized_text", "ocr_confidence", "original_coordinates", "pattern_groups"]


def _corner_boxes(width: int, height: int, width_fraction: float, height_fraction: float, overlap: float) -> dict[str, tuple[int, int, int, int]]:
    cw, ch = round(width * width_fraction), round(height * height_fraction)
    ox, oy = round(cw * overlap), round(ch * overlap)
    return {
        "top-left": (0, 0, min(width, cw + ox), min(height, ch + oy)),
        "top-right": (max(0, width - cw - ox), 0, width, min(height, ch + oy)),
        "bottom-left": (0, max(0, height - ch - oy), min(width, cw + ox), height),
        "bottom-right": (max(0, width - cw - ox), max(0, height - ch - oy), width, height),
    }


def _variants(crop: Image.Image) -> list[tuple[str, Image.Image, float]]:
    """Return variant name, image, and scale relative to the corner crop."""
    gray = ImageOps.grayscale(crop)
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)
    sharp = contrast.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    rgb = np.asarray(crop)
    import cv2

    gray_array = np.asarray(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_array)
    white = cv2.inRange(rgb, np.array([150, 150, 150], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = cv2.bitwise_and(white, cv2.inRange(hsv, np.array([0, 0, 120], dtype=np.uint8), np.array([180, 100, 255], dtype=np.uint8)))
    adaptive = cv2.adaptiveThreshold(gray_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    inverted = cv2.bitwise_not(adaptive)
    edges = cv2.Canny(gray_array, 50, 150)
    return [
        ("original_2x", crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS), 2.0),
        ("original_3x", crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS), 3.0),
        ("grayscale", gray, 1.0),
        ("contrast_grayscale", contrast, 1.0),
        ("sharpened", sharp, 1.0),
        ("clahe", Image.fromarray(clahe), 1.0),
        ("white_text_emphasis", Image.fromarray(white), 1.0),
        ("adaptive_threshold", Image.fromarray(adaptive), 1.0),
        ("inverted_adaptive_threshold", Image.fromarray(inverted), 1.0),
        ("edge_enhanced", Image.fromarray(edges), 1.0),
    ]


class PaddleAdapter:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self.engine = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True)

    def lines(self, image: Image.Image) -> list[OCRLine]:
        result = list(self.engine.predict(np.asarray(image)))
        lines: list[OCRLine] = []
        for page in result:
            data = getattr(page, "json", {})
            if callable(data):
                data = data()
            if isinstance(data, str):
                data = json.loads(data)
            data = data.get("res", data) if isinstance(data, dict) else {}
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_polys", data.get("rec_boxes", []))
            for index, text in enumerate(texts):
                bbox = _bbox_from_polygon(boxes[index]) if index < len(boxes) else None
                if bbox and text:
                    confidence = float(scores[index]) if index < len(scores) else None
                    lines.append(OCRLine(_normalize_text(str(text)), confidence, bbox))
        return lines


def _tesseract_lines(image: Image.Image) -> list[OCRLine]:
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, config="--psm 12", output_type=Output.DICT)
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for index, text in enumerate(data.get("text", [])):
        text = _normalize_text(text)
        if text:
            key = (int(data.get("block_num", [0])[index]), int(data.get("par_num", [0])[index]), int(data.get("line_num", [0])[index]))
            grouped.setdefault(key, []).append({"text": text, "left": data["left"][index], "top": data["top"][index], "width": data["width"][index], "height": data["height"][index], "conf": data.get("conf", [None])[index]})
    lines = []
    for words in grouped.values():
        box = (min(w["left"] for w in words), min(w["top"] for w in words), max(w["left"] + w["width"] for w in words), max(w["top"] + w["height"] for w in words))
        confidence = [float(w["conf"]) for w in words if str(w["conf"]) not in {"", "-1", "None"}]
        lines.append(OCRLine(" ".join(w["text"] for w in words), sum(confidence) / len(confidence) / 100 if confidence else None, box))
    return lines


def _dedupe_lines(lines: list[OCRLine]) -> list[OCRLine]:
    output: list[OCRLine] = []
    for line in sorted(lines, key=lambda item: item.confidence or 0, reverse=True):
        if any(box_iou(line.bbox, existing.bbox) >= 0.55 and _normalize_text(line.text).lower() == _normalize_text(existing.text).lower() for existing in output):
            continue
        output.append(line)
    return output


def _map_lines(lines: list[OCRLine], crop_box: tuple[int, int, int, int], scale: float) -> list[OCRLine]:
    x_offset, y_offset = crop_box[0], crop_box[1]
    mapped = []
    for line in lines:
        mapped.append(OCRLine(line.text, line.confidence, (line.bbox[0] / scale + x_offset, line.bbox[1] / scale + y_offset, line.bbox[2] / scale + x_offset, line.bbox[3] / scale + y_offset)))
    return mapped


def _candidate_groups(lines: list[OCRLine], grouping_distance: float) -> list[list[OCRLine]]:
    scored = []
    for line in lines:
        line.score, line.reasons = score_overlay_line(line.text)
        if line.score > 0:
            scored.append(line)
    groups: list[list[OCRLine]] = []
    for line in sorted(scored, key=lambda item: (item.bbox[1], item.bbox[0])):
        matching = []
        for index, group in enumerate(groups):
            block = _union(item.bbox for item in group)
            if block and max(block[0] - line.bbox[2], line.bbox[0] - block[2], block[1] - line.bbox[3], line.bbox[1] - block[3], 0) <= grouping_distance:
                matching.append(index)
        if not matching:
            groups.append([line])
        else:
            first = matching[0]
            groups[first].append(line)
            for index in reversed(matching[1:]):
                groups[first].extend(groups.pop(index))
    return groups


def _classify_group(group: list[OCRLine], corner_box: tuple[int, int, int, int], score_threshold: float, minimum_coverage: float, image: Image.Image, line_padding: float) -> dict[str, Any]:
    block = _union(line.bbox for line in group)
    reasons = sorted({reason for line in group for reason in line.reasons})
    score = sum(line.score for line in group) + min(2.0, len(group) * 0.25)
    strong_groups = sum(any(reason in line.reasons for reason in ("month_day_year", "clock_time", "street_address", "state_zip", "united_states")) for line in group)
    stacked = len(group) >= 2 and block is not None and (block[3] - block[1]) >= max(30, max(line.bbox[3] - line.bbox[1] for line in group) * 1.3)
    mask = make_text_mask(image, group, line_padding)
    likely = likely_white_mask(image, block)
    covered = int(np.logical_and(likely > 0, mask > 0).sum())
    likely_count = int((likely > 0).sum())
    coverage = covered / likely_count if likely_count else 0.0
    complete = bool(block and score >= score_threshold and strong_groups >= 2 and stacked and coverage >= minimum_coverage)
    status = "complete" if complete else "partial" if group and strong_groups >= 1 else "uncertain"
    rejection = []
    if strong_groups < 2: rejection.append("fewer_than_two_strong_pattern_groups")
    if not stacked: rejection.append("not_enough_stacked_lines")
    if coverage < minimum_coverage: rejection.append("mask_coverage_below_threshold")
    return {"status": status, "score": round(score, 3), "bbox": block, "lines": group, "pattern_groups": reasons, "coverage": coverage, "likely_pixels": likely_count, "covered_pixels": covered, "rejection_reasons": rejection, "corner": corner_name(block, corner_box)}


def corner_name(block: tuple[float, float, float, float] | None, corner_box: tuple[int, int, int, int]) -> str:
    return "unknown" if not block else "corner-zone"


def likely_white_mask(image: Image.Image, block: tuple[float, float, float, float] | None) -> np.ndarray:
    import cv2

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    if not block:
        return mask
    x1, y1, x2, y2 = [max(0, int(value)) for value in block]
    crop = np.asarray(image.crop((x1, y1, min(image.width, x2), min(image.height, y2))).convert("RGB"))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    local = cv2.inRange(crop, np.array([145, 145, 145], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    local = cv2.bitwise_and(local, cv2.inRange(hsv, np.array([0, 0, 110], dtype=np.uint8), np.array([180, 115, 255], dtype=np.uint8)))
    mask[y1:y1 + local.shape[0], x1:x1 + local.shape[1]] = local
    return mask


def make_text_mask(image: Image.Image, lines: list[OCRLine], padding: float) -> np.ndarray:
    import cv2

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = expanded_box(line.bbox, padding, image.width, image.height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def _paddle_or_tesseract(backend: str) -> Any:
    if backend in {"paddle", "auto"}:
        try:
            return PaddleAdapter(), "paddle", []
        except Exception as exc:
            if backend == "paddle":
                raise
            fallback = [f"paddle unavailable: {exc}"]
    else:
        fallback = []
    return None, "tesseract", fallback


def _ocr_variant(adapter: Any, backend: str, image: Image.Image) -> list[OCRLine]:
    return adapter.lines(image) if backend == "paddle" else _tesseract_lines(image)


def _draw_lines(image: Image.Image, lines: list[OCRLine], block: dict[str, Any] | None = None) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for line in lines:
        draw.rectangle(line.bbox, outline="yellow", width=2)
        draw.text((line.bbox[0], max(0, line.bbox[1] - 18)), f"{line.text} [{line.score:.1f}]", fill="yellow")
    if block and block.get("bbox"):
        draw.rectangle(block["bbox"], outline="lime", width=5)
        draw.text((block["bbox"][0], max(0, block["bbox"][1] - 24)), f"{block['status']} {block['score']:.1f} cov={block['coverage']:.2f}", fill="lime")
    return output


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        path = row["preview_path"]
        cards.append(f"<figure><img src='{html.escape(path)}'><figcaption>{html.escape(row['filename'])}<br>{html.escape(row['overlay_status'])} | score {row['overlay_score']} | coverage {row['estimated_mask_coverage']}</figcaption></figure>")
    content = "<!doctype html><meta charset='utf-8'><title>Run 2.1 overlay review</title><style>body{font-family:sans-serif;background:#222;color:#eee}main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;background:#000}figcaption{padding:4px}</style><main>" + "".join(cards) + "</main>"
    (run_dir / "overlay_contact_sheet.html").write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", default="experiments/expiry_detector_eval/outputs")
    parser.add_argument("--run-name", default="run_2_1_paddle_multivariant")
    parser.add_argument("--overlay-ocr-backend", choices=["paddle", "tesseract", "auto"], default="paddle")
    parser.add_argument("--corner-width-fraction", type=float, default=0.55)
    parser.add_argument("--corner-height-fraction", type=float, default=0.45)
    parser.add_argument("--corner-overlap", type=float, default=0.15)
    parser.add_argument("--ocr-grouping-distance", type=float, default=140.0)
    parser.add_argument("--overlay-score-threshold", type=float, default=6.0)
    parser.add_argument("--minimum-mask-coverage", type=float, default=0.70)
    parser.add_argument("--line-mask-padding", type=float, default=12.0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--save-variant-diagnostics", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    input_folder = Path(args.input_folder).expanduser().resolve()
    images = discover_images(input_folder)[:args.max_images if args.max_images else None]
    if not images:
        raise ValueError(f"No supported images found in {input_folder}")
    run_dir = Path(args.output_folder).expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter, backend, warnings = _paddle_or_tesseract(args.overlay_ocr_backend)
    overlay_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    status_counts = Counter()
    variant_counts = Counter()
    for image_path in images:
        relative = image_path.relative_to(input_folder)
        image = _open_image(image_path)
        corners = _corner_boxes(image.width, image.height, args.corner_width_fraction, args.corner_height_fraction, args.corner_overlap)
        all_lines: list[OCRLine] = []
        contributing_variants = set()
        started = time.perf_counter()
        corner_scores = []
        for corner, crop_box in corners.items():
            crop = image.crop(crop_box)
            corner_dir = run_dir / "corner_zones" / relative.parent / relative.stem / corner
            corner_dir.mkdir(parents=True, exist_ok=True)
            crop.save(corner_dir / "corner_crop.jpg")
            corner_lines: list[OCRLine] = []
            for variant_name, variant, scale in _variants(crop):
                lines = _map_lines(_ocr_variant(adapter, backend, variant), crop_box, scale)
                for line in lines:
                    line.score, line.reasons = score_overlay_line(line.text)
                corner_lines.extend(lines)
                variant_counts[variant_name] += 1
                if args.save_variant_diagnostics:
                    variant.save(corner_dir / f"{variant_name}.png")
                    _draw_lines(variant, lines).save(corner_dir / f"{variant_name}_ocr.png")
                for line in lines:
                    variant_rows.append({"filename": str(relative), "corner": corner, "variant": variant_name, "ocr_line_count": len(lines), "recognized_text": line.text, "ocr_confidence": line.confidence, "original_coordinates": line.bbox, "pattern_groups": ";".join(line.reasons)})
            corner_lines = [line for line in _dedupe_lines(corner_lines) if line.score > 0]
            groups = _candidate_groups(corner_lines, args.ocr_grouping_distance)
            candidates = [_classify_group(group, crop_box, args.overlay_score_threshold, args.minimum_mask_coverage, image, args.line_mask_padding) for group in groups]
            if candidates:
                best = max(candidates, key=lambda item: item["score"])
                best["corner"] = corner
                corner_scores.append(best)
                if best["status"] in {"complete", "partial"}:
                    all_lines.extend(best["lines"])
                    # The selected candidate is an ensemble across this corner's variants.
                    contributing_variants.update(variant_name for variant_name, _, _ in _variants(crop))
        all_lines = _dedupe_lines(all_lines)
        chosen = max(corner_scores, key=lambda item: item["score"]) if corner_scores else {"status": "not_found", "score": 0.0, "bbox": None, "lines": [], "pattern_groups": [], "coverage": 0.0, "likely_pixels": 0, "covered_pixels": 0, "rejection_reasons": ["no_pattern_matched"], "corner": "unknown"}
        # If the best evidence is not complete, the primary masked image remains unchanged.
        final_mask = make_text_mask(image, chosen["lines"], args.line_mask_padding) if chosen["status"] in {"complete", "partial"} else np.zeros((image.height, image.width), dtype=np.uint8)
        mask_path = run_dir / "masks" / relative.parent / f"{relative.stem}_final_mask.png"
        masked_path = run_dir / "masked_images" / relative.parent / f"{relative.stem}.jpg"
        preview_path = run_dir / "overlay_previews" / relative.parent / f"{relative.stem}_overlay_review.jpg"
        for path in (mask_path, masked_path, preview_path): path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final_mask).save(mask_path)
        if chosen["status"] == "complete":
            import cv2
            masked_array = cv2.inpaint(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), final_mask, 3, cv2.INPAINT_TELEA)
            Image.fromarray(cv2.cvtColor(masked_array, cv2.COLOR_BGR2RGB)).save(masked_path)
        else:
            image.save(masked_path)
        _draw_lines(image, all_lines, chosen).save(preview_path)
        elapsed = time.perf_counter() - started
        status_counts[chosen["status"]] += 1
        overlay_rows.append({"filename": str(relative), "selected_corner": chosen.get("corner", "unknown"), "overlay_status": chosen["status"], "overlay_score": chosen["score"], "recognized_overlay_text": " ".join(line.text for line in chosen["lines"]), "matched_pattern_groups": ";".join(chosen.get("pattern_groups", [])), "ocr_variant_count": len(contributing_variants), "overlay_line_count": len(chosen["lines"]), "overlay_x1": chosen["bbox"][0] if chosen.get("bbox") else "", "overlay_y1": chosen["bbox"][1] if chosen.get("bbox") else "", "overlay_x2": chosen["bbox"][2] if chosen.get("bbox") else "", "overlay_y2": chosen["bbox"][3] if chosen.get("bbox") else "", "likely_overlay_text_pixels": chosen.get("likely_pixels", 0), "covered_overlay_text_pixels": chosen.get("covered_pixels", 0), "estimated_mask_coverage": round(chosen.get("coverage", 0.0), 4), "ocr_processing_time": round(elapsed, 6), "mask_path": str(mask_path.relative_to(run_dir)), "preview_path": str(preview_path.relative_to(run_dir)), "masked_image_path": str(masked_path.relative_to(run_dir)), "rejection_reasons": ";".join(chosen.get("rejection_reasons", []))})
    _write_csv(run_dir / "overlay_detection.csv", OVERLAY_FIELDS, overlay_rows)
    _write_csv(run_dir / "variant_detections.csv", VARIANT_FIELDS, variant_rows)
    _write_csv(run_dir / "manual_overlay_review.csv", OVERLAY_REVIEW_FIELDS, [{"filename": row["filename"]} for row in overlay_rows])
    _contact_sheet(run_dir, overlay_rows)
    (run_dir / "summary.json").write_text(json.dumps({"run_name": args.run_name, "input_folder": str(input_folder), "total_images": len(images), "overlay_status_counts": dict(status_counts), "ocr_backend_used": backend, "backend_warnings": warnings, "variant_contribution_counts": dict(variant_counts), "acceptance_gate": "Run YOLO suppression only after manual_overlay_review.csv confirms acceptable_for_yolo_comparison=yes for the intended images", "settings": vars(args)}, indent=2, default=str), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        print(f"Run 2.1 complete: {run(build_parser().parse_args(argv))}")
        return 0
    except Exception as exc:
        print(f"Run 2.1 could not start: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

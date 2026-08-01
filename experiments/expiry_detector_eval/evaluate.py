"""Standalone evaluation harness for the pretrained expiration-date detector."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

MODEL_REPO = "krishuggingface/Expiry_Date_Detection"
MODEL_FILENAME = "best_model.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
DETECTION_FIELDS = [
    "filename", "image_width", "image_height", "detection_index", "class_id",
    "class_name", "confidence", "x1", "y1", "x2", "y2", "padded_x1",
    "padded_y1", "padded_x2", "padded_y2", "inference_time", "annotated_image_path",
    "crop_path", "model_name", "inference_image_size", "confidence_threshold", "device_used",
]
SUMMARY_FIELDS = ["filename", "detection_count", "inference_time", "device", "status"]
REVIEW_FIELDS = [
    "filename", "product_date_visible", "correct_region_found", "full_date_inside_crop",
    "false_positive_present", "number_of_real_date_regions", "notes",
]


class DeviceUnavailableError(ValueError):
    """Raised when a requested accelerator is not available."""


def _get_torch():
    import torch

    return torch


def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/explicit device selection and fail cleanly for unavailable devices."""
    requested = requested.strip().lower()
    torch = _get_torch()
    mps_available = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "mps" and not mps_available:
        raise DeviceUnavailableError("MPS was requested, but torch.backends.mps.is_available() is false.")
    if requested.startswith("cuda") and not cuda_available:
        raise DeviceUnavailableError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested.startswith("cuda") and not re.fullmatch(r"cuda(?::\d+)?", requested):
        raise DeviceUnavailableError(f"Unsupported CUDA device '{requested}'. Use cuda or cuda:<non-negative-index>.")
    if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
        raise DeviceUnavailableError(f"Unsupported device '{requested}'. Use auto, cpu, mps, or cuda[:index].")
    return requested


def discover_images(input_folder: Path) -> list[Path]:
    """Return supported images recursively, sorted for reproducible reruns."""
    def natural_key(path: Path) -> list[Any]:
        return [(0, int(part)) if part.isdigit() else (1, part.casefold()) for part in re.split(r"(\d+)", str(path.relative_to(input_folder)))]

    return sorted((p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)


def padded_box(box: Iterable[float], padding_percent: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Add percentage padding relative to box width/height, clipping to image bounds."""
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = max(0.0, x2 - x1) * padding_percent / 100.0
    pad_y = max(0.0, y2 - y1) * padding_percent / 100.0
    return (
        max(0, min(width, int(x1 - pad_x))),
        max(0, min(height, int(y1 - pad_y))),
        max(0, min(width, int(x2 + pad_x))),
        max(0, min(height, int(y2 + pad_y))),
    )


def build_output_paths(run_dir: Path, relative_filename: Path, detection_index: int) -> tuple[Path, Path, Path]:
    """Generate deterministic annotated/crop paths for one image and detection."""
    stem = relative_filename.stem
    parent = relative_filename.parent
    annotated_suffix = ".png" if relative_filename.suffix.lower() in {".heic", ".heif"} else relative_filename.suffix.lower()
    annotated = run_dir / "annotated" / parent / f"{stem}{annotated_suffix}"
    crop = run_dir / "crops" / parent / f"{stem}_detection_{detection_index:03d}.png"
    return annotated, crop, run_dir / "crops" / parent


def _open_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError as exc:
            raise RuntimeError("HEIC support requires the optional 'pillow-heif' package.") from exc
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _to_list(values: Any) -> list[Any]:
    return values.detach().cpu().tolist() if hasattr(values, "detach") else values.tolist() if hasattr(values, "tolist") else list(values)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def evaluate_image(model: Any, image_path: Path, input_folder: Path, run_dir: Path, conf: float, imgsz: int, padding_percent: float, device: str, model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image = _open_image(image_path)
    width, height = image.size
    started = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
    inference_time = time.perf_counter() - started
    result = results[0]
    annotated_path, _, _ = build_output_paths(run_dir, image_path.relative_to(input_folder), 1)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()
    if hasattr(plotted, "shape"):
        import numpy as np

        Image.fromarray(np.asarray(plotted)[:, :, ::-1]).save(annotated_path)
    else:
        image.save(annotated_path)

    boxes = getattr(result, "boxes", None)
    xyxy = _to_list(boxes.xyxy) if boxes is not None else []
    confidences = _to_list(boxes.conf) if boxes is not None else []
    classes = _to_list(boxes.cls) if boxes is not None else []
    names = getattr(result, "names", getattr(model, "names", {}))
    rows = []
    for index, (box, confidence, class_value) in enumerate(zip(xyxy, confidences, classes), start=1):
        class_id = int(class_value)
        padded = padded_box(box, padding_percent, width, height)
        _, crop_path, _ = build_output_paths(run_dir, image_path.relative_to(input_folder), index)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(padded).save(crop_path)
        rows.append({
            "filename": str(image_path.relative_to(input_folder)), "image_width": width, "image_height": height,
            "detection_index": index, "class_id": class_id, "class_name": _class_name(names, class_id),
            "confidence": round(float(confidence), 6), "x1": round(float(box[0]), 3), "y1": round(float(box[1]), 3),
            "x2": round(float(box[2]), 3), "y2": round(float(box[3]), 3), "padded_x1": padded[0], "padded_y1": padded[1],
            "padded_x2": padded[2], "padded_y2": padded[3], "inference_time": round(inference_time, 6),
            "annotated_image_path": str(annotated_path.relative_to(run_dir)), "crop_path": str(crop_path.relative_to(run_dir)),
            "model_name": model_name, "inference_image_size": imgsz, "confidence_threshold": conf, "device_used": device,
        })
    status = "DETECTED" if rows else "NO_DETECTIONS"
    return rows, {"filename": str(image_path.relative_to(input_folder)), "detection_count": len(rows), "inference_time": round(inference_time, 6), "device": device, "status": status}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _confidence_distribution(confidences: list[float]) -> dict[str, Any]:
    bins = Counter(f"{min(int(value * 10), 9) / 10:.1f}-{min(int(value * 10), 9) / 10 + 0.1:.1f}" for value in confidences)
    return {"count": len(confidences), "min": min(confidences) if confidences else None, "max": max(confidences) if confidences else None, "mean": statistics.mean(confidences) if confidences else None, "histogram": dict(sorted(bins.items()))}


def _software_versions() -> dict[str, str | None]:
    """Collect reproducibility details without importing optional experiment tools early."""
    import PIL
    import torch

    try:
        import ultralytics
        ultralytics_version: str | None = str(ultralytics.__version__)
    except ImportError:
        ultralytics_version = None
    return {"python": platform.python_version(), "pytorch": str(torch.__version__), "pytorch_cuda": torch.version.cuda, "pillow": str(PIL.__version__), "ultralytics": ultralytics_version}


def _resolved_config(args: argparse.Namespace, input_folder: Path, output_base: Path, run_dir: Path, device: str) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_identifier": MODEL_REPO,
        "model_filename": MODEL_FILENAME,
        "input_directory": str(input_folder),
        "output_directory": str(run_dir),
        "output_root": str(output_base),
        "requested_device": args.device,
        "selected_device": device,
        "confidence_threshold": args.confidence,
        "inference_image_size": args.imgsz,
        "crop_padding_percent": args.padding,
        "maximum_image_count": args.max_images,
        "supported_image_formats": sorted(IMAGE_SUFFIXES),
        "software_versions": _software_versions(),
    }


def _default_run_name() -> str:
    """Return a high-resolution, clean-image-friendly name that avoids accidental reuse."""
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"


def environment_health_check(requested_device: str = "auto") -> dict[str, Any]:
    """Resolve the device and verify the configured checkpoint can be constructed."""
    device = resolve_device(requested_device)
    torch = _get_torch()
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none detected"
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    YOLO(checkpoint)
    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "pytorch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "ultralytics_version": _software_versions()["ultralytics"],
        "model_identifier": MODEL_REPO,
        "model_checkpoint_loads": True,
        "requested_device": requested_device,
        "selected_inference_device": device,
    }


def evaluate(args: argparse.Namespace) -> Path:
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_base = Path(args.output_folder).expanduser().resolve()
    if not input_folder.is_dir():
        raise ValueError(f"Input image folder does not exist: {input_folder}")
    if args.max_images is not None and args.max_images < 0:
        raise ValueError("--max-images must be zero or a positive integer.")
    device = resolve_device(args.device)
    images = discover_images(input_folder)[:args.max_images if args.max_images is not None else None]
    if not images:
        raise ValueError(f"No supported images found in {input_folder}")
    run_name = args.run_name or _default_run_name()
    run_dir = output_base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = _resolved_config(args, input_folder, output_base, run_dir, device)
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = YOLO(checkpoint)
    detection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    image_times: list[float] = []
    all_confidences: list[float] = []
    total_started = time.perf_counter()
    for image_path in images:
        try:
            rows, summary_row = evaluate_image(model, image_path, input_folder, run_dir, args.confidence, args.imgsz, args.padding, device, MODEL_REPO)
            detection_rows.extend(rows)
            summary_rows.append(summary_row)
            image_times.append(summary_row["inference_time"])
            all_confidences.extend(float(row["confidence"]) for row in rows)
        except Exception as exc:  # Keep one bad image from stopping a folder evaluation.
            summary_rows.append({"filename": str(image_path.relative_to(input_folder)), "detection_count": 0, "inference_time": "", "device": device, "status": f"ERROR: {exc}"})
    _write_csv(run_dir / "detections.csv", DETECTION_FIELDS, detection_rows)
    _write_csv(run_dir / "image_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(run_dir / "manual_review.csv", REVIEW_FIELDS, [{"filename": row["filename"]} for row in summary_rows])
    processed = len(summary_rows)
    report = {"run_name": run_dir.name, "timestamp": resolved_config["timestamp"], "total_images_processed": processed, "images_with_detections": sum(row["status"] == "DETECTED" for row in summary_rows), "images_without_detections": sum(row["status"] == "NO_DETECTIONS" for row in summary_rows), "total_detections": len(detection_rows), "average_detections_per_image": len(detection_rows) / processed if processed else 0, "average_inference_time": statistics.mean(image_times) if image_times else None, "median_inference_time": statistics.median(image_times) if image_times else None, "total_processing_time": time.perf_counter() - total_started, "confidence_distribution": _confidence_distribution(all_confidences), "requested_device": args.device, "device_used": device, "model_name": MODEL_REPO, "inference_image_size": args.imgsz, "confidence_threshold": args.confidence, "padding_percent": args.padding, "maximum_image_count": args.max_images, "software_versions": resolved_config["software_versions"]}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder")
    parser.add_argument("--output-folder", default="experiments/expiry_detector_eval/outputs")
    parser.add_argument("--run-name", help="Optional unique name; defaults to a timestamped run directory.")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--padding", type=float, default=25.0, help="Crop padding as a percentage of the predicted box size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda[:index]")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--health-check", action="store_true", help="Print environment and model-load status without processing images.")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.health_check:
            print(json.dumps(environment_health_check(args.device), indent=2))
            return 0
        if not args.input_folder:
            raise ValueError("--input-folder is required unless --health-check is used.")
        run_dir = evaluate(args)
    except (DeviceUnavailableError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Evaluation could not start: {exc}")
        return 2
    print(f"Evaluation complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

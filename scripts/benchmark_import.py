#!/usr/bin/env python3
"""Compare the old full-analysis folder pass with the staged fast pass.

This script reads a local folder only. It does not copy images into uploads or
commit them, and it intentionally leaves stage-cache entries so a second run
shows warm-cache behavior.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

# Allow direct execution from the repository's scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.barcode.decoder import decode_image
from src.importing.fast_pass import run_fast_pass
from src.importing.grouping import suggest_groups
from src.ocr.paddle_engine import run_ocr
from src.processing import build_draft
from src.storage.database import init_db


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def load_images(folder: Path, limit: int | None) -> list[dict]:
    paths = sorted((path for path in folder.rglob("*") if path.suffix.lower() in SUPPORTED), key=lambda path: str(path).casefold())
    if limit:
        paths = paths[:limit]
    images = []
    for order, path in enumerate(paths, 1):
        data = path.read_bytes()
        with Image.open(path) as image:
            width, height = image.size
        images.append({
            "order": order,
            "path": str(path),
            "original_name": path.name,
            "mime": "image/*",
            "size": len(data),
            "content_hash": hashlib.sha256(data).hexdigest(),
            "width": width,
            "height": height,
        })
    return images


def run_legacy(images: list[dict]) -> dict:
    started = time.perf_counter()
    warnings = []
    for image in images:
        image["ocr"] = run_ocr(image["path"], image["order"])
        image["barcodes"], barcode_warnings = decode_image(image["path"], image["order"])
        warnings.extend(image["ocr"].get("warnings", []) + barcode_warnings)
    build_draft(images, warnings, perform_lookup=False)
    groups = suggest_groups(images)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"mode": "legacy", "images": len(images), "groups": len(groups), "elapsed_ms": round(elapsed_ms, 2), "warnings": len(warnings)}


def run_optimized(images: list[dict]) -> dict:
    started = time.perf_counter()
    cache_hits = 0
    warnings = []
    for image in images:
        result = run_fast_pass(image)
        cache_hits += int(result.get("cache_hit", False))
        warnings.extend(result.get("barcode_warnings", []))
    groups = suggest_groups(images)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"mode": "optimized_fast_pass", "images": len(images), "groups": len(groups), "elapsed_ms": round(elapsed_ms, 2), "cache_hits": cache_hits, "warnings": len(warnings)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mode", choices=("legacy", "optimized", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Only benchmark the first N filename-sorted images")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"Not a folder: {args.folder}")
    init_db()
    images = load_images(args.folder, args.limit)
    if not images:
        raise SystemExit("No supported images found")
    results = []
    if args.mode in {"legacy", "both"}:
        results.append(run_legacy([dict(image) for image in images]))
    if args.mode in {"optimized", "both"}:
        results.append(run_optimized([dict(image) for image in images]))
    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"{result['mode']}: {result['elapsed_ms'] / 1000:.2f}s for {result['images']} images, {result['groups']} groups, {result.get('cache_hits', 0)} fast-cache hits")


if __name__ == "__main__":
    main()

"""Shared per-image and observation analysis used by single and folder imports."""
import json
import time
from src.barcode.decoder import decode_image
from src.config import BARCODE_CONFIG_VERSION, OCR_CONFIG_VERSION, OCR_MAX_INFERENCE_SIDE
from src.ocr.paddle_engine import run_ocr
from src.storage.database import get_stage_cache, put_stage_cache, upsert_import_image
from src.models import Candidate
from src.extraction.labels import classify_labels
from src.extraction.dates import parse_dates
from src.extraction.product_text import extract_product_text
from src.extraction.merge import merge_candidates
from src.product_lookup.open_food_facts import lookup

def analyze_image_deferred(image: dict) -> dict:
    """Run or restore full OCR and barcode analysis for one confirmed image."""
    content_hash = image["content_hash"]
    ocr_key = f"{OCR_CONFIG_VERSION}:side={OCR_MAX_INFERENCE_SIDE}"
    ocr_cached = get_stage_cache(content_hash, "full_ocr", ocr_key)
    if ocr_cached:
        image["ocr"] = json.loads(ocr_cached["result_json"])
        image["ocr_elapsed_ms"] = ocr_cached["elapsed_ms"]
    else:
        started = time.perf_counter(); image["ocr"] = run_ocr(image["path"], image["order"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        image["ocr_elapsed_ms"] = elapsed_ms
        put_stage_cache(content_hash, "full_ocr", ocr_key, "COMPLETED" if image["ocr"].get("text") else "NO_DETECTION", image["ocr"], elapsed_ms, image["ocr"].get("warnings", []))
    barcode_key = BARCODE_CONFIG_VERSION
    barcode_cached = get_stage_cache(content_hash, "full_barcode", barcode_key)
    if barcode_cached:
        cached = json.loads(barcode_cached["result_json"]); image["barcodes"] = cached.get("barcodes", []); image["barcode_warnings"] = cached.get("warnings", [])
        image["barcode_elapsed_ms"] = barcode_cached["elapsed_ms"]
    else:
        started = time.perf_counter(); image["barcodes"], image["barcode_warnings"] = decode_image(image["path"], image["order"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        image["barcode_elapsed_ms"] = elapsed_ms
        put_stage_cache(content_hash, "full_barcode", barcode_key, "COMPLETED" if image["barcodes"] else "NO_DETECTION", {"barcodes": image["barcodes"], "warnings": image["barcode_warnings"]}, elapsed_ms, image["barcode_warnings"])
    image["warnings"] = image.get("ocr", {}).get("warnings", []) + image.get("barcode_warnings", [])
    if image.get("import_id"):
        deferred_status = "COMPLETED" if image["ocr"].get("text") or image["barcodes"] else "NO_DETECTION"
        deferred_json = {"ocr": image["ocr"], "barcodes": image["barcodes"]}
        deferred_elapsed_ms = image.get("ocr_elapsed_ms", 0) + image.get("barcode_elapsed_ms", 0)
        upsert_import_image(image["import_id"], image, "COMPLETED" if image.get("barcodes") else "NO_DETECTION", image.get("fast_json"), image.get("fast_elapsed_ms"), deferred_status, deferred_json, deferred_elapsed_ms)
    return image

def analyze_images_deferred(images: list[dict]) -> list[dict]:
    return [analyze_image_deferred(image) for image in images]

def build_draft(images: list[dict], warnings: list[str] | None = None, perform_lookup: bool = True) -> dict:
    warnings = warnings or []
    all_text = "\n".join(f"[IMAGE {image['order']}]\n{image['ocr'].get('text', '')}" for image in images)
    label_candidates: list[Candidate] = []; date_candidates: list[Candidate] = []; barcodes = []
    product_candidates = {"product_name": [], "brand": [], "category": []}
    for image in images:
        text = image["ocr"].get("text", "")
        labels = classify_labels(text, image["order"]); dates = parse_dates(text, image["order"])
        product = extract_product_text(text, image["order"])
        image["evidence_types"] = {"BARCODE": any(x.get("supported") for x in image.get("barcodes", [])), "DATE_LABEL": bool(labels or dates), "PRODUCT_IDENTITY": bool(product["product_name"] or product["brand"])}
        label_candidates += labels; date_candidates += dates
        for key in product_candidates: product_candidates[key] += product[key]
        barcodes += [item for item in image.get("barcodes", []) if item.get("supported")]
    barcode = merge_candidates([Candidate(item["value"], "BARCODE", item["image_order"], item.get("confidence"), json.dumps(item.get("raw", {}))) for item in barcodes])
    lookup_result = lookup(str(barcode.get("value", ""))) if perform_lookup and barcode.get("status") == "FOUND" else {"status": "NOT_ATTEMPTED"}
    return {"images": images, "warnings": warnings, "ocr_text": all_text, "barcode": barcode, "lookup": lookup_result, "label": merge_candidates(label_candidates), "dates": merge_candidates(date_candidates), "product": {key: merge_candidates(value) for key, value in product_candidates.items()}}

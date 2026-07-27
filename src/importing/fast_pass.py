"""Fast, resumable folder pass: hashing, reduced barcode decoding, and safe anchors only."""
import json
import time
from src.barcode.decoder import decode_fast
from src.config import BARCODE_CONFIG_VERSION, FAST_PASS_CONFIG_VERSION, OCR_TARGETED_MAX_INFERENCE_SIDE
from src.storage.database import get_stage_cache, put_stage_cache, upsert_import_image

def run_fast_pass(image: dict) -> dict:
    content_hash = image["content_hash"]
    fast_config_key = f"{FAST_PASS_CONFIG_VERSION}:side={OCR_TARGETED_MAX_INFERENCE_SIDE}"
    cached = get_stage_cache(content_hash, "fast_grouping", fast_config_key)
    if cached:
        result = json.loads(cached["result_json"])
        result["cache_hit"] = True
        image.update(result)
        image["fast_json"] = result
        image["fast_elapsed_ms"] = cached["elapsed_ms"]
        if image.get("import_id"):
            upsert_import_image(image["import_id"], image, cached["status"], result, cached["elapsed_ms"])
        return result
    started = time.perf_counter()
    barcodes, warnings, barcode_ms = decode_fast(image["path"], image["order"])
    result = {"barcodes": barcodes, "barcode_warnings": warnings, "barcode_elapsed_ms": barcode_ms, "evidence_types": {"BARCODE": any(item.get("supported") for item in barcodes), "DATE_LABEL": None, "PRODUCT_IDENTITY": None}, "stage": "fast_grouping", "config": {"fast_pass": FAST_PASS_CONFIG_VERSION, "barcode": BARCODE_CONFIG_VERSION}}
    elapsed_ms = (time.perf_counter() - started) * 1000
    put_stage_cache(content_hash, "fast_grouping", fast_config_key, "COMPLETED" if barcodes else "NO_DETECTION", result, elapsed_ms, warnings)
    image.update(result)
    image["fast_json"] = result
    image["fast_elapsed_ms"] = elapsed_ms
    if image.get("import_id"):
        upsert_import_image(image["import_id"], image, "COMPLETED" if barcodes else "NO_DETECTION", result, elapsed_ms)
    return result

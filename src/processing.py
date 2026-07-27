"""Shared per-image and observation analysis used by single and folder imports."""
import json
from src.models import Candidate
from src.extraction.labels import classify_labels
from src.extraction.dates import parse_dates
from src.extraction.product_text import extract_product_text
from src.extraction.merge import merge_candidates
from src.product_lookup.open_food_facts import lookup

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

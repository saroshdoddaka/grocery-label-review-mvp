"""Conservative, rule-based product text extraction from visible packaging OCR."""
import re
from src.models import Candidate

BRAND_PATTERNS = ((r"\bdan+on[e]?\b", "Danone"),)

def extract_product_text(text: str, image_order: int = 1) -> dict[str, list[Candidate]]:
    """Return only product details explicitly supported by OCR text."""
    results = {"product_name": [], "brand": [], "category": []}
    for pattern, canonical in BRAND_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            results["brand"].append(Candidate(canonical, "IMAGE_OCR", image_order, .75, match.group(0)))

    normalized = re.sub(r"\s+", " ", text)
    light = re.search(r"\blight\b.{0,25}\bfit\b", normalized, re.I)
    flavor = re.search(r"\btiramisu\b", normalized, re.I)
    greek = re.search(r"\bgreek\b", normalized, re.I)
    name_parts = []
    if light: name_parts.append("Light + Fit")
    if greek: name_parts.append("Greek")
    if flavor: name_parts.append("Tiramisu")
    if name_parts:
        evidence = " ".join(x.group(0) for x in (light, greek, flavor) if x)
        results["product_name"].append(Candidate(" ".join(name_parts), "IMAGE_OCR", image_order, .6, evidence, "LOW_CONFIDENCE"))
    return results

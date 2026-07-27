import re
from src.models import Candidate

LABELS = [("Best If Used By", "BEST_IF_USED_BY"), ("Use or Freeze By", "USE_OR_FREEZE_BY"), ("Sell By", "SELL_BY"), ("Best By", "BEST_BY"), ("Use By", "USE_BY"), ("Freeze By", "FREEZE_BY"), ("Expires", "EXPIRATION"), ("Exp", "EXPIRATION")]

def classify_labels(text: str, image_order: int = 1) -> list[Candidate]:
    normalized = re.sub(r"[\s._-]+", " ", text).strip()
    out = []
    for phrase, category in LABELS:
        pattern = r"\b" + r"\s+".join(re.escape(x) for x in phrase.split()) + r"\b"
        match = re.search(pattern, normalized, re.I)
        if match:
            out.append(Candidate(match.group(0), "IMAGE_OCR", image_order, .85, text[max(0, match.start()-30):match.end()+45]))
    return out

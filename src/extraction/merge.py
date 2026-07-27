from collections import Counter
from src.models import Candidate

def merge_candidates(candidates: list[Candidate]) -> dict:
    if not candidates: return {"value": "", "status": "MISSING", "candidates": [], "explanation": "No evidence detected."}
    counts = Counter(str(c.value).lower() for c in candidates)
    winner, count = counts.most_common(1)[0]
    same = [c for c in candidates if str(c.value).lower() == winner]
    status = "FOUND" if len(counts) == 1 or count > 1 else "CONFLICTING"
    best = sorted(same, key=lambda c: (c.confidence or 0, -(c.image_order or 99)), reverse=True)[0]
    return {"value": best.value, "status": status, "candidates": candidates, "source": best.source, "image_order": best.image_order, "confidence": best.confidence, "evidence": best.evidence, "explanation": "Multiple candidates disagree." if status == "CONFLICTING" else ""}

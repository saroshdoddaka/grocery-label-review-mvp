"""Deterministic ordered grouping for the fixed grocery photo capture protocol."""
from dataclasses import dataclass, asdict

@dataclass
class SuggestedGroup:
    start: int
    end: int
    confidence: str
    reasons: list[str]

    @property
    def size(self) -> int: return self.end - self.start + 1

    def json(self) -> dict: return asdict(self)

def _complete(image: dict) -> bool:
    evidence = image["evidence_types"]
    return all(evidence.get(key, False) for key in ("BARCODE", "DATE_LABEL", "PRODUCT_IDENTITY"))

def suggest_groups(images: list[dict]) -> list[SuggestedGroup]:
    """Parse filename-ordered images; never silently join more than three."""
    groups: list[SuggestedGroup] = []; index = 0
    while index < len(images):
        first = images[index]; first_evidence = first["evidence_types"]
        reasons = []
        if not first_evidence.get("BARCODE"):
            reasons.append("Expected first-image barcode was not decoded.")
        if _complete(first):
            groups.append(SuggestedGroup(index, index, "HIGH" if first_evidence["BARCODE"] else "LOW", reasons or ["All required evidence is in image 1."]))
            index += 1; continue
        missing = [key for key, found in first_evidence.items() if not found]
        max_end = min(index + (2 if len(missing) >= 2 else 1), len(images) - 1)
        end = index
        for candidate_index in range(index + 1, max_end + 1):
            candidate = images[candidate_index]
            if candidate["evidence_types"].get("BARCODE") and candidate_index > index:
                reasons.append(f"Image {candidate_index + 1} has a barcode and starts the next product.")
                break
            end = candidate_index
        group_images = images[index:end + 1]
        covered = {kind: any(image["evidence_types"].get(kind, False) for image in group_images) for kind in ("BARCODE", "DATE_LABEL", "PRODUCT_IDENTITY")}
        unresolved = [kind for kind, found in covered.items() if not found]
        if unresolved: reasons.append("Missing evidence: " + ", ".join(unresolved) + ".")
        confidence = "HIGH" if not unresolved and first_evidence.get("BARCODE") else "LOW"
        groups.append(SuggestedGroup(index, end, confidence, reasons or ["Ordered protocol suggests this group."]))
        index = end + 1
    return groups

def groups_from_boundaries(image_count: int, starts: set[int], excluded: set[int]) -> list[SuggestedGroup]:
    """Build reviewer-controlled groups. Starts are zero-based image indices."""
    included = [index for index in range(image_count) if index not in excluded]
    if not included: return []
    boundaries = sorted({included[0], *[index for index in starts if index in included]})
    groups = []
    for position, start in enumerate(boundaries):
        next_start = boundaries[position + 1] if position + 1 < len(boundaries) else image_count
        members = [index for index in included if start <= index < next_start]
        for offset in range(0, len(members), 3):
            block = members[offset:offset + 3]
            groups.append(SuggestedGroup(block[0], block[-1], "REVIEWED", ["Reviewer-confirmed boundary."]))
    return groups

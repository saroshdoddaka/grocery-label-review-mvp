import re
from datetime import date
from src.models import Candidate

MONTHS = {m.lower(): i for i, m in enumerate("January February March April May June July August September October November December".split(), 1)}
MONTHS.update({m[:3].lower(): i for m, i in MONTHS.items()})
NUMERIC = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})(?:[/.\-](\d{2}|\d{4}))?\b")
WORD = re.compile(r"\b(?:(\w+)\s+(\d{1,2})(?:,?\s+(\d{4}))?| (\d{1,2})\s+(\w+)(?:\s+(\d{4}))?)\b", re.I | re.X)
MONTH_YEAR_DAY = re.compile(r"\b([A-Za-z]+)\s+(\d{4})\s+(\d{1,2})\b")

def _year(value: str | None):
    if not value: return None
    return 2000 + int(value) if len(value) == 2 and int(value) <= 68 else 1900 + int(value) if len(value) == 2 else int(value)

def _candidate(raw, month, day, year, image_order, evidence):
    try:
        if not 1 <= int(month) <= 12 or not 1 <= int(day) <= 31: return None
        if year: date(int(year), int(month), int(day))
    except ValueError: return None
    return Candidate(raw, "IMAGE_OCR", image_order, .8, evidence)

def date_components(raw: str) -> tuple[int | None, int | None, int | None]:
    """Return deterministic month/day/year components for supported printed formats."""
    match = MONTH_YEAR_DAY.fullmatch(raw.strip())
    if match and match.group(1).lower() in MONTHS:
        return MONTHS[match.group(1).lower()], int(match.group(3)), int(match.group(2))
    return None, None, None

def _is_camera_overlay(context: str) -> bool:
    """Camera overlays include a date followed immediately by a clock time."""
    return bool(re.search(r"\bat\s+\d{1,2}:\d{2}(?::\d{2})?\b", context, re.I))

def parse_dates(text: str, image_order: int = 1) -> list[Candidate]:
    out = []
    for match in MONTH_YEAR_DAY.finditer(text):
        context = text[match.start():match.end()+40]
        if _is_camera_overlay(context) or match.group(1).lower() not in MONTHS:
            continue
        candidate = _candidate(match.group(0), MONTHS[match.group(1).lower()], match.group(3), int(match.group(2)), image_order, text[max(0, match.start()-35):match.end()+35])
        if candidate: out.append(candidate)
    for match in NUMERIC.finditer(text):
        context = text[match.start():match.end()+40]
        if _is_camera_overlay(context):
            continue
        candidate = _candidate(match.group(0), match.group(1), match.group(2), _year(match.group(3)), image_order, text[max(0, match.start()-35):match.end()+35])
        if candidate: out.append(candidate)
    for match in WORD.finditer(text):
        context = text[match.start():match.end()+40]
        if _is_camera_overlay(context):
            continue
        mon, day, year, day2, mon2, year2 = match.groups(); mon = mon or mon2; day = day or day2; year = year or year2
        if mon.lower() in MONTHS:
            candidate = _candidate(match.group(0), MONTHS[mon.lower()], day, _year(year), image_order, text[max(0, match.start()-35):match.end()+35])
            if candidate: out.append(candidate)
    return out

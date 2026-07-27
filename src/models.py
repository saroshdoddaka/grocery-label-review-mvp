from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
import json

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class Candidate:
    value: Any
    source: str = "UNKNOWN"
    image_order: int | None = None
    confidence: float | None = None
    evidence: str = ""
    status: str = "FOUND"
    bbox: Any = None

    def json(self) -> str:
        return json.dumps(asdict(self), default=str)

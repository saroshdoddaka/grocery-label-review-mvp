from src.models import Candidate
from src.extraction.merge import merge_candidates
def test_conflict_is_retained():
    result=merge_candidates([Candidate("A", image_order=1), Candidate("B", image_order=2)])
    assert result["status"] == "CONFLICTING" and len(result["candidates"]) == 2

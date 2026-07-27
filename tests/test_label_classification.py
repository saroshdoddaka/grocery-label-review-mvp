from src.extraction.labels import classify_labels
def test_longest_phrase_wins():
    values=[x.value for x in classify_labels("BEST IF USED BY 01/12/27")]
    assert values == ["BEST IF USED BY"]
def test_exp_not_inside_word():
    assert classify_labels("expiration") == []

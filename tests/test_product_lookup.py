def test_empty_lookup_not_attempted():
    from src.product_lookup.open_food_facts import lookup
    assert lookup("")["status"] == "NOT_ATTEMPTED"

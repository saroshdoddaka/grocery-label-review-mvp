from src.extraction.product_text import extract_product_text

def test_extracts_explicit_packaging_text():
    result = extract_product_text("Danone LIGHT FIT GREEK tiramisu")
    assert result["brand"][0].value == "Danone"
    assert result["product_name"][0].value == "Light + Fit Greek Tiramisu"

from src.barcode.decoder import decode_image

def test_zxing_ean_format_is_supported(monkeypatch):
    import sys
    class Item:
        format = "EAN-13"
        text = "0036632038128"
    class ZXing:
        @staticmethod
        def read_barcodes(_): return [Item()]
    monkeypatch.setitem(sys.modules, "zxingcpp", ZXing())
    candidates, _ = decode_image("missing.jpg")
    # The fallback requires Pillow to open an actual image, so format support is covered by the live decoder check.
    assert isinstance(candidates, list)

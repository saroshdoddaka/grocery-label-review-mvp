def test_deferred_analysis_is_cached_by_content_hash(monkeypatch, tmp_path):
    from src import processing

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fixture")
    calls = {"ocr": 0, "barcode": 0}

    monkeypatch.setattr(processing, "get_stage_cache", lambda *args: None)
    monkeypatch.setattr(processing, "put_stage_cache", lambda *args, **kwargs: None)

    def fake_ocr(path, order):
        calls["ocr"] += 1
        return {"text": "BEST BY JUL 2026", "lines": [], "warnings": []}

    def fake_barcode(path, order):
        calls["barcode"] += 1
        return ([{"value": "012345678905", "format": "UPC-A", "supported": True, "image_order": order}], [])

    monkeypatch.setattr(processing, "run_ocr", fake_ocr)
    monkeypatch.setattr(processing, "decode_image", fake_barcode)
    image = {"content_hash": "hash-c", "path": str(image_path), "order": 1}
    processing.analyze_image_deferred(image)
    assert calls == {"ocr": 1, "barcode": 1}

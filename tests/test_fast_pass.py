def test_fast_pass_does_not_call_ocr(monkeypatch, tmp_path):
    from src.importing import fast_pass

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fixture")
    calls = []

    monkeypatch.setattr(fast_pass, "decode_fast", lambda path, order: ([{"value": "012345678905", "format": "UPC-A", "supported": True}], [], 1.0))
    monkeypatch.setattr(fast_pass, "get_stage_cache", lambda *args: None)
    monkeypatch.setattr(fast_pass, "put_stage_cache", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = fast_pass.run_fast_pass({"content_hash": "hash-b", "path": str(image_path), "order": 1})

    assert result["evidence_types"] == {"BARCODE": True, "DATE_LABEL": None, "PRODUCT_IDENTITY": None}
    assert result["barcodes"][0]["value"] == "012345678905"
    assert calls


def test_fast_cache_preserves_no_detection_status(monkeypatch, tmp_path):
    from src.importing import fast_pass

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fixture")
    stored = {"status": "NO_DETECTION", "result_json": '{"barcodes": [], "evidence_types": {"BARCODE": false, "DATE_LABEL": null, "PRODUCT_IDENTITY": null}}', "elapsed_ms": 2.5}
    writes = []
    monkeypatch.setattr(fast_pass, "get_stage_cache", lambda *args: stored)
    monkeypatch.setattr(fast_pass, "upsert_import_image", lambda *args: writes.append(args))

    fast_pass.run_fast_pass({"content_hash": "hash-d", "path": str(image_path), "order": 1, "import_id": "batch"})
    assert writes[0][2] == "NO_DETECTION"

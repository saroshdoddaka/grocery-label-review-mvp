import json


def test_stage_cache_reuses_no_detection_and_tracks_hits(tmp_path, monkeypatch):
    import src.storage.database as database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "cache.sqlite3")
    database.init_db()
    database.put_stage_cache("hash-a", "fast_grouping", "v1", "NO_DETECTION", {"barcodes": []}, 4.2, ["none"])

    first = database.get_stage_cache("hash-a", "fast_grouping", "v1")
    second = database.get_stage_cache("hash-a", "fast_grouping", "v1")
    assert json.loads(first["result_json"]) == {"barcodes": []}
    assert second["cache_hit"] is True

    assert database.get_stage_cache("hash-a", "fast_grouping", "v2") is None


def test_import_image_persistence_matches_schema(tmp_path, monkeypatch):
    import src.storage.database as database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "imports.sqlite3")
    database.init_db()
    database.create_import_batch("batch", "day-folder", 1, {"stage": "fast"})
    database.upsert_import_image("batch", {"order": 1, "content_hash": "hash-e", "path": "photo.jpg", "original_name": "photo.jpg", "size": 10, "width": 100, "height": 80}, "NO_DETECTION", {"barcodes": []}, 1.5)

    with database.conn() as db:
        row = db.execute("SELECT image_order, fast_status, deferred_status FROM import_images WHERE import_id=?", ("batch",)).fetchone()
    assert tuple(row) == (1, "NO_DETECTION", None)

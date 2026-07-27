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

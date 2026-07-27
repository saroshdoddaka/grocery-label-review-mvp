import json, sqlite3
from src.config import DB_PATH
from src.models import utc_now

def conn():
    connection = sqlite3.connect(DB_PATH); connection.row_factory = sqlite3.Row; return connection

def init_db():
    with conn() as db:
        db.executescript("""CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, status TEXT, reviewed_json TEXT, evidence_json TEXT);
        CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, observation_id TEXT, image_order INTEGER, path TEXT, original_name TEXT, mime TEXT, size INTEGER, uploaded_at TEXT, ocr_json TEXT, barcode_json TEXT, FOREIGN KEY(observation_id) REFERENCES observations(id));
        CREATE TABLE IF NOT EXISTS lookup_cache (barcode TEXT PRIMARY KEY, status TEXT, response_json TEXT, looked_up_at TEXT);""")
        db.execute("""CREATE TABLE IF NOT EXISTS folder_group_reviews (id INTEGER PRIMARY KEY, import_id TEXT, group_order INTEGER, suggested_json TEXT, confirmed_json TEXT, created_at TEXT)""")
        db.execute("""CREATE TABLE IF NOT EXISTS stage_cache (content_hash TEXT, stage TEXT, config_key TEXT, status TEXT, result_json TEXT, elapsed_ms REAL, warnings_json TEXT, cache_hits INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT, PRIMARY KEY(content_hash, stage, config_key))""")
        db.execute("""CREATE TABLE IF NOT EXISTS import_batches (id TEXT PRIMARY KEY, source_name TEXT, created_at TEXT, updated_at TEXT, status TEXT, config_json TEXT, total_images INTEGER, fast_pass_elapsed_ms REAL, grouping_ready_at TEXT, deferred_elapsed_ms REAL, total_elapsed_ms REAL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS import_images (import_id TEXT, image_order INTEGER, content_hash TEXT, path TEXT, original_name TEXT, size INTEGER, width INTEGER, height INTEGER, fast_status TEXT, fast_json TEXT, fast_elapsed_ms REAL, deferred_status TEXT, deferred_json TEXT, deferred_elapsed_ms REAL, PRIMARY KEY(import_id, image_order), FOREIGN KEY(import_id) REFERENCES import_batches(id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS import_groups (import_id TEXT, group_order INTEGER, suggested_start INTEGER, suggested_end INTEGER, confirmed_start INTEGER, confirmed_end INTEGER, status TEXT, reasons_json TEXT, created_at TEXT, PRIMARY KEY(import_id, group_order))""")

def get_stage_cache(content_hash: str, stage: str, config_key: str):
    with conn() as db:
        row = db.execute("SELECT * FROM stage_cache WHERE content_hash=? AND stage=? AND config_key=?", (content_hash, stage, config_key)).fetchone()
        if not row: return None
        db.execute("UPDATE stage_cache SET cache_hits=cache_hits+1, updated_at=? WHERE content_hash=? AND stage=? AND config_key=?", (utc_now(), content_hash, stage, config_key))
        result = dict(row); result["cache_hit"] = True; return result

def put_stage_cache(content_hash: str, stage: str, config_key: str, status: str, result, elapsed_ms: float, warnings=None):
    now = utc_now()
    with conn() as db:
        db.execute("INSERT OR REPLACE INTO stage_cache(content_hash,stage,config_key,status,result_json,elapsed_ms,warnings_json,cache_hits,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,COALESCE((SELECT created_at FROM stage_cache WHERE content_hash=? AND stage=? AND config_key=?),?),?)", (content_hash, stage, config_key, status, json.dumps(result, default=str), elapsed_ms, json.dumps(warnings or []), 0, content_hash, stage, config_key, now, now))

def create_import_batch(import_id: str, source_name: str, total_images: int, config: dict):
    now = utc_now()
    with conn() as db:
        db.execute("INSERT OR IGNORE INTO import_batches VALUES(?,?,?,?,?,?,?,?,?,?,?)", (import_id, source_name, now, now, "FAST_RUNNING", json.dumps(config), total_images, None, None, None, None))

def upsert_import_image(import_id: str, image: dict, fast_status=None, fast_json=None, fast_elapsed_ms=None, deferred_status=None, deferred_json=None, deferred_elapsed_ms=None):
    with conn() as db:
        db.execute("INSERT OR REPLACE INTO import_images VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (import_id, image["order"], image["content_hash"], image["path"], image["original_name"], image["size"], image.get("width"), image.get("height"), fast_status, json.dumps(fast_json, default=str) if fast_json is not None else None, fast_elapsed_ms, deferred_status, json.dumps(deferred_json, default=str) if deferred_json is not None else None, deferred_elapsed_ms))

def update_import_batch(import_id: str, status: str, **timings):
    fields = ["status=?", "updated_at=?"]; values = [status, utc_now()]
    for key, value in timings.items():
        if key in {"fast_pass_elapsed_ms", "grouping_ready_at", "deferred_elapsed_ms", "total_elapsed_ms"}:
            fields.append(f"{key}=?"); values.append(value)
    values.append(import_id)
    with conn() as db: db.execute(f"UPDATE import_batches SET {', '.join(fields)} WHERE id=?", values)

def save_import_groups(import_id: str, suggested, confirmed=None):
    with conn() as db:
        db.execute("DELETE FROM import_groups WHERE import_id=?", (import_id,))
        total = max(len(suggested), len(confirmed or []))
        for order in range(1, total + 1):
            suggested_group = suggested[order - 1] if order <= len(suggested) else None
            reviewed = confirmed[order - 1] if confirmed and order <= len(confirmed) else None
            db.execute("INSERT INTO import_groups VALUES(?,?,?,?,?,?,?,?,?)", (import_id, order, suggested_group.start if suggested_group else None, suggested_group.end if suggested_group else None, reviewed.start if reviewed else None, reviewed.end if reviewed else None, "CONFIRMED" if reviewed else "SUGGESTED", json.dumps((suggested_group or reviewed).reasons), utc_now()))

def save_group_reviews(import_id, suggested, confirmed):
    with conn() as db:
        db.execute("DELETE FROM folder_group_reviews WHERE import_id=?", (import_id,))
        for order, group in enumerate(confirmed, 1):
            suggested_group = suggested[order - 1].json() if order <= len(suggested) else None
            db.execute("INSERT INTO folder_group_reviews(import_id,group_order,suggested_json,confirmed_json,created_at) VALUES (?,?,?,?,?)", (import_id, order, json.dumps(suggested_group), json.dumps(group.json()), utc_now()))

def save_observation(obs_id, reviewed, evidence, images):
    now = utc_now()
    with conn() as db:
        db.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?)", (obs_id, now, now, "REVIEWED" if reviewed.get("reviewed") else "UNRESOLVED", json.dumps(reviewed), json.dumps(evidence, default=str)))
        for image in images:
            db.execute("INSERT INTO images(observation_id,image_order,path,original_name,mime,size,uploaded_at,ocr_json,barcode_json) VALUES (?,?,?,?,?,?,?,?,?)", (obs_id, image["order"], image["path"], image["original_name"], image["mime"], image["size"], now, json.dumps(image.get("ocr", {}), default=str), json.dumps(image.get("barcodes", []), default=str)))

def list_observations():
    with conn() as db: return [dict(row) for row in db.execute("SELECT * FROM observations ORDER BY created_at DESC")]

def get_observation(obs_id):
    with conn() as db:
        observation = db.execute("SELECT * FROM observations WHERE id=?", (obs_id,)).fetchone()
        images = db.execute("SELECT * FROM images WHERE observation_id=? ORDER BY image_order", (obs_id,)).fetchall()
    return None if not observation else {"observation": dict(observation), "images": [dict(row) for row in images]}

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

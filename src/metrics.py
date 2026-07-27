from src.storage.database import init_db, list_observations

def summary():
    init_db()
    rows = list_observations()
    return {"total": len(rows), "reviewed": sum(row["status"] == "REVIEWED" for row in rows), "unresolved": sum(row["status"] != "REVIEWED" for row in rows)}

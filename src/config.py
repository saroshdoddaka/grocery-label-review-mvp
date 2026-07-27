from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = ROOT / "grocery_labels.sqlite3"
USER_AGENT = os.getenv("OPEN_FOOD_FACTS_USER_AGENT", "GroceryLabelReviewMVP/1.0 (local development)")
UPLOAD_DIR.mkdir(exist_ok=True)

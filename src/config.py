from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = ROOT / "grocery_labels.sqlite3"
USER_AGENT = os.getenv("OPEN_FOOD_FACTS_USER_AGENT", "GroceryLabelReviewMVP/1.0 (local development)")
OCR_MAX_INFERENCE_SIDE = int(os.getenv("OCR_MAX_INFERENCE_SIDE", "1800"))
OCR_TARGETED_MAX_INFERENCE_SIDE = int(os.getenv("OCR_TARGETED_MAX_INFERENCE_SIDE", "1200"))
OCR_CONFIG_VERSION = "paddleocr-fast-v2"
FAST_PASS_CONFIG_VERSION = "fast-group-v1"
BARCODE_CONFIG_VERSION = "zxing-fast-v1"
UPLOAD_DIR.mkdir(exist_ok=True)

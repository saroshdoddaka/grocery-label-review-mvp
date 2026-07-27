from pathlib import Path
from uuid import uuid4
from io import BytesIO
from PIL import Image
from src.config import UPLOAD_DIR

def save_upload(uploaded, order: int) -> dict:
    data = uploaded.getvalue()
    try: Image.open(BytesIO(data)).verify()
    except Exception as exc: raise ValueError(f"Image {order} is unreadable: {exc}")
    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    path = UPLOAD_DIR / f"{uuid4().hex}_image_{order}{suffix}"
    path.write_bytes(data)
    return {"order": order, "path": str(path), "original_name": uploaded.name, "mime": uploaded.type or "image/*", "size": len(data)}

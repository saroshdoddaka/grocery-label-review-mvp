import json
import streamlit as st
from PIL import Image

MAX_INFERENCE_SIDE = 1800

@st.cache_resource(show_spinner=False)
def _engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_det_limit_side_len=MAX_INFERENCE_SIDE,
    )

def _inference_image(path: str):
    image = Image.open(path).convert("RGB")
    original_width, original_height = image.size
    scale = min(1.0, MAX_INFERENCE_SIDE / max(original_width, original_height))
    if scale < 1.0:
        image = image.resize((round(original_width * scale), round(original_height * scale)), Image.Resampling.LANCZOS)
    return image, original_width, original_height

def _scale_box(box, scale_x: float, scale_y: float):
    if box is None: return None
    try:
        values = box.tolist() if hasattr(box, "tolist") else box
        return [[round(point[0] * scale_x, 1), round(point[1] * scale_y, 1)] for point in values]
    except (TypeError, IndexError, ValueError):
        return box

def run_ocr(path: str, image_order: int = 1) -> dict:
    try:
        inference_image, original_width, original_height = _inference_image(path)
        inference_width, inference_height = inference_image.size
        import numpy as np
        result = list(_engine().predict(np.asarray(inference_image))); lines = []
        for page in result:
            data = getattr(page, "json", {})
            if callable(data):
                data = data()
            if isinstance(data, str):
                data = json.loads(data)
            data = data.get("res", data) if isinstance(data, dict) else {}
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_polys", data.get("rec_boxes", []))
            for index, text in enumerate(texts):
                box = boxes[index] if index < len(boxes) else None
                lines.append({"text": text, "confidence": float(scores[index]) if index < len(scores) else None, "bbox": _scale_box(box, original_width / inference_width, original_height / inference_height)})
        return {"text": "\n".join(x["text"] for x in lines), "lines": lines, "raw": result, "warnings": []}
    except ImportError:
        return {"text": "", "lines": [], "raw": None, "warnings": ["PaddleOCR is unavailable. Enter fields manually or install OCR dependencies."]}
    except Exception as exc:
        return {"text": "", "lines": [], "raw": None, "warnings": [f"OCR failed: {exc}"]}

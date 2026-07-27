from PIL import Image


def test_ocr_resize_preserves_aspect_ratio(tmp_path):
    from src.ocr import paddle_engine

    path = tmp_path / "wide.jpg"
    Image.new("RGB", (4000, 2000), "white").save(path)
    resized, width, height = paddle_engine._inference_image(str(path))
    assert (width, height) == (4000, 2000)
    assert resized.size == (1800, 900)

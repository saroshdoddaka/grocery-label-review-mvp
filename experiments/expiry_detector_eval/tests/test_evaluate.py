from pathlib import Path
from types import SimpleNamespace
import sys
import types

from PIL import Image

from experiments.expiry_detector_eval import evaluate as evaluation


def test_padding_calculations_and_coordinate_clipping():
    assert evaluation.padded_box((10, 20, 30, 40), 25, 100, 100) == (5, 15, 35, 45)
    assert evaluation.padded_box((-10, -5, 20, 30), 50, 100, 80) == (0, 0, 35, 47)
    assert evaluation.padded_box((80, 70, 110, 100), 50, 100, 80) == (65, 55, 100, 80)


def test_image_file_discovery_is_recursive_and_case_insensitive(tmp_path):
    (tmp_path / "nested").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "a.JPG")
    Image.new("RGB", (8, 8)).save(tmp_path / "nested" / "b.png")
    (tmp_path / "notes.txt").write_text("not an image")
    assert [path.name for path in evaluation.discover_images(tmp_path)] == ["a.JPG", "b.png"]


def test_image_discovery_uses_natural_filename_order(tmp_path):
    for name in ("photo_10.jpg", "photo_2.jpg", "photo_1.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    assert [path.name for path in evaluation.discover_images(tmp_path)] == ["photo_1.jpg", "photo_2.jpg", "photo_10.jpg"]


def test_output_path_generation(tmp_path):
    annotated, crop, parent = evaluation.build_output_paths(tmp_path, Path("nested/photo.jpg"), 2)
    assert annotated == tmp_path / "annotated/nested/photo.jpg"
    assert crop == tmp_path / "crops/nested/photo_detection_002.png"
    assert parent == tmp_path / "crops/nested"


def test_device_fallback_behavior(monkeypatch):
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(evaluation, "_get_torch", lambda: fake_torch)
    assert evaluation.resolve_device("auto") == "cpu"
    assert evaluation.resolve_device("cpu") == "cpu"
    try:
        evaluation.resolve_device("mps")
    except evaluation.DeviceUnavailableError:
        pass
    else:
        raise AssertionError("Unavailable MPS should fail gracefully")


def test_auto_prefers_cuda_and_explicit_cuda_does_not_fallback(monkeypatch):
    available = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(evaluation, "_get_torch", lambda: available)
    assert evaluation.resolve_device("auto") == "cuda"
    unavailable = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(evaluation, "_get_torch", lambda: unavailable)
    try:
        evaluation.resolve_device("cuda:0")
    except evaluation.DeviceUnavailableError as exc:
        assert "CUDA was requested" in str(exc)
    else:
        raise AssertionError("Unavailable CUDA should fail rather than use CPU")


class _Boxes:
    xyxy = []
    conf = []
    cls = []


class _Result:
    boxes = _Boxes()
    names = {0: "expiration_date"}

    def plot(self):
        import numpy as np

        return np.zeros((10, 12, 3), dtype=np.uint8)


class _NoDetectionModel:
    names = {0: "expiration_date"}

    def predict(self, **kwargs):
        return [_Result()]


def test_images_with_no_detections_are_written_to_summary(tmp_path, monkeypatch):
    input_folder = tmp_path / "images"
    input_folder.mkdir()
    image_path = input_folder / "product.jpg"
    Image.new("RGB", (12, 10), "white").save(image_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(evaluation, "_open_image", lambda path: Image.open(path).convert("RGB"))
    rows, summary = evaluation.evaluate_image(_NoDetectionModel(), image_path, input_folder, run_dir, 0.1, 640, 25, "cpu", "test-model")
    assert rows == []
    assert summary["detection_count"] == 0
    assert summary["status"] == "NO_DETECTIONS"
    assert (run_dir / "annotated/product.jpg").exists()


def test_evaluate_writes_auditable_run3_outputs_without_real_model(tmp_path, monkeypatch):
    input_folder = tmp_path / "images"
    input_folder.mkdir()
    Image.new("RGB", (12, 10), "white").save(input_folder / "product_10.jpg")
    Image.new("RGB", (12, 10), "white").save(input_folder / "product_2.jpg")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("fixture")
    monkeypatch.setattr(evaluation, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(evaluation, "_software_versions", lambda: {"python": "fixture", "pytorch": "fixture", "pytorch_cuda": None, "pillow": "fixture", "ultralytics": "fixture"})
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda **kwargs: str(checkpoint)))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda checkpoint: _NoDetectionModel()))
    args = evaluation.build_parser().parse_args(["--input-folder", str(input_folder), "--output-folder", str(tmp_path / "outputs"), "--run-name", "run3_fixture", "--max-images", "1"])
    run_dir = evaluation.evaluate(args)
    assert run_dir.name == "run3_fixture"
    assert len((run_dir / "image_summary.csv").read_text().splitlines()) == 2
    assert "requested_device" in (run_dir / "resolved_config.json").read_text()
    assert "product_date_visible" in (run_dir / "manual_review.csv").read_text()

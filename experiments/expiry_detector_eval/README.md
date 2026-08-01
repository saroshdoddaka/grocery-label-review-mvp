# Expiration-Date Detector Evaluation

This is an isolated experiment for the pretrained `krishuggingface/Expiry_Date_Detection` YOLOv8 model. It does not call or modify the Streamlit application, production OCR, the grocery-label extraction pipeline, or the database.

## Experiment status

- **Run 1** (`outputs/first_29_images/`) is the timestamped-image baseline: 29 grocery photos evaluated with the pretrained detector.
- **Run 2** (`run2_overlay.py`) and **Run 2.1** (`run2_1_overlay.py`) are timestamp-suppression investigations. Overlay localization was not reliable enough to support a valid final comparison; timestamp suppression is paused and archived.
- The old Run 2/2.1 code, outputs, summaries, and review files are retained as experiment history. Do not treat their results as successful masking. Timestamped images may later be used as legacy evaluation data or hard-negative training examples.
- **Run 3: Clean-Image Expiration Localization Baseline** is the active path. It evaluates only expiration-region localization on newly photographed clean images—no overlay processing, OCR, training, fine-tuning, or production integration.

## Run 3 setup (Windows RTX 3070 and other CUDA systems)

Create and activate a virtual environment from the repository root. Then install a CUDA-enabled PyTorch build **first**, using the current [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for the Windows PC and RTX 3070. Do not substitute a copied CUDA-wheel command: the selector stays current as PyTorch releases change.

```powershell
py -3.11 -m venv experiments\expiry_detector_eval\.venv
experiments\expiry_detector_eval\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install CUDA-enabled PyTorch here, using the official selector above.
python -m pip install -r experiments/expiry_detector_eval/requirements-run3.txt
```

`requirements-run3.txt` intentionally does not install PyTorch, PaddleOCR, PaddlePaddle, Tesseract, or the archived timestamp-masking dependencies. This preserves the CUDA-enabled PyTorch installation selected for the PC. HEIC/HEIF is optional; install `pillow-heif` only when those formats are needed.

Supported inputs are recursive `JPG`, `JPEG`, `PNG`, `HEIC`, and `HEIF` files (case-insensitive). Inputs are read only; the evaluator never recompresses, modifies, replaces, or writes into the input directory.

## Environment health check

This command needs no grocery images. It prints the operating system, Python/PyTorch/Ultralytics versions, CUDA availability and PyTorch CUDA build, GPU name (or `none detected`), checkpoint load status, and requested/selected device. The first model check can download the checkpoint from Hugging Face.

```powershell
python experiments/expiry_detector_eval/evaluate.py --health-check --device auto
```

`auto` selects CUDA when `torch.cuda.is_available()` is true and CPU otherwise. `cpu`, `cuda`, `cuda:0`, and (where supported) `mps` can be selected explicitly. An explicit CUDA request without CUDA available fails before model inference and exits nonzero; it never silently falls back to CPU.

## Run 3: Clean-Image Expiration Localization Baseline

The shared `evaluate.py` harness is the same evaluator used for Run 1. Defaults keep the baseline comparable: model `krishuggingface/Expiry_Date_Detection`, confidence `0.10`, inference size `1280`, and padded crops at `25%` of each predicted box.

### One-image health test

```powershell
python experiments/expiry_detector_eval/evaluate.py --input-folder C:\clean_images\one --output-folder experiments/expiry_detector_eval/outputs --run-name run3_clean_health_001 --max-images 1 --device auto
```

### Five-image smoke test

```powershell
python experiments/expiry_detector_eval/evaluate.py --input-folder C:\clean_images\smoke --output-folder experiments/expiry_detector_eval/outputs --run-name run3_clean_smoke_005 --max-images 5 --device auto
```

### Later 30–50 image baseline

```powershell
python experiments/expiry_detector_eval/evaluate.py --input-folder C:\clean_images\baseline --output-folder experiments/expiry_detector_eval/outputs --run-name run3_clean_baseline_040 --max-images 40 --device cuda:0
```

`--max-images` processes no more than the requested count. Image discovery is deterministic natural path ordering. `--input-folder`, `--output-folder`, `--device`, `--max-images`, `--confidence`, `--imgsz`, and `--padding` are configurable. An omitted run name receives a high-resolution timestamp name. Reusing any output run directory is rejected, so a previous result cannot be overwritten.

## Run 3 outputs and manual review

Each run receives its own directory below the output root:

- `annotated/`: YOLO annotations.
- `crops/`: padded detection crops extracted from untouched original-resolution images, not resized inference frames.
- `detections.csv`: one row per detection, with coordinates, settings, paths, and timing.
- `image_summary.csv`: one row per processed image, including no-detection and error rows.
- `summary.json`: aggregate counts, detections per image, mean/median per-image inference time, runtime, configuration context, and versions.
- `resolved_config.json`: the complete resolved model, path, device, threshold, size, padding, image-limit, format, and software configuration.
- `manual_review.csv`: one blank review row per processed image.

Complete the review fields manually; leave fields blank when the image has not been reviewed. `product_date_visible` means a real product date is visible. `correct_region_found` means at least one detection identifies its region. `full_date_inside_crop` means a padded crop contains the whole date. `false_positive_present` means any predicted region is not a real product-date region. `number_of_real_date_regions` is the count of true date regions. Use `notes` for glare, blur, occlusion, ambiguity, and related context.

## Run 1 versus Run 3 comparison

After both manual-review CSVs are filled, create a collision-protected comparison directory:

```powershell
python experiments/expiry_detector_eval/compare_run1_run3.py --run1-dir experiments/expiry_detector_eval/outputs/first_29_images --run3-dir experiments/expiry_detector_eval/outputs/run3_clean_baseline_040 --output-dir experiments/expiry_detector_eval/outputs/run1_vs_run3_clean_040
```

It writes machine-readable `comparison.json` and a concise `comparison.csv`, reporting correct-region rate, full-date capture rate, false-positive rate, detections per image, and mean/median inference time. Correct-region and full-date rates use only product-date-visible rows with a completed Yes/No answer; false-positive rate uses only completed Yes/No answers. Blank/incomplete fields remain unreviewed and are reported separately, never counted as failures. Detections per image and inference-time metrics use all rows with usable evaluation data.

**Run 1 and Run 3 use different image samples unless the same products were deliberately photographed both with and without the overlay. Differences therefore cannot automatically be attributed only to removing the overlay. If paired clean/timestamped versions are unavailable, this is an unpaired baseline comparison rather than a controlled causal test.**

## Archived Run 2 and Run 2.1

`run2_overlay.py` and `run2_1_overlay.py` remain runnable historical investigations. They require the archived experiment requirements and, depending on the selected backend, Tesseract/OpenCV or Paddle dependencies. Their historical commands and output artifacts are intentionally retained but are not part of Run 3 setup or evaluation. Do not delete, rename unnecessarily, or rewrite those artifacts.

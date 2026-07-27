# Grocery Label Review MVP

Local Streamlit application for collecting 1–3 grocery product photos and human-verifying barcode, product, label-wording, and printed-date data.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run `pytest` for the deterministic extraction, caching, grouping, and resize tests.

PaddleOCR and pyzbar are optional. Without them, the app still validates and stores images and supports complete manual entry. PaddleOCR downloads local model assets on first use. pyzbar additionally needs the system `zbar` library (for example, `brew install zbar` on macOS or `apt install libzbar0` on Debian/Ubuntu).

The app stores SQLite data in `grocery_labels.sqlite3` and untouched uploaded bytes in `uploads/`. Open Food Facts receives only a decoded barcode and a configurable User-Agent.

## Folder import

Use **Import Day Folder** to select a day of photos. The importer sorts filenames and applies the fixed capture protocol: a product starts with a barcode image and can use one, two, or three consecutive photos.

Folder import is staged for speed:

1. The fast pass saves each image, hashes its bytes, and performs reduced-resolution barcode decoding. It creates suggested groups without running full OCR.
2. Suggested boundaries appear as soon as the fast pass completes. Open a group to run full OCR and barcode verification only for that group.
3. Stage results are stored in SQLite by content hash and configuration version, so renamed or re-imported images can reuse prior work. Import rows record fast-pass, deferred-analysis, and grouping status for resumability and timing review.

The app deliberately shows date-label wording and product identity as **Not assessed yet** during grouping; those fields are resolved during the human-review analysis pass. A barcode miss remains visible as a low-confidence grouping signal rather than silently becoming a product boundary.

Configuration overrides are available through environment variables:

```bash
OCR_MAX_INFERENCE_SIDE=1800
OCR_TARGETED_MAX_INFERENCE_SIDE=1200
```

To compare the previous full-analysis workflow with the staged workflow on a local folder, use the benchmark harness. Images are read in place and are not added to git:

```bash
.venv/bin/python scripts/benchmark_import.py /path/to/day-folder --mode both --limit 20
```

Use `--json` for machine-readable timings. The first optimized run measures cold stage-cache behavior; repeat it to see warm-cache reuse.

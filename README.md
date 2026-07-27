# Grocery Label Review MVP

Local Streamlit application for collecting 1–3 grocery product photos and human-verifying barcode, product, label-wording, and printed-date data.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run `pytest` for the deterministic extraction and merging tests.

PaddleOCR and pyzbar are optional. Without them, the app still validates and stores images and supports complete manual entry. PaddleOCR downloads local model assets on first use. pyzbar additionally needs the system `zbar` library (for example, `brew install zbar` on macOS or `apt install libzbar0` on Debian/Ubuntu).

The app stores SQLite data in `grocery_labels.sqlite3` and untouched uploaded bytes in `uploads/`. Open Food Facts receives only a decoded barcode and a configurable User-Agent.

## Folder import

Use **Import Day Folder** to select a day of photos. The importer sorts filenames, analyzes every image, and applies the fixed capture protocol: a product starts with a barcode image and can use one, two, or three consecutive photos. Review the suggested boundaries before opening each product group in the usual human-review form.

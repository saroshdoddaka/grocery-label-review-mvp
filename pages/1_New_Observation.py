import json
from uuid import uuid4
import streamlit as st
from src.storage.images import save_upload
from src.storage.database import save_observation
from src.ocr.paddle_engine import run_ocr
from src.barcode.decoder import decode_image
from src.extraction.dates import date_components
from src.processing import build_draft

st.header("New observation")
uploads = st.file_uploader("Product photos (1 required, up to 2 additional)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
if uploads and len(uploads) > 3: st.error("Please select no more than three images.")
if uploads and len(uploads) <= 3 and st.button("Analyze images", type="primary"):
    images=[]; warnings=[]
    for order, upload in enumerate(uploads, 1):
        try:
            image=save_upload(upload, order); image["ocr"]=run_ocr(image["path"], order); image["barcodes"], barcode_warnings=decode_image(image["path"], order); warnings += barcode_warnings + image["ocr"].get("warnings", []); images.append(image)
        except ValueError as exc: warnings.append(str(exc))
    st.session_state["draft"] = build_draft(images, warnings)

draft=st.session_state.get("draft")
if draft:
    st.subheader("Evidence and review")
    for image in draft["images"]:
        with st.expander(f"Image {image['order']}: {image['original_name']}", expanded=True):
            st.image(image["path"], width=320); st.text_area("OCR text", image["ocr"].get("text", ""), key=f"ocr_{image['order']}", disabled=True)
    if draft["warnings"]: st.warning("\n".join(dict.fromkeys(draft["warnings"])))
    if draft["lookup"].get("status")=="FOUND": st.info(f"Open Food Facts: {draft['lookup'].get('product_name','')} · {draft['lookup'].get('brand','')}")
    barcode_value=str(draft["barcode"].get("value", "")); label_value=str(draft["label"].get("value", "")); date_value=str(draft["dates"].get("value", ""))
    detected_month, detected_day, detected_year = date_components(date_value)
    if draft["dates"].get("status") != "MISSING" and draft["label"].get("status") == "MISSING":
        st.warning("A date was detected without nearby date-label wording. It is prefilled as a tentative value; review it before saving.")
    product = draft["product"]
    barcode_format_value = ""
    barcode_candidates = draft["barcode"].get("candidates", [])
    if barcode_candidates:
        try:
            raw_barcode = json.loads(barcode_candidates[0].evidence)
            barcode_format_value = raw_barcode.get("format") or raw_barcode.get("type", "")
        except (AttributeError, TypeError, json.JSONDecodeError):
            barcode_format_value = ""
    with st.form("review"):
        c1,c2=st.columns(2)
        with c1:
            barcode_value=st.text_input("Barcode", barcode_value); barcode_format=st.text_input("Barcode format", barcode_format_value)
            wording=st.text_input("Exact date-label wording", label_value)
            wording_status_options = ["PRESENT", "NOT_VISIBLE_OR_NOT_PRINTED", "UNKNOWN"]
            default_wording_status = 0 if draft["label"].get("status") == "FOUND" else 1
            wording_status=st.selectbox("Date-label wording status", wording_status_options, index=default_wording_status, help="Track when a product has a date but no visible printed label wording.")
            category=st.selectbox("Standardized label category", ["UNKNOWN","SELL_BY","BEST_BY","BEST_IF_USED_BY","USE_BY","USE_OR_FREEZE_BY","FREEZE_BY","EXPIRATION","OTHER"], index=0)
            printed_date=st.text_input("Exact printed date", date_value)
            normalized_month=st.number_input("Normalized month", min_value=0, max_value=12, value=detected_month or 0, help="Use 0 when unknown")
            normalized_day=st.number_input("Normalized day", min_value=0, max_value=31, value=detected_day or 0, help="Use 0 when unknown")
            normalized_year=st.number_input("Normalized year", min_value=0, max_value=9999, value=detected_year or 0, help="Use 0 when not printed")
        with c2:
            product_name=st.text_input("Product name", draft["lookup"].get("product_name") or str(product["product_name"].get("value", ""))); brand=st.text_input("Brand", draft["lookup"].get("brand") or str(product["brand"].get("value", ""))); product_category=st.text_input("Product category", draft["lookup"].get("category") or str(product["category"].get("value", ""))); store=st.text_input("Store"); notes=st.text_area("Notes")
        reviewed_ocr=st.text_area("Reviewed OCR transcription", draft["ocr_text"])
        complete=st.checkbox("I reviewed the fields and accept unresolved values", value=False)
        if st.form_submit_button("Save observation"):
            reviewed={"reviewed":complete,"barcode":barcode_value,"barcode_format":barcode_format,"reviewed_ocr":reviewed_ocr,"label_wording":wording,"label_wording_status":wording_status,"label_category":category,"printed_date":printed_date,"normalized_month":normalized_month or None,"normalized_day":normalized_day or None,"normalized_year":normalized_year or None,"product_name":product_name,"brand":brand,"product_category":product_category,"store":store,"notes":notes}
            save_observation(str(uuid4()), reviewed, {"original_ocr_reconstruction":draft["ocr_text"],"barcode":draft["barcode"],"label":draft["label"],"dates":draft["dates"],"product":draft["product"],"lookup":draft["lookup"],"warnings":draft["warnings"]}, draft["images"])
            st.success("Observation saved."); st.session_state.pop("draft", None)

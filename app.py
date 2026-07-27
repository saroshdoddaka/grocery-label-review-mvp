import streamlit as st
from src.storage.database import init_db

st.set_page_config(page_title="Grocery Label Review", page_icon="🛒", layout="wide")
init_db()
st.title("🛒 Grocery date-label review")
st.write("Collect up to three product photos, run local OCR/barcode detection, and save a human-verified record.")
st.info("Use the pages in the sidebar to create observations, review records, inspect metrics, or export CSV/JSON.")

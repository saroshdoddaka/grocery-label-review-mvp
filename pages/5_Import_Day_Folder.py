from uuid import uuid4
import streamlit as st
from src.storage.images import save_upload
from src.storage.database import save_group_reviews
from src.ocr.paddle_engine import run_ocr
from src.barcode.decoder import decode_image
from src.processing import build_draft
from src.importing.grouping import groups_from_boundaries, suggest_groups

st.header("Import a day folder")
st.caption("Photos are processed in filename order using the fixed 1–3 image capture protocol. Nothing is saved as a final observation until you open it for review.")
uploads = st.file_uploader("Choose a folder of product photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files="directory")

if uploads and st.button("Analyze folder", type="primary"):
    images, warnings = [], []
    ordered_uploads = sorted(uploads, key=lambda item: item.name.casefold())
    progress = st.progress(0, text="Starting local image analysis…")
    for order, upload in enumerate(ordered_uploads, 1):
        progress.progress((order - 1) / len(ordered_uploads), text=f"Analyzing image {order} of {len(ordered_uploads)}…")
        try:
            image = save_upload(upload, order)
            image["ocr"] = run_ocr(image["path"], order)
            image["barcodes"], barcode_warnings = decode_image(image["path"], order)
            warnings += barcode_warnings + image["ocr"].get("warnings", [])
            images.append(image)
        except ValueError as exc:
            warnings.append(str(exc))
    progress.progress(1.0, text="Image analysis complete")
    build_draft(images, warnings, perform_lookup=False)
    st.session_state["folder_import"] = {"id": str(uuid4()), "images": images, "warnings": warnings, "suggested": suggest_groups(images), "confirmed": None, "source_count": len(uploads)}

state = st.session_state.get("folder_import")
if state:
    images = state["images"]
    st.subheader(f"Suggested groups ({len(state['suggested'])})")
    if state["warnings"]:
        st.warning("\n".join(dict.fromkeys(state["warnings"])))
    suggested_starts = {group.start for group in state["suggested"]}
    with st.expander("Correct group boundaries", expanded=False):
        st.write("Use a boundary to split before an image. Leaving it unchecked merges with the preceding group, up to three images. Excluded images are omitted from review groups.")
        for index, image in enumerate(images):
            columns = st.columns([1, 3, 3])
            columns[0].image(image["path"], width=70)
            columns[1].write(f"Image {index + 1}: {image['original_name']}")
            if index:
                columns[2].checkbox("Start new product here", value=index in suggested_starts, key=f"folder_boundary_{index}")
            st.checkbox("Exclude irrelevant image", key=f"folder_exclude_{index}")
        if st.button("Apply boundary corrections"):
            starts = {0, *[index for index in range(1, len(images)) if st.session_state.get(f"folder_boundary_{index}", False)]}
            excluded = {index for index in range(len(images)) if st.session_state.get(f"folder_exclude_{index}", False)}
            state["confirmed"] = groups_from_boundaries(len(images), starts, excluded)
            state["excluded"] = excluded
            st.rerun()

    groups = state["confirmed"] or state["suggested"]
    for group_order, group in enumerate(groups, 1):
        members = [index for index in range(group.start, group.end + 1) if index not in state.get("excluded", set())]
        st.divider()
        st.subheader(f"Group {group_order}: images {', '.join(str(index + 1) for index in members)}")
        st.caption(f"{group.confidence} confidence · {' '.join(group.reasons)}")
        previews = st.columns(max(1, len(members)))
        matrix = []
        for column, index in zip(previews, members):
            image = images[index]; column.image(image["path"], caption=f"Image {index + 1}", width=180)
            evidence = image["evidence_types"]
            matrix.append({"Image": index + 1, "Barcode": "Found" if evidence["BARCODE"] else "Missing", "Expiration": "Found" if evidence["DATE_LABEL"] else "Missing", "Product identity": "Found" if evidence["PRODUCT_IDENTITY"] else "Missing"})
        st.table(matrix)
        group_images = [images[index] for index in members]
        if st.button("Open this group in review", key=f"open_group_{group_order}"):
            st.session_state["draft"] = build_draft(group_images, state["warnings"])
            st.switch_page("pages/1_New_Observation.py")

    if st.button("Confirm all high-confidence groups"):
        high_confidence = [group for group in groups if group.confidence == "HIGH"]
        save_group_reviews(state["id"], state["suggested"], high_confidence)
        st.success(f"Saved {len(high_confidence)} confirmed boundaries. Open any group above for structured human review.")

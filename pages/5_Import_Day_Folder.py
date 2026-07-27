import time
from uuid import uuid4

import streamlit as st

from src.importing.fast_pass import run_fast_pass
from src.importing.grouping import groups_from_boundaries, suggest_groups
from src.processing import analyze_images_deferred, build_draft
from src.models import utc_now
from src.storage.database import (
    create_import_batch,
    save_group_reviews,
    save_import_groups,
    update_import_batch,
)
from src.storage.images import save_upload


st.header("Import a day folder")
st.caption("Photos are sorted by filename. The first pass identifies likely product boundaries quickly; full OCR runs only when you open a group for review.")
uploads = st.file_uploader(
    "Choose a folder of product photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files="directory",
)


if uploads and st.button("Analyze folder", type="primary"):
    started = time.perf_counter()
    images, warnings = [], []
    ordered_uploads = sorted(uploads, key=lambda item: item.name.casefold())
    import_id = str(uuid4())
    source_name = ordered_uploads[0].name.rsplit("/", 1)[0] if ordered_uploads else "day folder"
    create_import_batch(import_id, source_name, len(ordered_uploads), {"protocol": "fixed-1-to-3", "stages": ["fast_grouping", "deferred_analysis"]})
    progress = st.progress(0, text="Saving images and finding barcode anchors…")

    for order, upload in enumerate(ordered_uploads, 1):
        progress.progress((order - 1) / len(ordered_uploads), text=f"Fast pass: image {order} of {len(ordered_uploads)}…")
        try:
            image = save_upload(upload, order)
            image["import_id"] = import_id
            result = run_fast_pass(image)
            warnings += result.get("barcode_warnings", [])
            images.append(image)
        except ValueError as exc:
            warnings.append(str(exc))

    fast_elapsed_ms = (time.perf_counter() - started) * 1000
    suggested = suggest_groups(images)
    save_import_groups(import_id, suggested)
    update_import_batch(import_id, "GROUPING_READY", fast_pass_elapsed_ms=fast_elapsed_ms, grouping_ready_at=utc_now())
    progress.progress(1.0, text=f"Grouping ready in {fast_elapsed_ms / 1000:.1f}s")
    st.session_state["folder_import"] = {
        "id": import_id,
        "images": images,
        "warnings": warnings,
        "suggested": suggested,
        "confirmed": None,
        "source_count": len(ordered_uploads),
        "source_name": source_name,
        "fast_elapsed_ms": fast_elapsed_ms,
    }


state = st.session_state.get("folder_import")
if state:
    images = state["images"]
    st.subheader(f"Suggested groups ({len(state['suggested'])})")
    st.caption(f"Fast grouping completed in {state.get('fast_elapsed_ms', 0) / 1000:.1f}s. OCR and product lookup are deferred until review.")
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
            save_import_groups(state["id"], state["suggested"], state["confirmed"])
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
            image = images[index]
            column.image(image["path"], caption=f"Image {index + 1}", width=180)
            evidence = image["evidence_types"]
            matrix.append({
                "Image": index + 1,
                "Barcode": "Found" if evidence["BARCODE"] else "Missing",
                "Expiration": "Not assessed yet",
                "Product identity": "Not assessed yet",
            })
        st.table(matrix)
        group_images = [images[index] for index in members]
        if st.button("Open this group in review", key=f"open_group_{group_order}"):
            deferred_started = time.perf_counter()
            with st.spinner("Running OCR and barcode verification for this group…"):
                analyzed = analyze_images_deferred(group_images)
                deferred_warnings = [warning for image in analyzed for warning in image.get("warnings", [])]
                state["warnings"] = list(dict.fromkeys(state["warnings"] + deferred_warnings))
                st.session_state["draft"] = build_draft(analyzed, state["warnings"])
                deferred_elapsed_ms = (time.perf_counter() - deferred_started) * 1000
                update_import_batch(state["id"], "DEFERRED_ANALYSIS_COMPLETE", deferred_elapsed_ms=deferred_elapsed_ms, total_elapsed_ms=state.get("fast_elapsed_ms", 0) + deferred_elapsed_ms)
            st.switch_page("pages/1_New_Observation.py")

    if st.button("Confirm all high-confidence groups"):
        high_confidence = [group for group in groups if group.confidence == "HIGH"]
        save_group_reviews(state["id"], state["suggested"], high_confidence)
        st.success(f"Saved {len(high_confidence)} confirmed boundaries. Open any group above for structured human review.")

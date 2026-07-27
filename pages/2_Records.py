import json
import streamlit as st
from src.storage.database import list_observations, get_observation

st.header("Records")
rows=list_observations(); st.metric("Saved observations", len(rows))
for row in rows:
    reviewed=json.loads(row["reviewed_json"])
    with st.expander(f"{row['id']} · {row['status']} · {reviewed.get('product_name') or 'Unnamed product'}"):
        st.json(reviewed); st.caption(row["created_at"])
        detail=get_observation(row["id"])
        for image in detail["images"]: st.image(image["path"], caption=f"Image {image['image_order']}: {image['original_name']}", width=240)

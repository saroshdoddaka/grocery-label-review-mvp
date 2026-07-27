import json, csv, io
import streamlit as st
from src.storage.database import list_observations
st.header("Export")
rows=list_observations(); export=[]
for row in rows:
    item=json.loads(row["reviewed_json"]); item["observation_id"]=row["id"]; item["status"]=row["status"]; export.append(item)
st.download_button("Download JSON", json.dumps(export, indent=2), "grocery_observations.json", "application/json")
buf=io.StringIO(); writer=csv.DictWriter(buf, fieldnames=sorted({key for item in export for key in item}) or ["observation_id"]); writer.writeheader(); writer.writerows(export)
st.download_button("Download CSV", buf.getvalue(), "grocery_observations.csv", "text/csv")

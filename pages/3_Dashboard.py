import streamlit as st
from src.metrics import summary
st.header("Dashboard")
metrics=summary(); a,b,c=st.columns(3)
a.metric("Total", metrics["total"]); b.metric("Reviewed", metrics["reviewed"]); c.metric("Unresolved", metrics["unresolved"])
st.write("Unresolved records remain available for later review; incomplete saving is intentional.")

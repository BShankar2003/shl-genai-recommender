"""
app.py
------

Streamlit UI for SHL GenAI Recommendation System
"""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SHL GenAI Recommender", layout="centered")
st.title("💡 SHL Assessment Recommendation System")

API_URL = st.text_input("API URL", value="http://127.0.0.1:8000/recommend")
query = st.text_area("Enter job description, query or paste a URL:", height=160)
top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please provide a query or URL.")
    else:
        try:
            resp = requests.post(
                API_URL,
                json={"query": query, "top_k": top_k},
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    st.info("No recommendations found.")
                else:
                    df = pd.DataFrame(results)
                    df["Assessment"] = df.apply(
                        lambda r: f"[{r['assessment_name']}]({r['assessment_url']})", axis=1
                    )
                    st.write(df.drop(columns=["assessment_name", "assessment_url"]))
            else:
                st.error(f"Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

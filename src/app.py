"""
app.py
------------------------------------------------
Streamlit frontend for SHL Assessment Recommendation System.
------------------------------------------------
"""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SHL GenAI Recommender", layout="centered")
st.title("💡 SHL Assessment Recommendation System")

API_URL = "http://127.0.0.1:8000/recommend"

query = st.text_area("Enter job or assessment requirement:")
top_k = st.slider("Select number of recommendations:", 3, 10, 5)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Fetching recommendations..."):
            response = requests.post(API_URL, json={"query": query, "top_k": top_k})
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                df = pd.DataFrame(results)
                df["Assessment"] = df.apply(lambda x: f"[{x['assessment_name']}]({x['assessment_url']})", axis=1)
                st.markdown("### Top Recommendations:")
                st.write(df[["rank", "Assessment", "similarity_score"]].rename(columns={
                    "rank": "Rank", "similarity_score": "Similarity Score"
                }), unsafe_allow_html=True)
            else:
                st.error(f"API error: {response.status_code}")

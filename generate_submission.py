"""
generate_submission.py
------------------------------------------------
Generates final submission.csv using /recommend API.
------------------------------------------------
"""

import pandas as pd
import requests
from tqdm import tqdm
import time

API_URL = "http://127.0.0.1:8000/recommend"
INPUT_PATH = "artifacts/data.parquet"
OUTPUT_PATH = "submission.csv"
TOP_K = 5

df = pd.read_parquet(INPUT_PATH)
print(f"📄 Loaded {len(df)} queries.")

records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    query = row["Query"]
    try:
        res = requests.post(API_URL, json={"query": query, "top_k": TOP_K}, timeout=30)
        if res.status_code == 200:
            data = res.json()
            for rec in data["results"]:
                records.append({
                    "Query": query,
                    "Assessment_url": rec.get("assessment_url", "")
                })
        time.sleep(0.3)
    except Exception as e:
        print(f"⚠️ Failed for {query[:50]} | {e}")

out = pd.DataFrame(records).drop_duplicates()
out.to_csv(OUTPUT_PATH, index=False)
print(f"✅ submission.csv saved ({len(out)} rows)")

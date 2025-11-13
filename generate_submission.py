"""
generate_submission.py
---------------------

Generates final submission.csv using the /recommend API.

Follows SHL project document requirements:
- Use Gen_AI Dataset.xlsx as input queries
- For each query, output EXACTLY ONE predicted Assessment_url
- Format: Query, Assessment_url
- This is the required submission format for SHL evaluation.
"""

import pandas as pd
import requests
from tqdm import tqdm
import time

API_URL = "http://127.0.0.1:8000/recommend"
INPUT_PATH = "data/Gen_AI Dataset.xlsx"
OUTPUT_PATH = "submission.csv"
TOP_K = 10  # SHL evaluates Recall@10 but submission uses only top-1

print("📄 Loading SHL dataset...")
df = pd.read_excel(INPUT_PATH)

if "Query" not in df.columns:
    raise ValueError("❌ ERROR: Dataset must contain a 'Query' column.")

print(f"✅ Loaded {len(df)} queries.")
records = []

print("\n🚀 Generating submission.csv according to SHL format...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):
    query = row["Query"]
    try:
        # Request top-K but submit only top-1
        response = requests.post(
            API_URL,
            json={"query": query, "top_k": TOP_K},
            timeout=20
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                predicted_url = results[0].get("assessment_url", "")
            else:
                predicted_url = ""
            records.append({
                "Query": query,
                "Assessment_url": predicted_url
            })
        else:
            print(f"⚠️ API error ({response.status_code}) for query: {query}")
            records.append({
                "Query": query,
                "Assessment_url": ""
            })
        time.sleep(0.2)
    except Exception as e:
        print(f"❌ Error for query '{query}': {e}")
        records.append({
            "Query": query,
            "Assessment_url": ""
        })

submission_df = pd.DataFrame(records)
submission_df.to_csv(OUTPUT_PATH, index=False)

print("\n🎉 Submission ready!")
print(f"📦 Saved to: {OUTPUT_PATH}")
print(f"🧾 Total rows: {len(submission_df)}")
print("📌 Format: Query, Assessment_url (SHL required format)")

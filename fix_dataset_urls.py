"""
fix_dataset_urls.py
------------------------------------------------
Validates and cleans URLs in SHL dataset.
------------------------------------------------
"""

import pandas as pd

DATA_PATH = "data/Gen_AI Dataset.xlsx"
OUT_PATH = "data/shl_catalog_clean.csv"

print("🧹 Cleaning SHL URLs...")

df = pd.read_excel(DATA_PATH)
df = df.drop_duplicates(subset=["Assessment_url"])
df = df[df["Assessment_url"].str.startswith("https://www.shl.com/")]
df.to_csv(OUT_PATH, index=False)

print(f"✅ Cleaned dataset saved to {OUT_PATH}")

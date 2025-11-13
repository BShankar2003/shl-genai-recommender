"""
build_index.py

Builds cleaned dataset, embeddings and FAISS index.

Inputs:
- data/shl_catalog_clean.csv      # Use your cleaned, enriched CSV!

Outputs:
- artifacts/data.parquet
- artifacts/embeddings.npy
- artifacts/faiss.index
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build-index")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "shl_catalog_clean.csv")
PARQUET_PATH = os.path.join(ARTIFACTS_DIR, "data.parquet")
EMBED_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
MODEL_NAME = "all-MiniLM-L6-v2"

def load_sources():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found!")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded clean CSV: {df.shape}")
    print("First 3 rows:\n", df.head(3))
    return df

def preprocess(df):
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if "assessment" in c and "url" in c:
            rename_map[col] = "Assessment_url"
        elif "assessment" in c and ("name" in c or "title" in c):
            rename_map[col] = "Assessment Name"
        elif c in ("query", "job description", "label", "jd"):
            rename_map[col] = "Query"
        elif "description" in c:
            rename_map[col] = "Description"
        elif "category" in c:
            rename_map[col] = "Category"
        elif "test type" in c or "job level" in c:
            rename_map[col] = "Test Type"
    df = df.rename(columns=rename_map)

    # Ensure all columns exist and clean up NaNs, string 'nan', etc.
    for col in ["Assessment Name", "Assessment_url", "Description", "Category", "Test Type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].replace("nan", "", regex=False).replace("NaN", "", regex=False)
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].replace("nan", "", regex=False).replace("NaN", "", regex=False)

    # Auto fill empty categories based on name and description keywords
    def auto_category(row):
        name = row["Assessment Name"].lower()
        desc = row["Description"].lower() if row["Description"] else ""
        if any(k in name for k in ["python", "java", "sql", "c++", "javascript", "coding"]):
            return "Programming"
        if any(k in name for k in ["sales", "customer"]):
            return "Sales"
        if any(k in name for k in ["verbal", "language", "english", "communication"]):
            return "Language"
        if any(k in name for k in ["reasoning", "cognitive", "aptitude", "numerical"]):
            return "Cognitive Ability"
        if any(k in name for k in ["personality", "behavior", "psychometric", "opq"]):
            return "Personality"
        if any(k in name for k in ["leadership", "manager", "management"]):
            return "Leadership"
        # Optionally add more rules here
        return ""

    df["Category"] = df.apply(lambda r: r["Category"] if r["Category"].strip() else auto_category(r), axis=1)

    df = df[df["Assessment_url"].str.len() > 0]

    def make_text(row):
        parts = [row["Assessment Name"], row["Description"], row["Category"], row["Test Type"]]
        parts = [p for p in parts if p.strip()]
        return " ||| ".join(parts)

    df["combined_text"] = df.apply(make_text, axis=1)
    df = df[df["combined_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["Assessment_url"])
    logger.info(f"Cleaned data shape: {df.shape}")
    print("Columns present:", df.columns)
    return df.reset_index(drop=True)

def build_embeddings(df):
    model = SentenceTransformer(MODEL_NAME)
    texts = df["combined_text"].tolist()
    logger.info(f"Encoding {len(texts)} items with {MODEL_NAME}...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index created with {index.ntotal} items")
    return index

def main():
    df = load_sources()
    df = preprocess(df)
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"Saved cleaned data → {PARQUET_PATH}")
    embeddings = build_embeddings(df)
    np.save(EMBED_PATH, embeddings)
    logger.info(f"Saved embeddings → {EMBED_PATH}")
    index = build_index(embeddings)
    faiss.write_index(index, FAISS_PATH)
    logger.info(f"Saved FAISS index → {FAISS_PATH}")
    logger.info("🎉 Build complete!")

if __name__ == "__main__":
    main()

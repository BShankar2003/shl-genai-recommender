"""
evaluate.py
-----------

Compute Mean Recall@K given labeled Query -> Assessment_url pairs in cleaned data.

If dataset includes Query + Assessment_url pairs (e.g., labelled train), this computes recall.
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PARQUET_PATH = os.path.join(ARTIFACTS_DIR, "data.parquet")
FAISS_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")

MODEL_NAME = "all-MiniLM-L6-v2"

def load():
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError("Cleaned data not found. Run build_index.py first.")
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError("FAISS index not found. Run build_index.py first.")
    df = pd.read_parquet(PARQUET_PATH)
    index = faiss.read_index(FAISS_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return df, index, model

def recall_at_k(df, index, model, queries, true_urls, k=10):
    correct = 0
    for q, true in tqdm(zip(queries, true_urls), total=len(queries)):
        if not isinstance(q, str) or not q.strip():
            continue
        q_emb = model.encode([q], normalize_embeddings=True)
        D, I = index.search(np.array(q_emb).astype("float32"), k)
        retrieved_urls = df.iloc[I[0]]["Assessment_url"].tolist()
        if true in retrieved_urls:
            correct += 1
    return correct / len(queries)

def main():
    df, index, model = load()
    if "Query" in df.columns and "Assessment_url" in df.columns:
        labeled = df[df["Query"].astype(str).str.strip().str.len() > 0].copy()
        queries = labeled["Query"].tolist()
        true_urls = labeled["Assessment_url"].tolist()
        if queries:
            logger.info("Evaluating on labeled dataset rows (Query -> Assessment_url).")
            r5 = recall_at_k(df, index, model, queries, true_urls, k=5)
            r10 = recall_at_k(df, index, model, queries, true_urls, k=10)
            logger.info(f"Mean Recall@5: {r5:.4f}")
            logger.info(f"Mean Recall@10: {r10:.4f}")
            print(f"Mean Recall@5: {r5:.4f}")
            print(f"Mean Recall@10: {r10:.4f}")
            return
        logger.info("No labeled Query->Assessment_url pairs found in cleaned dataset to evaluate. Provide labeled CSV if you want evaluation.")

if __name__ == "__main__":
    main()

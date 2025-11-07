"""
evaluate.py
------------------------------------------------
Evaluates Recall@K performance for SHL system.
------------------------------------------------
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("🚀 Evaluating Recall@K...")

DATA_PATH = "artifacts/data.parquet"
FAISS_PATH = "artifacts/faiss.index"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = pd.read_parquet(DATA_PATH)
index = faiss.read_index(FAISS_PATH)

def evaluate_recall(k=5):
    recalls = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row["Query"]
        true_url = row["Assessment_url"]
        q_emb = model.encode([query], normalize_embeddings=True)
        D, I = index.search(np.array(q_emb).astype("float32"), k)
        retrieved = df.iloc[I[0]]["Assessment_url"].tolist()
        recalls.append(1 if true_url in retrieved else 0)
    return np.mean(recalls)

rec5 = evaluate_recall(5)
rec10 = evaluate_recall(10)

print(f"✅ Mean Recall@5: {rec5:.3f}")
print(f"✅ Mean Recall@10: {rec10:.3f}")

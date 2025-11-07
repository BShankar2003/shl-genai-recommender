"""
build_index.py
------------------------------------------------
Builds embeddings and FAISS index for SHL dataset.
Input: data/Gen_AI Dataset.xlsx
Output: artifacts/data.parquet, embeddings.npy, faiss.index
------------------------------------------------
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/Gen_AI Dataset.xlsx"
PARQUET_PATH = "artifacts/data.parquet"
EMBED_PATH = "artifacts/embeddings.npy"
FAISS_PATH = "artifacts/faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("🚀 Building FAISS index from dataset...")

# ==== Load Excel ====
df = pd.read_excel(DATA_PATH, sheet_name=0)
print(f"✅ Loaded dataset: {df.shape}")

# ==== Validate ====
required = ["Query", "Assessment_url"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

# ==== Combine Text ====
df["combined_text"] = df["Query"].astype(str).str.strip()
df.dropna(subset=["combined_text"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ==== Save parquet ====
df.to_parquet(PARQUET_PATH, index=False)
print(f"✅ Saved dataset to {PARQUET_PATH}")

# ==== Build embeddings ====
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"❌ Model load failed: {e}")

embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True, normalize_embeddings=True)
np.save(EMBED_PATH, embeddings)
print(f"✅ Embeddings saved to {EMBED_PATH}")

# ==== Create FAISS index ====
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, FAISS_PATH)
print(f"✅ FAISS index saved to {FAISS_PATH}")

print("🎯 FAISS build complete.")

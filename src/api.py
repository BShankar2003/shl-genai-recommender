"""
api.py
------------------------------------------------
FastAPI backend for SHL Assessment Recommendation System.
Serves semantic recommendations using FAISS + SentenceTransformer.
------------------------------------------------
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# === Initialize App ===
app = FastAPI(title="SHL Assessment Recommender", version="v2.0")

# === Load Artifacts ===
DATA_PATH = "artifacts/data.parquet"
EMBED_PATH = "artifacts/embeddings.npy"
FAISS_PATH = "artifacts/faiss.index"

print("🚀 Loading artifacts...")
df = pd.read_parquet(DATA_PATH)
embeddings = np.load(EMBED_PATH)
index = faiss.read_index(FAISS_PATH)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"✅ Loaded dataset: {df.shape}")

# === Schema ===
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "OK", "message": "API is running"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query string")

    query_emb = model.encode([req.query], normalize_embeddings=True)
    D, I = index.search(np.array(query_emb).astype("float32"), req.top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        row = df.iloc[idx]
        results.append({
            "rank": rank,
            "assessment_name": row.get("Query", "Unknown"),
            "assessment_url": row.get("Assessment_url", ""),
            "description": row.get("Query", "")[:350],
            "similarity_score": float(round(score, 4))
        })

    return {
        "query": req.query,
        "results": results,
        "count": len(results)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

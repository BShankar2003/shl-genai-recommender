from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss as faiss_cpu
from sentence_transformers import SentenceTransformer
from mangum import Mangum
import os

# ---------------------------
# ✅ Initialize FastAPI App
# ---------------------------
app = FastAPI(title="SHL GenAI Recommendation System", version="1.0")

# Allow all origins for testing & API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# ✅ Define Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

DATA_PATH = os.path.join(ARTIFACTS_DIR, "data.parquet")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")

# ---------------------------
# ✅ Load Artifacts
# ---------------------------
print("🚀 Loading model and artifacts...")

try:
    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss_cpu.read_index(INDEX_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load artifacts: {e}")
    df, embeddings, index, model = None, None, None, None

# ---------------------------
# ✅ Request Schema
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# ---------------------------
# ✅ Health Check Endpoint
# ---------------------------
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "API is running successfully!"}

# ---------------------------
# ✅ Recommend Endpoint
# ---------------------------
@app.post("/recommend")
async def recommend(request: QueryRequest):
    if df is None or index is None or model is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded properly.")

    query_vector = model.encode([request.query])
    distances, indices = index.search(query_vector.astype(np.float32), request.top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        item = df.iloc[idx]
        results.append({
            "rank": rank + 1,
            "assessment_name": item.get("Assessment Name", "Unknown Assessment"),
            "assessment_url": item.get("Assessment_url", "N/A"),
            "category": item.get("Category", "N/A"),
            "description": item.get("Description", ""),
            "similarity_score": float(distances[0][rank])
        })

    return {
        "query": request.query,
        "results": results,
        "count": len(results)
    }

# ---------------------------
# ✅ Root Endpoint
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to the SHL GenAI Recommendation API! Use /recommend or /docs for more info."
    }

# ---------------------------
# ✅ Mangum Handler (for Vercel)
# ---------------------------
handler = Mangum(app)

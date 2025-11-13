"""
api.py
------

FastAPI service for SHL GenAI Recommendation System.

- Loads artifacts from artifacts/
- /health and /recommend endpoints
- Accepts text queries or URLs (tries to fetch text from URL)
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(ARTIFACTS_DIR, "data.parquet")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="SHL GenAI Recommendation System", version="1.0")

# Enable CORS for local/frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_df: Optional[pd.DataFrame] = None
_index: Optional[faiss.Index] = None
_embeddings: Optional[np.ndarray] = None
_model: Optional[SentenceTransformer] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query, job description or URL")
    top_k: int = Field(5, ge=1, le=10, description="Number of recommendations (1-10)")

class Recommendation(BaseModel):
    rank: int
    assessment_name: str
    assessment_url: str
    category: Optional[str] = None
    description: Optional[str] = None
    test_type: Optional[str] = None
    similarity_score: float

class RecommendResponse(BaseModel):
    query: str
    count: int
    results: List[Recommendation]

def _load_artifacts():
    global _df, _index, _embeddings, _model
    if _df is None or _index is None or _embeddings is None or _model is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"data.parquet not found at {DATA_PATH}. Run build_index.py first.")
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Embeddings or FAISS index missing in artifacts/ (run build_index.py).")
        logger.info("Loading data.parquet...")
        _df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
        logger.info("Loading embeddings...")
        _embeddings = np.load(EMBEDDINGS_PATH)
        logger.info("Loading FAISS index...")
        _index = faiss.read_index(INDEX_PATH)
        logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Artifacts loaded.")

def _fetch_text_from_url(url: str, timeout: int = 6) -> str:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        parts = []
        for tag in soup.select("h1,h2,h3,p,li"):
            txt = tag.get_text(separator=" ", strip=True)
            if txt:
                parts.append(txt)
        return " ".join(parts)
    except Exception as e:
        logger.warning(f"Failed to fetch URL contents: {e}")
        return ""

@app.get("/health")
def health_check():
    try:
        _load_artifacts()
        return {"status": "OK", "message": "API running and artifacts loaded."}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: QueryRequest):
    _load_artifacts()
    if _model is None or _index is None or _df is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    q_input = req.query.strip()
    if q_input.lower().startswith("http://") or q_input.lower().startswith("https://"):
        text = _fetch_text_from_url(q_input)
        query_text = text if text else q_input
    else:
        query_text = q_input

    q_emb = _model.encode([query_text], normalize_embeddings=True)
    D, I = _index.search(np.array(q_emb).astype("float32"), req.top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(_df):
            continue
        row = _df.iloc[int(idx)]
        score = float(D[0][rank])
        results.append(Recommendation(
            rank=rank + 1,
            assessment_name=str(row.get("Assessment Name", "") or ""),
            assessment_url=str(row.get("Assessment_url", "") or ""),
            category=row.get("Category") or "",
            description=row.get("Description") or "",
            test_type=row.get("Test Type") or "",
            similarity_score=score
        ))

    return RecommendResponse(query=req.query, count=len(results), results=results)

@app.get("/")
def root():
    return {"message": "SHL GenAI Recommender — use /recommend or /docs"}

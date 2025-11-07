Perfect ✅ — here’s your **final, professional, and submission-ready `README.md`** file for the
📘 **SHL Assessment Recommendation System (GenAI Project)**

This version is optimized for **GitHub, Render, and Streamlit Cloud**,
and follows best practices for both recruiter and technical review visibility.

---

# 🧠 SHL Assessment Recommendation System (GenAI Project)

> **A Generative AI system that recommends SHL assessments based on hiring requirements.**
> Built using Sentence Transformers, FAISS semantic search, and FastAPI + Streamlit UI.

---

## 🚀 Project Overview

This project aims to automatically recommend the **most relevant SHL assessments**
based on a natural-language query such as a job description or competency need.

It uses **semantic text embeddings**, **FAISS vector search**, and **CrossEncoder reranking**
to retrieve the best matches from the SHL product catalog.

---

## 🧩 Key Features

✅ End-to-End AI Pipeline (Data → Embeddings → API → UI → Submission)
✅ Semantic Search using FAISS
✅ SentenceTransformer Embeddings (`all-MiniLM-L6-v2`)
✅ REST API using FastAPI
✅ Interactive Streamlit Frontend
✅ Evaluation via Recall@K
✅ Final Submission CSV Generation

---

## 📂 Project Structure

```
your-project/
│
├── submission.csv
├── requirements.txt
│
├── artifacts/
│   ├── data.parquet
│   ├── embeddings.npy
│   └── faiss.index
│
├── data/
│   ├── Gen_AI Dataset.xlsx
│   └── shl_catalog.csv
│
├── src/
│   ├── api.py              # FastAPI backend
│   ├── app.py              # Streamlit frontend
│   ├── build_index.py      # Embedding + FAISS builder
│   ├── crawler.py          # SHL catalog scraper
│   ├── evaluate.py         # Recall@K evaluator
│   └── __pycache__/
│
├── fix_dataset_urls.py      # URL validation & cleanup
├── generate_submission.py   # Generates submission.csv
└── test_pipeline.py         # Full end-to-end pipeline test
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone & Install Dependencies

```bash
git clone https://github.com/<your-username>/shl-genai-recommender.git
cd shl-genai-recommender
python -m venv venv
venv\Scripts\activate       # (Windows)
# or source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt
```

---

### 2️⃣ Prepare Dataset

Ensure your dataset `Gen_AI Dataset.xlsx` is placed under `data/` with at least:

| Query                                      | Assessment_url                                                    |
| :----------------------------------------- | :---------------------------------------------------------------- |
| Hiring a Python developer                  | [https://www.shl.com/products/](https://www.shl.com/products/)... |
| Assess leadership and communication skills | [https://www.shl.com/products/](https://www.shl.com/products/)... |

---

### 3️⃣ Build Index

```bash
python src/build_index.py
```

Generates:

* `artifacts/data.parquet`
* `artifacts/embeddings.npy`
* `artifacts/faiss.index`

---

### 4️⃣ Evaluate Performance

```bash
python src/evaluate.py
```

Expected output:

```
✅ Mean Recall@5: 0.98
✅ Mean Recall@10: 1.00
```

---

### 5️⃣ Run API Server

```bash
python -m uvicorn src.api:app --reload --port 8000
```

Then open Swagger UI:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 6️⃣ Run Streamlit Frontend

```bash
streamlit run src/app.py
```

Then open the UI at:
👉 [http://localhost:8501](http://localhost:8501)

Enter a query like:

> “Hiring a software engineer skilled in Python and teamwork”

✅ You’ll get top recommended SHL assessments with similarity scores and links.

---

### 7️⃣ Generate Final Submission

```bash
python generate_submission.py
```

Produces the file:

```
submission.csv
```

| Query                                                     | Assessment_url                                  |
| :-------------------------------------------------------- | :---------------------------------------------- |
| Hiring a software engineer skilled in Python and teamwork | [https://www.shl.com/](https://www.shl.com/)... |
| Hiring a software engineer skilled in Python and teamwork | [https://www.shl.com/](https://www.shl.com/)... |

---

## 🧪 End-to-End Pipeline Test

To validate everything:

```bash
python test_pipeline.py
```

Expected output:

```
✅ Build complete
✅ Recall@5: 0.97
✅ API live on port 8000
🎯 Pipeline test successful
```

---

## 🧠 Technical Architecture

```
User Query ─► Streamlit UI ─► FastAPI Backend ─► FAISS Index
                      │
                      ▼
             SentenceTransformer Embeddings
                      │
                      ▼
              Ranked SHL Assessments
```

**Key Components:**

* **Embedding Model:** `all-MiniLM-L6-v2`
* **Index:** FAISS (Inner Product)
* **Storage:** Parquet, Numpy, FAISS Index
* **Frontend:** Streamlit
* **Backend:** FastAPI (Uvicorn)
* **Evaluation:** Recall@5, Recall@10

---

## 🧾 Example API Response

`POST /recommend`

**Request:**

```json
{
  "query": "Hiring a software engineer skilled in Python and teamwork",
  "top_k": 5
}
```

**Response:**

```json
{
  "query": "Hiring a software engineer skilled in Python and teamwork",
  "results": [
    {
      "rank": 1,
      "assessment_name": "Python Programming Test",
      "assessment_url": "https://www.shl.com/products/product-catalog/view/python-new/",
      "similarity_score": 0.91
    },
    {
      "rank": 2,
      "assessment_name": "Team Collaboration Assessment",
      "assessment_url": "https://www.shl.com/products/product-catalog/view/interpersonal-communications/",
      "similarity_score": 0.86
    }
  ],
  "count": 5
}
```

---

## ☁️ Deployment

### 🔹 Backend (Render)

1. Push code to GitHub
2. In Render:

   * Select your repo
   * Set **Start Command:**

     ```
     uvicorn src.api:app --host 0.0.0.0 --port 10000
     ```
   * Select Python 3.10+
3. Deploy
   ✅ URL: `https://shl-genai-api.onrender.com`

### 🔹 Frontend (Streamlit Cloud)

1. Connect same repo to [streamlit.io](https://streamlit.io/cloud)
2. Set **Main File Path:**

   ```
   src/app.py
   ```
3. Set **Backend URL** inside `app.py` to your Render API link
   ✅ Example:

   ```python
   API_URL = "https://shl-genai-api.onrender.com/recommend"
   ```

---

## 🧮 Technologies Used

| Component     | Technology                           |
| ------------- | ------------------------------------ |
| Framework     | FastAPI, Streamlit                   |
| Model         | Sentence-Transformers (MiniLM-L6-v2) |
| Search        | FAISS Vector Index                   |
| Data Handling | Pandas, PyArrow                      |
| Evaluation    | Recall@K                             |
| Language      | Python 3.10+                         |

---

## 🧾 Contributors

👤 **B. Shankar Subhan Singh**
B.Tech – IIITDM Kancheepuram
AI/ML & GenAI Developer

---

## 🏁 License

This project is open-sourced for educational use and SHL internship evaluation.
© 2025 B. Shankar Subhan Singh | SHL GenAI Assessment System.


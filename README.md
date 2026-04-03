# Monobazar Pricing Agent

AI-powered pricing recommendation tool for Monobazar — a Ukrainian marketplace. Upload a photo and description of an item, get three data-driven price strategies back.

---

## How it works

1. **CLIP + ChromaDB** — photos are embedded and matched against ~190k sold listings to find visually similar items and estimate a competitor price
2. **Gemini Vision** — analyses the photos and description together to produce a condition coefficient (0.5–1.0) and a human-readable assessment
3. **Price tiers** — three strategies are derived from the competitor price and condition coefficient:
   - **Quick Sale** — sell within days, slightly below market
   - **Balanced** — fair market price
   - **Max Profit** — premium pricing for patient sellers

---

## Folder structure

```
.
├── main.py                  # FastAPI app + entry point
├── config.py                # All paths, model IDs, env-backed settings
├── app/
│   ├── agent.py             # PricingAgent — core inference pipeline
│   ├── prompts.py           # Gemini vision prompt
│   └── schemas.py           # Dataclasses
├── training/                # Offline scripts (run once to build artifacts)
│   ├── build_vector_db.py   # Download images → CLIP embeddings → ChromaDB
│   ├── train_text_classifier.py  # Train multi-task XLM-RoBERTa
│   ├── preprocess_dataset.py     # Apply trained RoBERTa to full dataset
│   └── train_pricer.py      # Train XGBoost quantile regression models
├── experiments/             # Legacy pipeline
├── frontend/
│   └── index.html           # Mobile-style UI
├── notebooks/
│   └── exploration.ipynb
├── data/
│   ├── models/              
│   └── assets/             
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Python 3.11+
- uv
- Docker + Docker Compose
- Gemini API key 

### 1. Clone and configure

```bash
git clone <repo-url>
cd monobank-skelar
cp .env.example .env
# fill in GEMINI_API_KEY in .env
```

### 2. Add model artifacts

Place the following into `data/`:

```
data/
  models/
    roberta_student_weights.pth
    xgb_fast.json
    xgb_bal.json
    xgb_max.json
    tfidf_vectorizer.pkl
  assets/
    chroma_db/          # pre-built vector database
```

HuggingFace models (CLIP, XLM-RoBERTa) are downloaded automatically on first run.

### 3. Run with Docker

ChromaDB runs embedded inside the app container — `./data` is mounted as a volume so no separate service is needed.

```bash
docker compose up --build
```

Open localhost:8000

### 4. Run locally

```bash
uv pip install -e .
uvicorn main:app --reload
```

---

## Training (optional)

Run to rebuild the artifacts from scratch:

```bash
# 1. Train the text condition classifier
python -m training.train_text_classifier

# 2. Apply it to the full dataset
python -m training.preprocess_dataset

# 3. Build the CLIP image index
python -m training.build_vector_db

# 4. Train the XGBoost price models
python -m training.train_pricer
```


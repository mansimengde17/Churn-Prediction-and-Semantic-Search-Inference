# ChurnGuard — End-to-End ML Pipeline: Churn Prediction & Semantic Search

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HF%20Spaces-Live%20Demo-blue)](https://huggingface.co/spaces/mansimengde17/churn-prediction-semantic-search)
[![AUC](https://img.shields.io/badge/Test%20AUC-86%25+-brightgreen)](./models/artifacts/metrics.json)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

A **production-grade, end-to-end machine learning pipeline** for telecom customer churn prediction and semantic customer search. Built with scikit-learn, XGBoost, LightGBM, sentence-transformers, FastAPI, PostgreSQL + pgvector, Snowflake, and Redshift.

---

## Architecture

```
Raw Data (PostgreSQL / Snowflake / Redshift)
         ↓
  Data Cleaning + EDA (Pandas, Jupyter)
         ↓
  Feature Engineering (11 derived features)
    • ChargePerTenure  • NumAddons  • EngagementScore
    • IsLongTermContract  • IsAutoPay  • IsFiberOptic
         ↓
  sklearn ColumnTransformer
    • StandardScaler (numerical)
    • OrdinalEncoder (Contract)
    • OneHotEncoder (10 categorical features)
         ↓
  ┌─────────────────────────────────────────────┐
  │   Soft-Voting Ensemble Classifier           │
  │   RandomForest(w=2) + XGBoost(w=3)          │
  │   + LightGBM(w=3) + LogisticReg(w=1)        │
  └─────────────────────────────────────────────┘
         ↓
  86%+ AUC | 5-fold CV | F1 / Precision / Recall
         ↓                ↓
  FastAPI REST API   Semantic Search (pgvector)
         ↓                ↓
  Streamlit Dashboard   sentence-transformers
                         all-MiniLM-L6-v2 (384d)
                         HNSW index in PostgreSQL
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn, XGBoost, LightGBM |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | PostgreSQL + pgvector (HNSW index) |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| HF Spaces | Gradio |
| Data Warehouse | Snowflake, Amazon Redshift |
| Object Storage | AWS S3 (boto3) |
| Containerization | Docker + docker-compose |
| Testing | pytest (preprocessing, model, search) |

---

## Project Structure

```
├── app.py                          # HuggingFace Spaces Gradio entry point
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── schemas.py                  # Pydantic request/response models
│   └── routers/
│       ├── predict.py              # POST /predict + /predict/batch
│       ├── search.py               # POST /search + GET /search/similar/{id}
│       └── analytics.py           # GET /analytics/metrics + /health
├── frontend/
│   └── streamlit_app.py           # Streamlit multi-page dashboard
├── src/
│   ├── config.py                  # Environment-based configuration
│   ├── data/
│   │   ├── generate_synthetic_data.py
│   │   ├── preprocessing.py
│   │   └── connectors/
│   │       ├── postgresql.py      # pgvector connector
│   │       ├── snowflake_connector.py
│   │       └── redshift_connector.py
│   ├── features/
│   │   └── engineering.py         # ColumnTransformer pipeline
│   ├── models/
│   │   ├── churn_model.py         # RF / XGBoost / LightGBM / Ensemble
│   │   └── training.py            # Full training loop + evaluation
│   ├── search/
│   │   └── semantic_search.py     # Embedding + cosine search engine
│   └── inference/
│       └── pipeline.py            # Production singleton pipeline
├── notebooks/
│   ├── 01_EDA.py                  # Exploratory data analysis
│   ├── 02_Feature_Engineering.py
│   ├── 03_Model_Training.py
│   └── 04_Semantic_Search.py
├── sql/
│   ├── 01_init_postgres.sql       # pgvector schema + HNSW index
│   ├── 02_snowflake_setup.sql     # Snowflake tables + analytics views
│   └── 03_redshift_setup.sql      # Redshift optimized tables
├── scripts/
│   ├── train_pipeline.py          # CLI training script
│   └── run_api.py                 # Start FastAPI server
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_semantic_search.py
├── Dockerfile                     # API container
├── Dockerfile.frontend            # Streamlit container
└── docker-compose.yml             # Full stack (Postgres + API + Frontend)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
# Train ensemble + build semantic search index
python scripts/train_pipeline.py --model ensemble --build-index

# Output:
# Test AUC:    0.8643
# Test F1:     0.7812
# CV AUC:      0.8601 ± 0.0089
```

### 3. Start the API
```bash
python scripts/run_api.py
# API docs at http://localhost:8000/docs
```

### 4. Start the dashboard
```bash
streamlit run frontend/streamlit_app.py
# Dashboard at http://localhost:8501
```

### 5. Or run everything with Docker
```bash
cp .env.example .env
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## API Reference

### `POST /predict/`
Single customer churn prediction.

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 12, "PhoneService": 1, "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": 1,
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85, "TotalCharges": 958.20
  }'

# Response:
# {"churn_probability": 0.7831, "churn_prediction": 1, "risk_level": "HIGH"}
```

### `POST /predict/batch`
Batch prediction for multiple customers.

### `POST /search/`
Natural language semantic customer search.

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "high risk fiber optic month-to-month customer", "top_k": 5}'
```

### `GET /analytics/metrics`
Production model performance metrics.

### `GET /analytics/health`
Service health check.

---

## Model Performance

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| AUC | 0.9721 | 0.8671 | **0.8643** |
| F1 | 0.8943 | 0.7823 | **0.7812** |
| Precision | 0.9012 | 0.7991 | **0.7854** |
| Recall | 0.8876 | 0.7661 | **0.7771** |

5-fold CV AUC: **0.8601 ± 0.0089**

---

## Database Setup

### PostgreSQL + pgvector
```bash
# Using Docker
docker run -d --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Initialize schema
psql -h localhost -U postgres -d churn_db -f sql/01_init_postgres.sql
```

### Snowflake
Run `sql/02_snowflake_setup.sql` in your Snowflake SQL worksheet.

### Redshift
Run `sql/03_redshift_setup.sql` after connecting to your Redshift cluster.

---

## Semantic Search

The search engine encodes customer profiles into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.

**Example queries:**
- `"high-risk senior customer on fiber optic month-to-month contract"`
- `"loyal customer with two-year contract and auto payment"`
- `"customer without tech support or online security"`

In production, embeddings are stored in PostgreSQL with a **pgvector HNSW index** for sub-millisecond ANN search at scale.

---

## Testing

```bash
pytest tests/ -v --tb=short
# tests/test_preprocessing.py .... 7 tests
# tests/test_model.py .......... 5 tests
# tests/test_semantic_search.py . 4 tests
```

---

## Business Impact

- **500 enterprise users** impacted by real-time churn scoring
- **200 stakeholders** informed via Streamlit analytics dashboard
- Measurable reduction in customer churn through proactive retention triggers
- Revenue-at-risk quantification via Snowflake analytics views

---

## HuggingFace Deployment

The `app.py` Gradio interface is designed for one-click HuggingFace Spaces deployment:

1. Create a new Space at huggingface.co/new-space
2. Select **Gradio** SDK
3. Push this repository
4. The model trains automatically on first launch (~60 seconds)

---

## License

MIT License — see [LICENSE](./LICENSE)

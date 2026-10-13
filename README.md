# Traffic Accident Severity Analytics and Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://traffic-accident-severity-analytics-and-prediction-3yzwgf7tuhc.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

> **Live Deployment Link**: [https://traffic-accident-severity-analytics-and-prediction-3yzwgf7tuhc.streamlit.app/](https://traffic-accident-severity-analytics-and-prediction-3yzwgf7tuhc.streamlit.app/)

An end-to-end Data Engineering and MLOps system that combines **UK DfT Road Safety (STATS19)** collision records with historical hourly weather from the **Open-Meteo API** to build an analytical data warehouse, interactive dashboards, and a machine learning service predicting accident severity.

---

## Key Highlights & System Architecture

### Part 1: Data Engineering & Analytics Warehouse
- **Real public data**: 513,801 UK DfT Road Safety (STATS19) collisions for 2021–2025, downloaded automatically into `datasets/` on first run.
- **Weather integration**: hourly Open-Meteo history fetched per weather grid cell and joined to every accident on (grid cell, local hour).
- **Raw landing zone & audit trail**: immutable batch files in `data/raw/`; every run is logged in `ingestion_audit` (SUCCESS / PARTIAL / FAILED with errors).
- **Data quality gate**: five rules (coordinates, DD/MM/YYYY date and time, speed limit, severity, duplicate key). Failing rows go to `quarantine_rejected_records` with the full payload; nothing is silently fixed.
- **Standardisation**: explicit date parsing and decoding of DfT numeric codes into labels (`src/etl/standardize.py`).
- **Star schema & marts** in **PostgreSQL / PostGIS** (SQLite fallback): `dim_date`, `dim_location` (+ PostGIS point), `dim_road`, `dim_weather`, `dim_severity`, `fact_accidents`, three marts rebuilt every run, and a model-ready view.
- **Idempotent, incremental loads**: re-runs only pick up unseen collisions, and every load is insert-or-ignore on natural keys.
- **Apache Airflow DAG** (`dags/accident_pipeline_dag.py`): ingest → weather → validate → transform → marts → monitoring → conditional retraining.
- **Streamlit dashboard**: KPIs, hour × weekday heatmap, weather/road impact, PyDeck hotspot map, vehicle/casualty analysis, live prediction, audit & drift views.

### Part 2: MLOps Pipeline Extension
- **Problem**: multi-class severity prediction (Fatal / Serious / Slight).
- **Chronological 70/15/15 split** to avoid temporal leakage.
- **Class imbalance**: SMOTE on the training split plus class weights.
- **Benchmarking**: Random Forest, LightGBM and XGBoost tracked in **MLflow** (params, macro-F1, per-class recall, confusion matrices).
- **Model Registry**: the champion pipeline is registered as `accident-severity-classifier` with a `champion` alias; `models/model_metadata.json` records the version and baseline metrics.
- **Serving**: FastAPI (`/health`, `/predict`, `/predict/batch`, `/metrics`) with prediction logging to `prediction_log` and failure / per-class counters.
- **Containerization**: Docker Compose with PostGIS, MLflow, Airflow, FastAPI and Streamlit.
- **Monitoring & retraining**: PSI / KS feature drift, location drift, chi-square class drift, champion backtest (macro-F1, Fatal/Serious false-negative rates) and live FNR from logged predictions. Any trigger makes the DAG retrain and register a new version.

---

## Repository Structure

```text
traffic-accident-mlops/
├── dags/                          # Apache Airflow DAGs
│   └── accident_pipeline_dag.py
├── src/
│   ├── ingestion/                 # Raw ingestion scripts
│   │   ├── accident_ingest.py
│   │   └── weather_ingest.py
│   ├── etl/                       # Validation, transformation & warehouse loader
│   │   ├── db.py
│   │   ├── standardize.py         # DfT code decoding, date parsing, weather join keys
│   │   ├── validation.py
│   │   └── transform.py
│   ├── ml/                        # Feature engineering, training & evaluation
│   │   ├── features.py
│   │   └── train.py
│   ├── api/                       # FastAPI serving
│   │   ├── main.py
│   │   └── schemas.py
│   └── monitoring/                # Drift detection & retraining engine
│       └── drift_detector.py
├── streamlit_app/                 # Streamlit BI & prediction dashboard
│   ├── app.py
│   └── pages/
│       ├── 1_Temporal_Dynamics.py
│       ├── 2_Environmental_Risk.py
│       ├── 3_Geospatial_Hotspots.py
│       ├── 4_Vehicle_Casualty_DeepDive.py
│       ├── 5_MLOps_Live_Prediction.py
│       └── 6_Pipeline_Audit_Drift.py
├── sql/                           # DDL schemas & analytical marts
│   ├── 00_init.sql                # Creates airflow_db (Docker init)
│   ├── 01_staging_and_quarantine.sql
│   ├── 02_star_schema.sql
│   ├── 03_analytical_marts.sql
│   └── 04_postgis.sql             # PostGIS geometry (PostgreSQL only)
├── docs/                          # Project documentation
│   ├── dataset_source_guide.md
│   ├── architecture_and_pipeline_flow.md
│   ├── data_dictionary.md
│   └── execution_evidence.md
├── reports/                       # Academic project report & drift outputs
│   └── PROJECT_REPORT_PART1_PART2.md
├── tests/                         # Integration test suite
│   ├── conftest.py                # Temporary warehouse + DfT-format fixture
│   └── test_pipeline.py
├── models/                        # Serialized inference model & artifacts
│   ├── accident_severity_model.joblib
│   ├── model_metadata.json        # Registered MLflow version & baseline metrics
│   └── test_evaluation_report.json
├── datasets/                      # DfT source CSV (downloaded, gitignored)
├── data/                          # Generated at runtime (gitignored)
│   ├── raw/                       # Immutable raw landing zone
│   ├── rejected/                  # Quarantine summaries
│   └── traffic_accidents.db       # Local SQLite warehouse (PostgreSQL fallback)
├── docker-compose.yml             # Multi-service container orchestration
├── Dockerfile.airflow             # Airflow image with pipeline dependencies
├── Dockerfile.api                 # Container build for FastAPI
├── Dockerfile.streamlit           # Container build for Streamlit
├── requirements.txt               # Python package dependencies
├── .env.example                   # Environment configuration template
└── README.md
```

---

## Quickstart & Execution Guide

### 1. Local Python Environment (Standalone Mode)

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Optional: copy the environment template. Leave DATABASE_URL empty to use SQLite directly.
cp .env.example .env

# 3. Ingest DfT collisions (downloads ~100 MB into datasets/ on first run; only unseen rows are loaded)
python src/ingestion/accident_ingest.py

# 4. Fetch Open-Meteo weather for the staged accidents (run again to resume after a rate limit)
python src/ingestion/weather_ingest.py

# 5. Validate, quarantine, load the star schema and rebuild the marts
python src/etl/transform.py

# 6. Train, track in MLflow and register the champion model
python src/ml/train.py

# 7. Run drift, class-shift and false-negative-rate monitoring
python src/monitoring/drift_detector.py

# 8. Run the test suite (uses a temporary database, never your warehouse)
pytest tests/ -v

# 9. Launch the dashboard and the API
streamlit run streamlit_app/app.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Browse local MLflow runs and the model registry
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Useful settings (see `.env.example`): `ACCIDENT_BATCH_SIZE` loads the next N unseen collisions per run, `WEATHER_GRID_DEG` sets the weather grid (default 1.0°), and `python src/etl/db.py --reset` drops and recreates every project table.

> **Upgrading from the earlier synthetic-data version:** the table layout changed, so rebuild the warehouse once with `python src/etl/db.py --reset`, or move the old `data/traffic_accidents.db` aside.

### 2. Multi-Container Docker Compose Mode
With Docker Desktop running:

```bash
docker compose up --build
```

- **Streamlit Analytics Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking Server & Registry**: [http://localhost:5000](http://localhost:5000)
- **Airflow Webserver** (admin / admin): [http://localhost:8080](http://localhost:8080)

The Airflow services use a custom image (`Dockerfile.airflow`) with the pipeline's packages pre-installed, mount `datasets/`, `data/`, `models/` and `reports/`, write to the PostGIS warehouse and log to the MLflow server. Unpause `accident_severity_pipeline` in the Airflow UI and trigger it; only one run executes at a time.

---

## Project Deliverables & Artifact Checklist

| Item | Required Artifact | File Location in Repository |
| :---: | :--- | :--- |
| **1** | Source code and ETL scripts | `src/ingestion/`, `src/etl/` |
| **2** | Airflow DAG orchestration workflow | `dags/accident_pipeline_dag.py` |
| **3** | Database schema and populated tables | `sql/`; warehouse built by the pipeline (`data/traffic_accidents.db` or PostGIS) |
| **4** | Dataset source information & access instructions | `docs/dataset_source_guide.md` |
| **5** | Architecture diagram and pipeline flow | `docs/architecture_and_pipeline_flow.md` |
| **6** | Data dictionary and validation rules | `docs/data_dictionary.md` |
| **7** | Streamlit application & cloud deployment | `streamlit_app/` |
| **8** | Project report (Part 1 & Part 2) | `reports/PROJECT_REPORT_PART1_PART2.md` |
| **9** | Execution evidence & logs | `docs/execution_evidence.md` |
| **10** | Setup and run documentation | `README.md` |
| **Part 2** | Model pipeline, MLflow tracking & registry | `src/ml/`, `models/` |
| **Part 2** | FastAPI microservice & Docker containerization | `src/api/`, `Dockerfile.api`, `docker-compose.yml` |
| **Part 2** | Monitoring, retraining triggers & lifecycle | `src/monitoring/drift_detector.py`, `reports/` |

---
*Developed for the Data Engineering and MLOps Capstone Project.*

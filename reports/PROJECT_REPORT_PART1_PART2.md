# Traffic Accident Severity Analytics and Prediction
## Project Report – Part 1 (Data Pipeline) and Part 2 (MLOps Extension)

**Course**: Data Engineering and MLOps · **Mode**: Individual Project · **Total Marks**: 100 (Part 1: 50 | Part 2: 50)
**Data**: UK DfT Road Safety Data (STATS19) collisions 2021–2025, enriched with Open-Meteo hourly weather
**Results in this report**: end-to-end run of 13 Oct 2026 (see `docs/execution_evidence.md`)

---

## Table of Contents
1. Problem Understanding and Objectives
2. Data Sources and Access
3. Architecture
4. Ingestion and Raw Landing Zone
5. Data Quality, Validation and Quarantine
6. Standardisation, Weather Join and Star Schema
7. Orchestration with Apache Airflow
8. Dashboard and Analytical Findings
9. Feature Engineering, Split and Class Imbalance
10. Model Benchmarking and Evaluation
11. MLflow Tracking, Registry and Packaging
12. Deployment: FastAPI and Docker
13. Monitoring: Drift, Class Shift and False-Negative Rates
14. Retraining Criteria and Model Lifecycle
15. Limitations and Next Steps

---

## 1. Problem Understanding and Objectives

Road collisions in Great Britain are recorded by police in the STATS19 system and graded **Fatal**, **Serious** or **Slight**. Emergency services, highway authorities and councils want to know *where and when* severe collisions happen and *which conditions* make a collision more likely to be severe. Two products answer that:

- **Part 1 – analytics warehouse and dashboard**: integrate collision records with weather, model them in a location- and time-aware star schema, and visualise severity patterns.
- **Part 2 – severity prediction service**: train a classifier on the warehouse, track and register it with MLflow, serve it through FastAPI in Docker, and monitor it for drift and for missed severe collisions.

The prediction task is hard by nature: only **1.47%** of collisions are fatal and **22.7%** serious, and STATS19 does not record the dominant causes of severity (impact speed, seat-belt use, vulnerable road users per collision). A useful model therefore has to be judged on its ability to *find* severe collisions (recall, false-negative rate) rather than on accuracy.

---

## 2. Data Sources and Access

| Source | Content | Licence | Access |
| :--- | :--- | :--- | :--- |
| UK DfT Road Safety Data – collisions, last 5 years | 513,801 collisions, 2021–2025, 44 coded columns | Open Government Licence v3.0 | Auto-downloaded from `data.dft.gov.uk` into `datasets/` |
| Open-Meteo Historical Weather API | Hourly temperature, precipitation, wind, visibility, WMO weather code | CC BY 4.0, no key | REST, `archive-api.open-meteo.com/v1/archive` |

Full access instructions, column mapping and licence notes are in `docs/dataset_source_guide.md`. DfT categorical fields are numeric codes (decoded with the DfT *Road Safety Open Data Guide*), dates are `DD/MM/YYYY`, and `-1` means missing. Both facts drive design decisions in sections 5 and 6.

---

## 3. Architecture

```text
DfT CSV ─┐                                        ┌─ Streamlit dashboard (7 pages)
         ├─► Ingestion ─► Raw landing (data/raw) ─┤
Open-Meteo┘      │            │                    │
                 ▼            ▼                    │
          ingestion_audit   Staging (stg_*) ─► Quality gate ─► quarantine_rejected_records
                                      │
                                      ▼
            Transform: parse dates, decode codes, weather join (grid cell + hour)
                                      │
                                      ▼
       PostgreSQL/PostGIS star schema (dim_date, dim_location, dim_road, dim_weather,
       dim_severity, fact_accidents) ─► marts + v_model_training_dataset
                                      │
                                      ▼
   Feature pipeline ─► RF / LightGBM / XGBoost ─► MLflow tracking + Model Registry (champion)
                                      │
                                      ▼
            FastAPI (Docker) ─► prediction_log ─► Monitoring ─► retraining trigger ─► Airflow retrain
```

Layers: **source** (DfT, Open-Meteo) → **ingestion** (Python/Pandas, requests) → **raw/staging** (immutable files, `stg_*` tables) → **transformation** (validation, standardisation, weather join) → **storage** (PostgreSQL + PostGIS, SQLite fallback for local runs) → **analytics** (marts, Streamlit) → **MLOps** (MLflow, FastAPI, Docker, monitoring). Mermaid versions of the architecture, DAG sequence and inference flow are in `docs/architecture_and_pipeline_flow.md`.

---

## 4. Ingestion and Raw Landing Zone

**Accidents** (`src/ingestion/accident_ingest.py`). The DfT CSV is read by its own header, so the column order and naming generation (`accident_*` or `collision_*`) do not matter and no data row is lost. Before loading, the script removes collisions that were already loaded, staged or quarantined, so a scheduled re-run only takes **new** records. `ACCIDENT_BATCH_SIZE` limits each run to the next *N* collisions in date order, which simulates data arriving over time. Each batch is written unchanged to `data/raw/accidents_raw_<batch>.csv`, appended to `stg_accidents_raw`, and recorded in `ingestion_audit` (source, file, status, row counts, timestamps). A failed run records `FAILED` with the exception text.

**Weather** (`src/ingestion/weather_ingest.py`). For every staged accident the location is snapped to a weather grid cell (`WEATHER_GRID_DEG`, default 1.0°) and the time to its local hour. The script requests each grid cell once per calendar year with ISO dates and `timezone=Europe/London`, so API hours line up with police-recorded local times. The full API responses are stored in `data/raw/weather_raw_<batch>.jsonl.gz`, and only the hours that match an accident are loaded into `stg_weather_raw`. Failed or rate-limited requests are logged (audit status `PARTIAL`/`FAILED`) and are **never replaced by invented values**. The next run fetches only what is still missing.

*Grid choice.* The free Open-Meteo tier counts a year-long request as about 26 calls with a 10,000/day limit. A full backfill needs about 7,600 weighted calls at 1.0° (62 cells) against about 24,000 at 0.5° (196 cells), so 1.0° lets the whole 5-year history load in one day. The run loaded **328,038** accident-hours from 293 requests with **0 failures**.

---

## 5. Data Quality, Validation and Quarantine

`src/etl/validation.py` applies five rules per batch, and a record is quarantined with the first rule it fails:

| Rule | Check | Rejected in run |
| :--- | :--- | ---: |
| VR-01 Coordinates | present and inside the UK box (49–61.5 N, −8.5–2.5 E) | 53 |
| VR-03 Date/time | date parses as `DD/MM/YYYY` (or ISO), time as `HH:MM` | 0 |
| VR-02 Speed limit | present and 10–100 mph (DfT `-1` = missing) | 3 |
| VR-04 Severity | in {1, 2, 3} | 0 |
| VR-05 Unique key | present, not repeated in batch, not already loaded | 0 |

Rejected rows go to `quarantine_rejected_records` with the reason and the full JSON payload, the batch's audit row is updated with the rejected count, and a summary JSON is written to `data/rejected/`. The analytical dataset is never edited by hand and **no value is silently imputed**. An earlier version replaced unparseable dates with a fixed timestamp, which is exactly the silent rewrite this design avoids. When the DAG has already run validation as its own task, the transform re-uses the split without writing the rejections a second time.

---

## 6. Standardisation, Weather Join and Star Schema

**Standardisation** (`src/etl/standardize.py`):
- Dates are parsed with an explicit day-first format, because pandas' default guess reads `02/10/2023` as 10 February. Dates are stored as ISO `YYYY-MM-DD`.
- Road type, light, weather, road surface and urban/rural codes are decoded to DfT labels (for example `6` → `Single carriageway`, `7` → `Fog or mist`).
- The ONS district code (`E09000033`) is used for local authority, and the nation (England/Wales/Scotland) is derived from its prefix.

**Weather join**: accidents are joined to `stg_weather_raw` on (grid latitude, grid longitude, local hour). Result: **513,745 of 513,745** loaded accidents (100%) have measured weather. The ERA5-based archive returns no visibility values, so `visibility_m` is null and is handled by the model's imputer.

**Star schema** (`sql/02_star_schema.sql`), grain one row per collision:

| Table | Key / grain | Rows |
| :--- | :--- | ---: |
| `fact_accidents` | one collision; vehicles, casualties | 513,745 |
| `dim_date` | hour (`YYYYMMDDHH`); weekday, weekend, time-of-day bucket | 42,170 |
| `dim_location` | lat/lon + urban/rural + ONS district; PostGIS point (PostgreSQL) | 512,323 |
| `dim_road` | road type × speed × light × surface | 782 |
| `dim_weather` | Open-Meteo observation (cell, hour) × reported condition | 385,740 |
| `dim_severity` | Fatal / Serious / Slight | 3 |

**Marts** (`sql/03_analytical_marts.sql`) are dropped and rebuilt on every run: `agg_hourly_severity_mart` (day × hour × severity), `agg_weather_road_risk_mart` (weather × road × speed × severity), `agg_location_hotspots_mart` (0.001° bins with fatal/serious/slight counts), and the view `v_model_training_dataset`.

**Idempotency**: every dimension and fact load is a set-based insert-or-ignore on the natural key (`INSERT OR IGNORE` on SQLite, `ON CONFLICT DO NOTHING` on PostgreSQL), and staging is cleared per processed batch. Re-running the pipeline leaves all tables unchanged (tested). On PostgreSQL, `sql/04_postgis.sql` enables PostGIS and maintains a GIST-indexed `geometry(Point, 4326)` on `dim_location`.

---

## 7. Orchestration with Apache Airflow

`dags/accident_pipeline_dag.py` (`accident_severity_pipeline`, `@daily`, `catchup=False`, 2 retries, 2-minute delay):

```text
check_db_readiness → ingest_raw_accidents → ingest_weather_api → validate_and_quarantine
   → transform_star_schema → refresh_data_marts → drift_and_retrain_check
   → retraining_required (ShortCircuit) → retrain_and_register_model
```

Batch IDs travel between tasks through XCom, and the downstream tasks process everything still waiting in staging, so rows left by a failed run are picked up by the next one. `max_active_runs=1` stops overlapping runs. In the Docker verification run all nine tasks succeeded on PostGIS: the class-shift trigger fired, and the DAG retrained and registered version 2 (XGBoost, test macro-F1 0.373, ROC-AUC 0.642, Fatal recall 0.242) and moved the `champion` alias to it. Validation runs once as its own visible quality-gate task. The last two tasks implement automatic retraining: the monitoring task pushes its decision to XCom, and the short-circuit skips retraining unless a trigger fired. In Docker Compose, the Airflow services write to the PostGIS warehouse (`DATABASE_URL`), log to the MLflow server, and mount `datasets/`, `data/`, `models/` and `reports/`.

---

## 8. Dashboard and Analytical Findings

The Streamlit app (`streamlit_app/`) has a KPI home page plus six pages: Temporal Dynamics (hour × weekday heatmap, time-of-day, weekday vs weekend), Environmental Risk (weather, surface, speed limit, light), Geospatial Hotspots (PyDeck hexagon and point layers, top local authorities), Vehicle & Casualty analysis, Live Prediction, and Pipeline Audit & Drift. All seven pages were executed headlessly against the populated warehouse without errors.

Findings from the 513,745 collisions:

| Factor | Observation |
| :--- | :--- |
| Time | Volume peaks at 15:00–17:00 (17:00 = 44,785 collisions), but the fatal share is highest at **Night (2.61%)**, about twice the daytime rate (1.13–1.30%). |
| Day | Weekends are more deadly per collision: Sunday 1.95%, Saturday 1.75%, weekdays 1.25–1.44%. |
| Light | Fatal share 4.82% on dark roads with **no lighting**, 1.47% on lit dark roads, 1.21% in daylight. |
| Speed limit | Fatal share climbs from 0.47% at 20 mph to 3.84% at 60 mph; serious-or-fatal share from 19.7% to 34.5%. |
| Area | Rural collisions are 3.5× more likely to be fatal (2.83%) than urban (0.80%). |
| Weather | Most collisions occur in fine weather (80%); the highest fatal shares are in fog (2.82%) and high winds (2.36–2.62%). |
| Geography | England 477,443, Scotland 20,530, Wales 15,772; the busiest district by volume is E08000025 (Birmingham). |

These patterns support the modelling choices: speed limit, time of day, lighting and rural setting carry most of the severity signal, while day-to-day weather adds a smaller amount.

---

## 9. Feature Engineering, Split and Class Imbalance

`src/ml/features.py` loads the star-schema join and builds:
- **Numeric** (median imputation, standard scaling): hour, speed limit, temperature, precipitation, visibility, wind speed, vehicles, casualties, and cyclical `hour_sin` / `hour_cos`.
- **Categorical** (one-hot, `handle_unknown="ignore"`): day name, time of day, urban/rural, road type, light, surface, reported weather.

**Split**: chronological 70 / 15 / 15 by date and hour, giving train 359,621 (2021 → mid-2024), validation 77,062 and test 77,062 (latest months). Preprocessing and SMOTE are fit on the training split only.

**Imbalance**: SMOTE oversamples the training split (359,621 → 825,330 rows). Random Forest and LightGBM also use balanced class weights `w_j = N / (K·n_j)` = Slight 0.44, Serious 1.51, Fatal 22.6, and XGBoost uses SMOTE alone.

---

## 10. Model Benchmarking and Evaluation

Validation results (selection metric: macro-F1):

| Model | Accuracy | Macro-F1 | Fatal recall | ROC-AUC (OvR) |
| :--- | ---: | ---: | ---: | ---: |
| Random Forest (150 trees, depth 12, min leaf 20, 30% bootstrap) | 0.597 | 0.365 | **0.467** | 0.639 |
| **LightGBM** (180 trees, depth 7, lr 0.04) – champion | **0.701** | **0.372** | 0.156 | 0.648 |
| XGBoost (160 trees, depth 6, lr 0.05) | 0.683 | 0.371 | 0.186 | 0.648 |

**Held-out test (champion LightGBM, 77,062 most recent collisions)**:

| Class | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| Slight | 0.757 | 0.867 | 0.808 | 56,517 |
| Serious | 0.347 | 0.152 | 0.212 | 19,415 |
| Fatal | 0.056 | 0.188 | 0.087 | 1,130 |
| **Macro avg** | 0.387 | 0.403 | **0.369** | |

Accuracy 0.677, ROC-AUC 0.639, Fatal FNR 0.812, Serious FNR 0.848. The test confusion matrix is logged as `models/cm_test_champion.png`.

**Interpretation.** The model clearly learns real signal: ROC-AUC 0.64 is well above chance, and it finds 18.8% of fatal collisions while predicting Fatal for only a small share of cases (precision 5.6%, about 4× the 1.47% base rate). Its most-used features are speed limit, precipitation, hour (sin/cos), vehicles and casualties, and single-carriageway roads, which matches the analysis in section 8. Severity is still only partly predictable from STATS19 collision-level fields, so recall on the minority classes stays low.

**Trade-off.** On validation, Random Forest caught 3× more fatal collisions than LightGBM (46.7% vs 15.6% recall) at lower overall accuracy. Selecting by macro-F1 favours LightGBM. For an emergency-dispatch use case that prioritises missed fatalities, Random Forest or a lowered Fatal decision threshold would be the better operating point. This is a deployment decision recorded here rather than hidden in the metric.

---

## 11. MLflow Tracking, Registry and Packaging

- **Tracking** (`src/ml/train.py`): every candidate run logs parameters, accuracy, macro/weighted F1, Fatal and Serious recall, ROC-AUC and a validation confusion matrix. The champion run adds test metrics, the Fatal/Serious FNR and the test confusion matrix. When no server is running, MLflow uses a local SQLite backend (`mlflow.db`, artifacts in `mlruns/`); in Docker it uses the MLflow server.
- **Registry**: the champion is registered as **`accident-severity-classifier` version 2** with the alias **`champion`** (version 1 was the same LightGBM model saved with scikit-learn 1.6; version 2 re-saves it with 1.7.2 so it loads on Python 3.10–3.14, with identical metrics). Each retrain creates a new version and moves the alias, which gives a versioned history with rollback.
- **Packaging**: preprocessing and classifier form one scikit-learn `Pipeline`, saved compressed as `models/accident_severity_model.joblib` (0.8 MB). `models/model_metadata.json` records the registered version, run ID, features and the **baseline test metrics** that monitoring compares against.

---

## 12. Deployment: FastAPI and Docker

**API** (`src/api/main.py`):
- `GET /health`: model loaded, model type, registered version, uptime.
- `POST /predict` and `POST /predict/batch`: Pydantic validation (hour 0–23, UK coordinates, speed 10–100, DfT labels). Returns the predicted class, probabilities, a risk score (0.1·P(slight) + 0.5·P(serious) + 1.0·P(fatal)) and the model version. Every prediction is written to `prediction_log` with an optional `accident_index`.
- `GET /metrics`: Prometheus text with successful requests, **failed requests**, prediction-log failures, predictions per class, average latency and the served model version.

**Containers** (`docker-compose.yml`): `postgis/postgis:15-3.4` (warehouse and Airflow metadata DB, initialised by `sql/00_init.sql` … `04_postgis.sql`), `mlflow` v3.4 (SQLite backend, served artifacts), `airflow-init` / `webserver` / `scheduler` (LocalExecutor, custom image from `Dockerfile.airflow` with the pipeline's ML packages and the OpenMP runtime baked in; one active DAG run at a time), `fastapi` and `streamlit` (built from `python:3.10-slim`). Credentials come from environment variables (`.env.example`) and are never copied into images.

---

## 13. Monitoring: Drift, Class Shift and False-Negative Rates

`src/monitoring/drift_detector.py` compares the oldest 70% of the warehouse (the training period) with the newest 30%, and writes `reports/drift_report_<date>.json`:

| Check | Method | Result (13 Oct 2026 run) |
| :--- | :--- | :--- |
| Feature drift | PSI (10 quantile bins) + KS on temperature, precipitation, wind, speed | All PSI < 0.03: stable |
| Location drift | KS on latitude/longitude | KS 0.017 (p < 0.01 because n is large, but below the 0.15 practical threshold): stable |
| Class drift | chi-square on the severity mix | Serious 22.0% → **24.4%**, p < 0.01: **drift** |
| Model performance | champion scored on the current window | macro-F1 0.370; Fatal FNR 0.828; Serious FNR 0.850 (within tolerance) |
| Live FNR | `prediction_log` joined to recorded outcomes by `accident_index` | activates once ≥ 30 labelled predictions exist |
| Latency & failures | API `/metrics` counters | exposed for Prometheus scraping |
| Input quality | quarantine counts per batch in `ingestion_audit` | 56 of 513,801 rejected (0.01%) |

The class shift is a real finding. The share of collisions recorded as Serious rose in the most recent 18 months. This is consistent with more police forces moving to injury-based severity reporting (CRASH), which is why DfT also publishes adjusted-severity columns; using those adjusted columns as the label is a possible refinement. A model trained on the older mix will under-call Serious collisions, so retraining was triggered.

---

## 14. Retraining Criteria and Model Lifecycle

Retraining triggers, all implemented in code:
1. **Covariate shift**: PSI > 0.25 on temperature, precipitation, wind or speed limit.
2. **Location shift**: KS > 0.15 with p < 0.01 on latitude or longitude.
3. **Class shift**: severity mix changes by more than 2 percentage points with chi-square p < 0.01.
4. **Performance drop**: current-window macro-F1 more than 10% below the registered baseline.
5. **Safety**: Fatal or Serious false-negative rate more than 0.10 above baseline, on the backtest or on labelled live predictions.

**Lifecycle**: the daily DAG ingests new collisions → validates and loads → refreshes marts → runs monitoring. If any trigger fires, `retrain_and_register_model` trains on the full warehouse with a fresh chronological split, logs to MLflow, registers a new version and moves the `champion` alias. Serving uses the artifact and metadata of the current champion, and rollback means pointing the alias at the previous version and redeploying its artifact.

---

## 15. Limitations and Next Steps

- **Predictive ceiling**: collision-level STATS19 fields explain only part of severity. Joining the DfT vehicle and casualty tables (vehicle type, pedestrian involvement, age) is the most promising next feature set.
- **Operating point**: tune the Fatal/Serious decision thresholds (or cost-weighted selection) to the users' tolerance for missed severe collisions.
- **Leakage check**: `number_of_casualties` is only known after a collision. It is kept for analytical scoring and should be dropped for a pre-incident risk use case.
- **Weather resolution**: a 1.0° grid fits the free API tier for a full backfill. Daily incremental batches allow a finer 0.25–0.5° grid, and visibility is unavailable from the archive API.
- **Deployment hardening**: add a promotion gate that compares challenger and champion on the same time window before moving the `champion` alias.

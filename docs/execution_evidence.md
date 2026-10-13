# Execution Evidence & Pipeline Demonstration

End-to-end run on the real UK DfT collision file (`dft-road-casualty-statistics-collision-last-5-years.csv`, 513,801 collisions, 2021–2025) on 13 Oct 2026, local mode (SQLite warehouse, local MLflow with SQLite backend). Log lines are copied from the run; timestamps are local time.

> Screenshots of the Airflow DAG graph, MLflow runs/registry, FastAPI `/docs` and each dashboard page should be added to `docs/screenshots/` after running `docker compose up --build`.

---

## 1. Accident Ingestion (DfT CSV → raw landing zone → staging)
```text
accident_ingest - INFO - 513,801 collisions in source file; 513,801 new collisions selected for batch_acc_38611701.
accident_ingest - INFO - Saved raw landing file to: data\raw\accidents_raw_batch_acc_38611701.csv
accident_ingest - INFO - Ingestion batch batch_acc_38611701 logged to audit (513,801 records).
```
Every data row in the file was staged (the header is read by name, no row is skipped). Wall time: 29 s.

---

## 2. Weather Ingestion (Open-Meteo Historical API)
```text
weather_ingest - INFO - 328,038 accident hours need weather across 62 grid cells (1.0 deg); 293 API requests planned.
weather_ingest - INFO - Weather batch batch_wea_1ee69c50: SUCCESS, 328,038 hourly readings staged, 0 failed requests.
```
293 requests (one per grid cell and year), full responses kept in `data/raw/weather_raw_batch_wea_1ee69c50.jsonl.gz`. Wall time: about 7 minutes.

---

## 3. Validation, Quarantine & Star Schema Load
```text
data_validation - INFO - Starting quality checks on 513,801 records...
data_validation - WARNING - Quarantined 56 invalid records out of 513,801. Logged to quarantine_rejected_records.
etl_transform - INFO - Transforming 513,745 validated accident records...
etl_transform - INFO - Weather matched for 513,745 of 513,745 accidents (100.0%).
etl_transform - INFO - Loaded 513,745 new facts (fact_accidents now holds 513,745 rows).
etl_transform - INFO - Staging cleared for batch(es): batch_acc_38611701
etl_transform - INFO - ETL transformation, star schema load, and analytical marts refresh complete!
```
Wall time: 45 s.

| Quarantine reason | Rows |
| :--- | ---: |
| `INVALID_COORDINATES` (blank or outside UK bounds) | 53 |
| `INVALID_SPEED_LIMIT` (DfT `-1` = missing) | 3 |

### Populated tables after the run
| Table | Rows |
| :--- | ---: |
| `fact_accidents` | 513,745 |
| `dim_date` (hour grain) | 42,170 |
| `dim_location` | 512,323 |
| `dim_road` | 782 |
| `dim_weather` | 385,740 |
| `dim_severity` | 3 |
| `agg_hourly_severity_mart` | 504 |
| `agg_weather_road_risk_mart` | 3,365 |
| `agg_location_hotspots_mart` | 318,017 |
| `stg_weather_raw` | 328,038 |
| `stg_accidents_raw` (cleared after load) | 0 |
| `quarantine_rejected_records` | 56 |

### `ingestion_audit`
| Source | Status | Ingested | Rejected |
| :--- | :--- | ---: | ---: |
| UK DfT Road Safety Data (STATS19 collisions) | SUCCESS | 513,801 | 56 |
| Open-Meteo Historical Weather API | SUCCESS | 328,038 | 0 |

Re-running ingestion and the ETL afterwards selects 0 new collisions and leaves every table unchanged (covered by `test_pipeline_rerun_is_idempotent`).

---

## 4. Model Training, MLflow Tracking & Registry
```text
feature_engineering - INFO - Chronological split - Train: 359621, Val: 77062, Test: 77062
model_training - INFO - Applied SMOTE: Resampled training set from 359621 to 825330 records.
model_training - INFO - Calculated class weights: {0: 0.436, 1: 1.513, 2: 22.622}
model_training - INFO - RandomForest Results - Overall Accuracy: 0.5971, Macro F1: 0.3650, Fatal Recall: 0.4675, ROC-AUC: 0.6392
model_training - INFO - LightGBM Results - Overall Accuracy: 0.7010, Macro F1: 0.3717, Fatal Recall: 0.1558, ROC-AUC: 0.6479
model_training - INFO - XGBoost Results - Overall Accuracy: 0.6832, Macro F1: 0.3714, Fatal Recall: 0.1861, ROC-AUC: 0.6480
model_training - INFO - === Best Model Selected: LightGBM with Val Macro-F1: 0.3717 ===
Successfully registered model 'accident-severity-classifier'.
model_training - INFO - Registered accident-severity-classifier version 1 with alias 'champion'.
model_training - INFO - Test results - Macro F1: 0.3689, Fatal recall: 0.1885, Serious recall: 0.1522
```
Artifacts: `models/accident_severity_model.joblib` (0.8 MB), `models/model_metadata.json`, `models/test_evaluation_report.json`, validation confusion matrices `models/cm_*.png`, test confusion matrix `models/cm_test_champion.png`. Wall time: about 70 s.

---

## 5. Monitoring & Retraining Decision
```text
drift_monitoring - INFO - Analyzing drift - Reference: 359621 records, Current: 154124 records
drift_monitoring - INFO - Drift evaluation report written to: reports\drift_report_2026-10-13.json
drift_monitoring - INFO - Retraining Trigger Status: TRIGGERED
```
Reference window 2021-01-01 to 2024-06-27, current window 2024-06-27 to 2025-12-31. Feature PSI all below 0.03 (stable). Location KS 0.017 (statistically significant on 500k rows but far below the 0.15 practical threshold). Severity mix: Serious 22.0% → 24.4% (chi-square p < 0.01), which fired the retraining trigger. Champion backtest on the current window: macro-F1 0.370, Fatal FNR 0.828, Serious FNR 0.850 (within 0.10 of the test baseline).

---

## 6. Automated Tests
```text
14 passed
```
Tests run against a temporary SQLite warehouse and a DfT-format fixture (`tests/conftest.py`), covering day-first date parsing, code decoding, weather join keys, ingestion row counts, quarantine rules, idempotent re-runs, mart refresh, the DAG call signature, model prediction, `/health`, `/predict` with prediction logging, and `/metrics` counters.

## 7. Dashboard Smoke Test
All seven Streamlit pages (`app.py` and `pages/1`–`6`) were executed headlessly with `streamlit.testing.v1.AppTest` against the populated warehouse: no exceptions, live data (not demo mode).

---

## 8. Docker Compose & Airflow Verification (PostGIS warehouse)
Stack: `postgis/postgis:15-3.4`, `ghcr.io/mlflow/mlflow:v3.4.0`, custom `traffic-airflow:2.8.1` (Dockerfile.airflow), FastAPI and Streamlit images. All services healthy: Airflow `/health` healthy, FastAPI `/health` serving `accident-severity-classifier/v1`, Streamlit `/_stcore/health` ok, MLflow UI 200.

DAG `accident_severity_pipeline`, run `manual__2026-10-13T20:33:21`: all 9 tasks succeeded in 3 min 27 s.

| Task | State |
| :--- | :--- |
| check_db_readiness → ingest_raw_accidents → ingest_weather_api → validate_and_quarantine → transform_star_schema → refresh_data_marts → drift_and_retrain_check | success |
| retraining_required (class-mix shift detected) | success |
| retrain_and_register_model | success: XGBoost registered as `accident-severity-classifier` **v2**, alias `champion` moved from v1 |

PostGIS warehouse after the runs: 513,745 facts, 385,740 weather observations (100% from Open-Meteo), 512,323 locations with PostGIS geometry, marts rebuilt.

Retrained champion in Docker (same 359,621-row training split): XGBoost, test macro-F1 0.373, ROC-AUC 0.642, Fatal recall 0.242, Serious recall 0.182. LightGBM and XGBoost are within 0.001 macro-F1 on validation, so either can win a given run. The committed artifact in `models/` is the local LightGBM v1 described above.

Recovery behaviour observed: two earlier overlapping runs (before `max_active_runs=1`) had staged the file twice. The fixed pipeline processed everything left in staging, loaded each collision once, and quarantined the second copy as `DUPLICATE_ACCIDENT_INDEX`.

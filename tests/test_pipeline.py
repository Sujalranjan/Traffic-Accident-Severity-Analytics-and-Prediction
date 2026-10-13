"""
Automated Unit and Integration Tests for the Pipeline & FastAPI Microservice.
Runs against a temporary SQLite warehouse (see conftest.py).
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.etl import standardize as std
from src.etl.db import get_engine, init_db
from src.ingestion.accident_ingest import ingest_accidents
from src.etl.transform import run_etl_pipeline
from src.api.main import app, get_model
from conftest import GOOD_ROWS, BAD_ROWS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _count(sql: str) -> int:
    return int(pd.read_sql(sql, con=get_engine()).iloc[0, 0])


@pytest.fixture
def client():
    return TestClient(app)


# ---------- Standardisation ----------

def test_dft_dates_are_parsed_day_first():
    parsed = std.parse_accident_date(pd.Series(["02/10/2023", "22/02/2023", "2024-03-01", "31/02/2024", None]))
    assert parsed[0] == pd.Timestamp("2023-10-02")   # 2 October, not 10 February
    assert parsed[1] == pd.Timestamp("2023-02-22")
    assert parsed[2] == pd.Timestamp("2024-03-01")
    assert pd.isna(parsed[3]) and pd.isna(parsed[4])


def test_dft_codes_are_decoded():
    decoded = std.decode_column(pd.Series(["6", "3", "-1", "", "Roundabout"]), std.ROAD_TYPE)
    assert decoded.tolist() == ["Single carriageway", "Dual carriageway", std.MISSING_LABEL, std.MISSING_LABEL, "Roundabout"]
    assert std.decode_column(pd.Series(["7"]), std.WEATHER_CONDITIONS)[0] == "Fog or mist"


def test_weather_join_keys():
    assert std.weather_grid(pd.Series([51.26, -0.24]), 0.5).tolist() == [51.5, -0.0]
    hour = std.observation_hour(pd.Series([pd.Timestamp("2023-10-02 08:15")]))
    assert hour[0] == "2023-10-02T08:00"


# ---------- Ingestion, validation and ETL ----------

def test_database_init():
    engine = get_engine()
    init_db(engine)
    assert _count("SELECT COUNT(*) FROM dim_severity") == 3


def test_ingest_reads_every_row_and_returns_batch(dft_csv):
    batch_id = ingest_accidents(source_file=dft_csv, download_if_missing=False)
    assert batch_id.startswith("batch_acc_")
    staged = _count(f"SELECT COUNT(*) FROM stg_accidents_raw WHERE batch_id = '{batch_id}'")
    assert staged == len(GOOD_ROWS) + len(BAD_ROWS)   # header handled, no data row dropped


def test_etl_loads_clean_rows_and_quarantines_bad_ones():
    result = run_etl_pipeline()
    assert result["loaded"] == len(GOOD_ROWS)
    assert _count("SELECT COUNT(*) FROM quarantine_rejected_records") == len(BAD_ROWS)
    assert _count("SELECT COUNT(*) FROM stg_accidents_raw") == 0

    reasons = pd.read_sql("SELECT rejection_reason FROM quarantine_rejected_records", con=get_engine())["rejection_reason"]
    assert reasons.str.startswith("INVALID_DATETIME").sum() == 2
    assert reasons.str.startswith("INVALID_SPEED_LIMIT").sum() == 1
    assert reasons.str.startswith("INVALID_COORDINATES").sum() == 1

    dates = pd.read_sql("SELECT d.full_date, d.hour FROM fact_accidents f JOIN dim_date d ON f.date_key = d.date_key "
                        "WHERE f.accident_index = '2023010000001'", con=get_engine())
    assert str(dates["full_date"][0])[:10] == "2023-10-02" and dates["hour"][0] == 8

    labels = pd.read_sql("SELECT DISTINCT road_type FROM dim_road", con=get_engine())["road_type"]
    assert "Single carriageway" in set(labels) and not labels.str.fullmatch(r"-?\d+").any()


def test_pipeline_rerun_is_idempotent(dft_csv):
    facts_before = _count("SELECT COUNT(*) FROM fact_accidents")
    quarantine_before = _count("SELECT COUNT(*) FROM quarantine_rejected_records")
    batch_id = ingest_accidents(source_file=dft_csv, download_if_missing=False)
    assert _count(f"SELECT COUNT(*) FROM stg_accidents_raw WHERE batch_id = '{batch_id}'") == 0
    run_etl_pipeline(batch_id=batch_id)
    run_etl_pipeline()
    assert _count("SELECT COUNT(*) FROM fact_accidents") == facts_before
    assert _count("SELECT COUNT(*) FROM quarantine_rejected_records") == quarantine_before


def test_marts_are_refreshed():
    assert _count("SELECT SUM(accident_count) FROM agg_hourly_severity_mart") == _count("SELECT COUNT(*) FROM fact_accidents")


def test_dag_calls_match_function_signatures():
    tree = ast.parse((PROJECT_ROOT / "dags" / "accident_pipeline_dag.py").read_text(encoding="utf-8"))
    calls = {n.func.id: n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls["ingest_accidents"].keywords and not calls["ingest_accidents"].args


# ---------- Model and API ----------

def test_model_pipeline_predict():
    model = get_model()
    assert model is not None
    sample_df = pd.DataFrame([{
        "hour": 17, "speed_limit": 30, "temperature_c": 12.0, "precipitation_mm": 1.5,
        "visibility_m": 8000.0, "wind_speed_kmh": 20.0, "number_of_vehicles": 2, "number_of_casualties": 1,
        "hour_sin": np.sin(2 * np.pi * 17 / 24.0), "hour_cos": np.cos(2 * np.pi * 17 / 24.0),
        "day_name": "Friday", "time_of_day": "Evening Rush", "urban_or_rural": "Urban",
        "road_type": "Single carriageway", "light_conditions": "Daylight",
        "road_surface_conditions": "Wet or damp", "weather_condition": "Raining no high winds",
    }])
    pred = model.predict(sample_df)
    proba = model.predict_proba(sample_df)
    assert pred[0] in [0, 1, 2]
    assert np.isclose(np.sum(proba[0]), 1.0, atol=1e-3)


def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_api_prediction_is_served_and_logged(client):
    payload = {
        "accident_index": "2023010000001",
        "hour": 18, "day_name": "Friday", "time_of_day": "Evening Rush",
        "latitude": 51.5074, "longitude": -0.1278, "urban_or_rural": "Urban",
        "road_type": "Single carriageway", "speed_limit": 30, "light_conditions": "Daylight",
        "road_surface_conditions": "Wet or damp", "weather_condition": "Raining no high winds",
        "temperature_c": 10.0, "precipitation_mm": 2.0, "visibility_m": 5000.0, "wind_speed_kmh": 22.0,
        "number_of_vehicles": 2, "number_of_casualties": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_severity_label"] in ["Fatal", "Serious", "Slight"]
    assert 0.0 <= data["severity_risk_score"] <= 1.0
    assert _count("SELECT COUNT(*) FROM prediction_log WHERE accident_index = '2023010000001'") >= 1


def test_metrics_expose_failures_and_class_counts(client):
    text = client.get("/metrics").text
    assert "model_request_failures_total" in text
    assert 'model_predictions_total{severity="Slight"}' in text


def test_revalidating_a_staged_batch_does_not_duplicate_quarantine(dft_csv, tmp_root):
    from src.etl.validation import validate_and_quarantine
    other = tmp_root / "dft_second_file.csv"
    other.write_text(dft_csv.read_text(encoding="utf-8").replace("2024010000006", "2024019999006"), encoding="utf-8")
    batch_id = ingest_accidents(source_file=other, download_if_missing=False)
    before = _count("SELECT COUNT(*) FROM quarantine_rejected_records")
    validate_and_quarantine(batch_id=batch_id)
    once = _count("SELECT COUNT(*) FROM quarantine_rejected_records")
    validate_and_quarantine(batch_id=batch_id)   # e.g. the next run after a failed transform
    assert once == before + 1
    assert _count("SELECT COUNT(*) FROM quarantine_rejected_records") == once
    run_etl_pipeline(batch_id=batch_id, persist_rejections=False)

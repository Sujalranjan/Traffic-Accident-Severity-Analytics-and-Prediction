"""
Apache Airflow DAG: Traffic Accident Severity Analytics & MLOps Pipeline.
Orchestrates end-to-end ingestion, quality quarantine, dimensional ETL,
mart materialization, monitoring, and conditional model retraining.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Airflow DAG definition
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def airflow_check_db():
    from src.etl.db import get_engine, init_db
    engine = get_engine()
    init_db(engine)
    print("Database connection and schema initialization verified.")


def airflow_ingest_accidents(**context):
    from src.ingestion.accident_ingest import ingest_accidents
    # Batch size comes from ACCIDENT_BATCH_SIZE (unset = all unseen collisions)
    batch_id = ingest_accidents()
    context["ti"].xcom_push(key="accident_batch_id", value=batch_id)
    print(f"Ingested accident batch: {batch_id}")


def airflow_ingest_weather(**context):
    from src.ingestion.weather_ingest import ingest_weather_for_accidents
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="accident_batch_id", task_ids="ingest_raw_accidents")
    weather_batch = ingest_weather_for_accidents(batch_id=batch_id)
    print(f"Enriched atmospheric weather batch: {weather_batch}")


def airflow_validate_and_quarantine(**context):
    from src.etl.validation import validate_and_quarantine
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="accident_batch_id", task_ids="ingest_raw_accidents")
    res = validate_and_quarantine(batch_id=batch_id)
    print(f"Validation complete: Total={res.get('total')}, Valid={res.get('valid')}, Quarantined={res.get('rejected')}")


def airflow_transform_star_schema(**context):
    from src.etl.transform import run_etl_pipeline
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="accident_batch_id", task_ids="ingest_raw_accidents")
    # Rejections were already persisted by the validate_and_quarantine task
    result = run_etl_pipeline(batch_id=batch_id, persist_rejections=False)
    print(f"Star schema load complete: {result}")


def airflow_refresh_data_marts():
    from src.etl.db import get_engine, run_sql_file
    run_sql_file(get_engine(), "03_analytical_marts.sql")
    print("Analytical data marts refreshed for BI dashboards.")


def airflow_drift_check(**context):
    from src.monitoring.drift_detector import run_drift_analysis
    report = run_drift_analysis()
    decision = report.get("retraining_decision", {})
    retrain = bool(decision.get("retrain_triggered", False))
    context["ti"].xcom_push(key="retrain_triggered", value=retrain)
    print(f"Drift check completed. Retraining triggered: {retrain}. Reasons: {decision.get('triggers')}")


def airflow_should_retrain(**context):
    return bool(context["ti"].xcom_pull(key="retrain_triggered", task_ids="drift_and_retrain_check"))


def airflow_retrain_model():
    from src.ml.train import train_and_evaluate_models
    result = train_and_evaluate_models()
    if result is None:
        raise RuntimeError("Retraining skipped: not enough data in the warehouse")
    champion, macro_f1 = result
    print(f"Retrained and registered new champion: {champion} (test macro-F1 {macro_f1:.4f})")


with DAG(
    dag_id="accident_severity_pipeline",
    default_args=default_args,
    description="Automated Traffic Accident Severity Analytics and MLOps Pipeline",
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "data-engineering", "accident-severity"]
) as dag:

    task_check_db = PythonOperator(
        task_id="check_db_readiness",
        python_callable=airflow_check_db
    )

    task_ingest_accidents = PythonOperator(
        task_id="ingest_raw_accidents",
        python_callable=airflow_ingest_accidents
    )

    task_ingest_weather = PythonOperator(
        task_id="ingest_weather_api",
        python_callable=airflow_ingest_weather
    )

    task_validate_quarantine = PythonOperator(
        task_id="validate_and_quarantine",
        python_callable=airflow_validate_and_quarantine
    )

    task_transform_star_schema = PythonOperator(
        task_id="transform_star_schema",
        python_callable=airflow_transform_star_schema
    )

    task_refresh_marts = PythonOperator(
        task_id="refresh_data_marts",
        python_callable=airflow_refresh_data_marts
    )

    task_drift_monitor = PythonOperator(
        task_id="drift_and_retrain_check",
        python_callable=airflow_drift_check
    )

    task_should_retrain = ShortCircuitOperator(
        task_id="retraining_required",
        python_callable=airflow_should_retrain
    )

    task_retrain = PythonOperator(
        task_id="retrain_and_register_model",
        python_callable=airflow_retrain_model
    )

    # Pipeline task dependencies
    (
        task_check_db
        >> task_ingest_accidents
        >> task_ingest_weather
        >> task_validate_quarantine
        >> task_transform_star_schema
        >> task_refresh_marts
        >> task_drift_monitor
        >> task_should_retrain
        >> task_retrain
    )

"""
FastAPI Model Microservice for Traffic Accident Severity Prediction.
Provides real-time inference, batch predictions, health checks, and monitoring metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import json
import time
import joblib
import datetime
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    AccidentFeatureInput, PredictionResult,
    BatchPredictionRequest, BatchPredictionResponse, HealthResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_service")

app = FastAPI(
    title="Traffic Accident Severity Prediction API",
    description="Production MLOps inference microservice predicting road traffic accident severity",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
START_TIME = time.time()
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "accident_severity_model.joblib"))
MODEL_METADATA_PATH = Path(MODEL_PATH).with_name("model_metadata.json")
MODEL_PIPELINE = None
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"


def _model_version() -> str:
    """Registered MLflow model version written by the training pipeline."""
    try:
        with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
            return json.load(f).get("model_version", "unregistered")
    except (OSError, ValueError):
        return "unregistered"


MODEL_VERSION = _model_version()

# Performance counters
REQUEST_COUNT = 0
FAILURE_COUNT = 0
LOG_FAILURE_COUNT = 0
TOTAL_INFERENCE_TIME_MS = 0.0
PREDICTED_CLASS_COUNTS = {"Fatal": 0, "Serious": 0, "Slight": 0}


def log_prediction(input_data, code: int, probas, latency_ms: float):
    """Best-effort write to prediction_log; a logging failure never fails the request."""
    global LOG_FAILURE_COUNT
    if not LOG_PREDICTIONS:
        return
    try:
        from sqlalchemy import text
        from src.etl.db import get_engine
        with get_engine().begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO prediction_log (accident_index, request_payload, predicted_severity_code,
                                                p_fatal, p_serious, p_slight, model_version, latency_ms, predicted_at)
                    VALUES (:idx, :payload, :code, :pf, :ps, :pl, :ver, :lat, :at)
                """),
                {
                    "idx": input_data.accident_index,
                    "payload": input_data.model_dump_json(),
                    "code": code,
                    "pf": float(probas[2]), "ps": float(probas[1]), "pl": float(probas[0]),
                    "ver": MODEL_VERSION,
                    "lat": latency_ms,
                    "at": datetime.datetime.now(datetime.timezone.utc),
                },
            )
    except Exception as e:
        LOG_FAILURE_COUNT += 1
        logger.warning(f"Prediction log write failed: {e}")


def get_model():
    """Loads model pipeline lazily or at startup."""
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        p = Path(MODEL_PATH)
        if p.exists():
            MODEL_PIPELINE = joblib.load(p)
            logger.info(f"Successfully loaded model pipeline from {p}")
        else:
            logger.warning(f"Model file not found at {p}. Will attempt fallback.")
    return MODEL_PIPELINE


@app.on_event("startup")
def startup_event():
    get_model()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns microservice health, model status, and uptime."""
    model = get_model()
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=type(model.named_steps["classifier"]).__name__ if model else None,
        version=MODEL_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


def process_features_to_df(acc: AccidentFeatureInput) -> pd.DataFrame:
    """Prepares input dictionary and derives cyclical temporal features."""
    hour = acc.hour
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    data = {
        "hour": [hour],
        "speed_limit": [acc.speed_limit],
        "temperature_c": [acc.temperature_c],
        "precipitation_mm": [acc.precipitation_mm],
        "visibility_m": [acc.visibility_m],
        "wind_speed_kmh": [acc.wind_speed_kmh],
        "number_of_vehicles": [acc.number_of_vehicles],
        "number_of_casualties": [acc.number_of_casualties],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "day_name": [acc.day_name],
        "time_of_day": [acc.time_of_day],
        "urban_or_rural": [acc.urban_or_rural],
        "road_type": [acc.road_type],
        "light_conditions": [acc.light_conditions],
        "road_surface_conditions": [acc.road_surface_conditions],
        "weather_condition": [acc.weather_condition]
    }
    return pd.DataFrame(data)


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_severity(input_data: AccidentFeatureInput):
    """
    Predicts accident severity for a given set of environmental, temporal, and road features.
    Returns predicted class (Fatal, Serious, Slight), probabilities, and risk score.
    """
    global REQUEST_COUNT, FAILURE_COUNT, TOTAL_INFERENCE_TIME_MS
    t0 = time.time()

    model = get_model()
    if model is None:
        FAILURE_COUNT += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train or provide model artifact."
        )
        
    try:
        input_df = process_features_to_df(input_data)
        
        # Predicted class (0=Slight, 1=Serious, 2=Fatal)
        pred_class_idx = int(model.predict(input_df)[0])
        probas = model.predict_proba(input_df)[0]
        
        # Mappings
        idx_to_label = {0: "Slight", 1: "Serious", 2: "Fatal"}
        idx_to_code = {0: 3, 1: 2, 2: 1} # 1: Fatal, 2: Serious, 3: Slight
        
        # Composite risk score: (0.1 * P_slight + 0.5 * P_serious + 1.0 * P_fatal)
        risk_score = round(float(0.1 * probas[0] + 0.5 * probas[1] + 1.0 * probas[2]), 4)
        
        prob_dict = {
            "Slight": round(float(probas[0]), 4),
            "Serious": round(float(probas[1]), 4),
            "Fatal": round(float(probas[2]), 4)
        }
        
        elapsed_ms = (time.time() - t0) * 1000.0
        REQUEST_COUNT += 1
        TOTAL_INFERENCE_TIME_MS += elapsed_ms
        PREDICTED_CLASS_COUNTS[idx_to_label[pred_class_idx]] += 1
        log_prediction(input_data, idx_to_code[pred_class_idx], probas, elapsed_ms)

        return PredictionResult(
            predicted_severity_code=idx_to_code[pred_class_idx],
            predicted_severity_label=idx_to_label[pred_class_idx],
            severity_risk_score=risk_score,
            class_probabilities=prob_dict,
            model_version=MODEL_VERSION,
            inference_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
    except Exception as e:
        FAILURE_COUNT += 1
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_request: BatchPredictionRequest):
    """Executes high-throughput batch predictions."""
    t0 = time.time()
    results = []
    for item in batch_request.accidents:
        results.append(predict_severity(item))
    elapsed_ms = (time.time() - t0) * 1000.0
    return BatchPredictionResponse(
        predictions=results,
        total_processed=len(results),
        batch_latency_ms=round(elapsed_ms, 2)
    )


@app.get("/metrics", tags=["Monitoring"])
def prometheus_metrics():
    """Prometheus monitoring metrics endpoint."""
    avg_latency = (TOTAL_INFERENCE_TIME_MS / REQUEST_COUNT) if REQUEST_COUNT > 0 else 0.0
    class_lines = "".join(
        f'model_predictions_total{{severity="{label}"}} {count}\n' for label, count in PREDICTED_CLASS_COUNTS.items()
    )
    metrics_text = (
        f"# HELP model_requests_total Total number of successful predictions\n"
        f"# TYPE model_requests_total counter\n"
        f"model_requests_total {REQUEST_COUNT}\n\n"
        f"# HELP model_request_failures_total Prediction requests that failed (model missing or inference error)\n"
        f"# TYPE model_request_failures_total counter\n"
        f"model_request_failures_total {FAILURE_COUNT}\n\n"
        f"# HELP prediction_log_failures_total Predictions that could not be written to prediction_log\n"
        f"# TYPE prediction_log_failures_total counter\n"
        f"prediction_log_failures_total {LOG_FAILURE_COUNT}\n\n"
        f"# HELP model_predictions_total Predictions served per severity class\n"
        f"# TYPE model_predictions_total counter\n"
        f"{class_lines}\n"
        f"# HELP model_inference_avg_latency_ms Average inference latency in ms\n"
        f"# TYPE model_inference_avg_latency_ms gauge\n"
        f"model_inference_avg_latency_ms {avg_latency:.2f}\n\n"
        f"# HELP model_info Registered model version currently served\n"
        f"# TYPE model_info gauge\n"
        f'model_info{{version="{MODEL_VERSION}"}} 1\n'
    )
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=metrics_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

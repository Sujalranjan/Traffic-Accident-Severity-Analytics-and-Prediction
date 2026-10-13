"""
Pydantic Schemas for FastAPI Traffic Accident Severity Prediction Service.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class AccidentFeatureInput(BaseModel):
    """Feature values use the decoded DfT labels stored in the warehouse (see src/etl/standardize.py)."""
    accident_index: Optional[str] = Field(
        default=None, example="2024010012345",
        description="Optional DfT collision index; lets monitoring compare the prediction with the recorded severity",
    )
    hour: int = Field(..., ge=0, le=23, example=17, description="Hour of day (0-23)")
    day_name: str = Field(default="Friday", example="Friday", description="Day of week name")
    time_of_day: str = Field(default="Evening Rush", example="Evening Rush")
    latitude: float = Field(..., ge=49.0, le=61.5, example=51.5074, description="Latitude (UK bounds)")
    longitude: float = Field(..., ge=-8.5, le=2.5, example=-0.1278, description="Longitude (UK bounds)")
    urban_or_rural: str = Field(default="Urban", example="Urban")
    road_type: str = Field(default="Single carriageway", example="Single carriageway")
    speed_limit: int = Field(default=30, ge=10, le=100, example=30)
    light_conditions: str = Field(default="Daylight", example="Daylight")
    road_surface_conditions: str = Field(default="Dry", example="Wet or damp")
    weather_condition: str = Field(default="Fine no high winds", example="Raining no high winds")
    temperature_c: float = Field(default=12.0, example=9.5)
    precipitation_mm: float = Field(default=0.0, example=2.4)
    visibility_m: float = Field(default=10000.0, example=3500.0)
    wind_speed_kmh: float = Field(default=15.0, example=24.0)
    number_of_vehicles: int = Field(default=2, ge=1, le=20, example=2)
    number_of_casualties: int = Field(default=1, ge=1, le=20, example=1)


class PredictionResult(BaseModel):
    predicted_severity_code: int = Field(..., description="1: Fatal, 2: Serious, 3: Slight")
    predicted_severity_label: str = Field(..., description="Fatal, Serious, or Slight")
    severity_risk_score: float = Field(..., description="Normalized composite severity risk score (0.0 to 1.0)")
    class_probabilities: Dict[str, float] = Field(..., description="Probability per severity class")
    model_version: str
    inference_timestamp: str


class BatchPredictionRequest(BaseModel):
    accidents: List[AccidentFeatureInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    total_processed: int
    batch_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    version: str
    uptime_seconds: float

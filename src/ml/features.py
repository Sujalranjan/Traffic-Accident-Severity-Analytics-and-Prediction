"""
Feature Engineering and Dataset Preparation Pipeline for Accident Severity Prediction.
Implements chronological train/val/test split and preprocessor pipelines.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.etl.db import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("feature_engineering")

# Feature definition
NUMERIC_FEATURES = [
    "hour", "speed_limit", "temperature_c", "precipitation_mm",
    "visibility_m", "wind_speed_kmh", "number_of_vehicles", "number_of_casualties",
    "hour_sin", "hour_cos"
]

CATEGORICAL_FEATURES = [
    "day_name", "time_of_day", "urban_or_rural", "road_type",
    "light_conditions", "road_surface_conditions", "weather_condition"
]


def load_dataset_from_warehouse() -> pd.DataFrame:
    """
    Loads enriched accident data by joining fact and dimension tables.
    """
    engine = get_engine()
    query = """
        SELECT 
            f.accident_index,
            d.full_date,
            d.year,
            d.month,
            d.day_name,
            d.hour,
            d.is_weekend,
            d.time_of_day,
            l.latitude,
            l.longitude,
            l.urban_or_rural,
            r.road_type,
            r.speed_limit,
            r.light_conditions,
            r.road_surface_conditions,
            w.weather_condition,
            w.temperature_c,
            w.precipitation_mm,
            w.visibility_m,
            w.wind_speed_kmh,
            f.number_of_vehicles,
            f.number_of_casualties,
            s.severity_code,
            s.severity_name
        FROM fact_accidents f
        JOIN dim_date d ON f.date_key = d.date_key
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_road r ON f.road_key = r.road_key
        JOIN dim_weather w ON f.weather_key = w.weather_key
        JOIN dim_severity s ON f.severity_key = s.severity_key
    """
    df = pd.read_sql(query, con=engine)
    logger.info(f"Loaded {len(df)} records from warehouse for feature engineering.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived temporal and spatial features.
    """
    df = df.copy()
    
    # Cyclical hour encoding (hour_sin, hour_cos)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    
    # Fill categorical NAs
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            
    # Coerce numerics; missing values (e.g. no weather match) are imputed inside the pipeline
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Constructs a robust Scikit-Learn ColumnTransformer pipeline.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ], remainder="drop")
    
    return preprocessor


def get_chronological_splits(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Performs problem-appropriate chronological train/validation/test split
    as explicitly required in Section 8 of the assignment rubric.
    Prevents temporal data leakage.
    """
    df = engineer_features(df)
    
    # Sort chronologically by date and hour
    # full_date is stored as ISO YYYY-MM-DD by the ETL
    full_date = pd.to_datetime(df["full_date"].astype(str).str.slice(0, 10), format="%Y-%m-%d")
    df["datetime_sort"] = full_date + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values(by="datetime_sort").reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    # Target: severity_code (1=Fatal, 2=Serious, 3=Slight) mapped to 0, 1, 2 for zero-indexed classifiers
    # 0: Slight (majority), 1: Serious, 2: Fatal (critical minority)
    target_mapping = {3: 0, 2: 1, 1: 2}
    
    X_train = train_df[feature_cols]
    y_train = train_df["severity_code"].map(target_mapping).values
    
    X_val = val_df[feature_cols]
    y_val = val_df["severity_code"].map(target_mapping).values
    
    X_test = test_df[feature_cols]
    y_test = test_df["severity_code"].map(target_mapping).values
    
    logger.info(f"Chronological split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

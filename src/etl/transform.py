"""
ETL Transformation and Star Schema Warehouse Loader.
Transforms validated staging data, decodes DfT codes, joins Open-Meteo weather by
grid cell and local hour, populates dimensions & facts, and refreshes analytical marts.
Safe to re-run: every load is an insert-or-ignore on the table's natural key.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import pandas as pd
from sqlalchemy import text
from src.etl.db import get_engine, init_db, insert_ignore_duplicates, run_sql_file, apply_postgis, is_sqlite
from src.etl.validation import validate_and_quarantine
from src.etl import standardize as std
from src.ingestion.weather_ingest import WEATHER_GRID_DEG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("etl_transform")

SNOW_ICE_CODES = {56, 57, 66, 67, 71, 73, 75, 77, 85, 86}


def time_of_day_bucket(hour: int) -> str:
    if 6 <= hour < 10:
        return "Morning Rush"
    elif 10 <= hour < 16:
        return "Midday"
    elif 16 <= hour < 20:
        return "Evening Rush"
    else:
        return "Night"


def weather_risk_level(cond: str, temp, precip, code) -> str:
    """Risk from measured weather when available, otherwise from the police-reported condition."""
    cond = cond or ""
    if pd.notna(temp) or pd.notna(precip):
        if (pd.notna(code) and int(code) in SNOW_ICE_CODES) or (pd.notna(precip) and precip > 5.0) \
                or (pd.notna(temp) and temp < 0):
            return "Severe"
        if (pd.notna(precip) and precip > 0.5) or (pd.notna(code) and int(code) in {45, 48}):
            return "Moderate"
        return "Low"
    if "Snow" in cond or "Fog" in cond:
        return "Severe"
    if "Rain" in cond or "high winds" in cond:
        return "Moderate"
    return "Low"


def load_weather_observations(engine) -> pd.DataFrame:
    w = pd.read_sql(
        "SELECT latitude, longitude, weather_timestamp, temperature_2m, precipitation, wind_speed_10m, "
        "visibility, weather_code, weather_description FROM stg_weather_raw",
        con=engine,
    )
    if w.empty:
        return w
    w["grid_lat"] = w["latitude"].astype(float).round(4)
    w["grid_lon"] = w["longitude"].astype(float).round(4)
    return w.drop_duplicates(subset=["grid_lat", "grid_lon", "weather_timestamp"], keep="last")


def _keys(engine, table: str, cols: list) -> pd.DataFrame:
    return pd.read_sql(f"SELECT {', '.join(cols)} FROM {table}", con=engine)


def run_etl_pipeline(batch_id: str = None, persist_rejections: bool = True) -> dict:
    """
    Full ETL transformation pipeline for one staged batch (or all staged rows).
    persist_rejections=False when validation already ran as a separate DAG task.
    """
    engine = get_engine()
    init_db(engine)
    sqlite = is_sqlite(engine)

    # 1. Validation & Quarantine
    validation_res = validate_and_quarantine(batch_id=batch_id, persist=persist_rejections)
    df = validation_res.get("valid_df")
    processed_batches = validation_res.get("batches", set())

    if df is None or df.empty:
        logger.warning("No valid records to process in ETL pipeline.")
        _cleanup_staging(engine, batch_id, processed_batches)
        return {"loaded": 0, "rejected": validation_res.get("rejected", 0)}

    logger.info(f"Transforming {len(df):,} validated accident records...")

    # 2. Standardise date & time (DD/MM/YYYY parsed explicitly; VR-03 guarantees validity)
    df["dt"] = std.parse_accident_datetime(df["accident_date"], df["accident_time"])
    df["full_date"] = df["dt"].dt.date
    df["year"] = df["dt"].dt.year
    df["quarter"] = df["dt"].dt.quarter
    df["month"] = df["dt"].dt.month
    df["month_name"] = df["dt"].dt.strftime("%B")
    df["day_of_month"] = df["dt"].dt.day
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["day_name"] = df["dt"].dt.strftime("%A")
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["hour"] = df["dt"].dt.hour
    df["time_of_day"] = df["hour"].apply(time_of_day_bucket)
    df["date_key"] = df["dt"].dt.strftime("%Y%m%d%H").astype(int)

    # 3. Decode DfT codes to labels
    df["severity_key"] = pd.to_numeric(df["accident_severity"], errors="coerce").astype(int)
    df["road_type"] = std.decode_column(df["road_type"], std.ROAD_TYPE)
    df["light_conditions"] = std.decode_column(df["light_conditions"], std.LIGHT_CONDITIONS)
    df["weather_condition"] = std.decode_column(df["weather_conditions"], std.WEATHER_CONDITIONS)
    df["road_surface_conditions"] = std.decode_column(df["road_surface_conditions"], std.ROAD_SURFACE_CONDITIONS)
    df["urban_or_rural"] = std.decode_column(df["urban_or_rural_area"], std.URBAN_OR_RURAL)
    df["local_authority"] = df["local_authority"].fillna("Unknown").astype(str).str.strip()
    df["region"] = std.region_from_ons_code(df["local_authority"])
    df["latitude"] = pd.to_numeric(df["latitude"]).astype(float).round(6)
    df["longitude"] = pd.to_numeric(df["longitude"]).astype(float).round(6)
    df["speed_limit"] = pd.to_numeric(df["speed_limit"]).astype(int)
    for col in ["number_of_vehicles", "number_of_casualties"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)

    # 4. Join Open-Meteo weather on (grid cell, local hour)
    df["grid_lat"] = std.weather_grid(df["latitude"], WEATHER_GRID_DEG)
    df["grid_lon"] = std.weather_grid(df["longitude"], WEATHER_GRID_DEG)
    df["observed_hour"] = std.observation_hour(df["dt"])
    weather = load_weather_observations(engine)
    if weather.empty:
        for col in ["temperature_2m", "precipitation", "wind_speed_10m", "visibility", "weather_code", "weather_description"]:
            df[col] = np.nan
    else:
        df = df.merge(
            weather.drop(columns=["latitude", "longitude"]),
            left_on=["grid_lat", "grid_lon", "observed_hour"],
            right_on=["grid_lat", "grid_lon", "weather_timestamp"],
            how="left",
        )
    has_weather = df["temperature_2m"].notna() | df["precipitation"].notna()
    logger.info(f"Weather matched for {has_weather.sum():,} of {len(df):,} accidents ({has_weather.mean():.1%}).")

    df["weather_source"] = np.where(has_weather, "Open-Meteo", "Unavailable")
    df["weather_nk"] = np.where(
        has_weather,
        df["grid_lat"].map("{:.4f}".format) + "|" + df["grid_lon"].map("{:.4f}".format) + "|" + df["observed_hour"] + "|" + df["weather_condition"],
        "unavailable|" + df["weather_condition"],
    )
    df["weather_risk_level"] = [
        weather_risk_level(c, t, p, k)
        for c, t, p, k in zip(df["weather_condition"], df["temperature_2m"], df["precipitation"], df["weather_code"])
    ]

    # 5. Load dimensions (insert-or-ignore on natural keys)
    dim_date = df[[
        "date_key", "full_date", "year", "quarter", "month", "month_name",
        "day_of_month", "day_of_week", "day_name", "is_weekend", "hour", "time_of_day",
    ]].drop_duplicates(subset=["date_key"])
    insert_ignore_duplicates(engine, dim_date, "dim_date", ["date_key"])

    loc_cols = ["latitude", "longitude", "urban_or_rural", "local_authority"]
    dim_location = df[loc_cols + ["region"]].drop_duplicates(subset=loc_cols)
    insert_ignore_duplicates(engine, dim_location, "dim_location", loc_cols)

    road_cols = ["road_type", "speed_limit", "light_conditions", "road_surface_conditions"]
    insert_ignore_duplicates(engine, df[road_cols].drop_duplicates(), "dim_road", road_cols)

    dim_weather = pd.DataFrame({
        "weather_nk": df["weather_nk"],
        "grid_latitude": np.where(has_weather, df["grid_lat"], np.nan),
        "grid_longitude": np.where(has_weather, df["grid_lon"], np.nan),
        "observed_hour": np.where(has_weather, df["observed_hour"], None),
        "weather_condition": df["weather_condition"],
        "weather_description": df["weather_description"],
        "weather_code": pd.to_numeric(df["weather_code"], errors="coerce").astype("Int64"),
        "temperature_c": df["temperature_2m"],
        "precipitation_mm": df["precipitation"],
        "visibility_m": df["visibility"],
        "wind_speed_kmh": df["wind_speed_10m"],
        "weather_risk_level": df["weather_risk_level"],
        "weather_source": df["weather_source"],
    }).drop_duplicates(subset=["weather_nk"])
    insert_ignore_duplicates(engine, dim_weather, "dim_weather", ["weather_nk"])

    # 6. Resolve surrogate keys
    loc_db = _keys(engine, "dim_location", ["location_key"] + loc_cols)
    loc_db["latitude"] = loc_db["latitude"].astype(float).round(6)
    loc_db["longitude"] = loc_db["longitude"].astype(float).round(6)
    df = df.merge(loc_db, on=loc_cols, how="left")

    road_db = _keys(engine, "dim_road", ["road_key"] + road_cols)
    road_db["speed_limit"] = road_db["speed_limit"].astype(int)
    df = df.merge(road_db, on=road_cols, how="left")

    wea_db = _keys(engine, "dim_weather", ["weather_key", "weather_nk"])
    df = df.merge(wea_db, on="weather_nk", how="left")

    unresolved = df[["location_key", "road_key", "weather_key"]].isna().any(axis=1)
    if unresolved.any():
        raise RuntimeError(f"{unresolved.sum()} accidents could not be matched to their dimension keys")

    # 7. Load facts
    facts = df[[
        "accident_index", "date_key", "location_key", "road_key", "weather_key",
        "severity_key", "number_of_vehicles", "number_of_casualties",
    ]].drop_duplicates(subset=["accident_index"]).astype({"location_key": int, "road_key": int, "weather_key": int})
    before = pd.read_sql("SELECT COUNT(*) AS n FROM fact_accidents", con=engine)["n"].iloc[0]
    insert_ignore_duplicates(engine, facts, "fact_accidents", ["accident_index"])
    after = pd.read_sql("SELECT COUNT(*) AS n FROM fact_accidents", con=engine)["n"].iloc[0]
    logger.info(f"Loaded {after - before:,} new facts (fact_accidents now holds {after:,} rows).")

    # 8. Refresh marts and spatial column
    run_sql_file(engine, "03_analytical_marts.sql")
    if not sqlite:
        apply_postgis(engine)

    # 9. Clear the processed batch(es) from staging
    _cleanup_staging(engine, batch_id, processed_batches)
    logger.info("ETL transformation, star schema load, and analytical marts refresh complete!")
    return {"loaded": int(after - before), "rejected": validation_res.get("rejected", 0),
            "weather_match_rate": float(has_weather.mean())}


def _cleanup_staging(engine, batch_id=None, processed_batches=()):
    """Removes processed rows (valid and quarantined) from staging so they are not reprocessed."""
    batches = {batch_id} if batch_id else set(processed_batches)
    if not batch_id and not processed_batches:
        return
    with engine.begin() as conn:
        for b in batches:
            conn.execute(text("DELETE FROM stg_accidents_raw WHERE batch_id = :b"), {"b": b})
    logger.info(f"Staging cleared for batch(es): {', '.join(sorted(batches))}")


if __name__ == "__main__":
    run_etl_pipeline()

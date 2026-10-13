"""
Ingestion script for the Open-Meteo Historical Weather API.

For every accident waiting in staging, the location is snapped to a weather grid
cell (WEATHER_GRID_DEG, default 1.0 degree, sized so a full
5-year backfill fits the Open-Meteo free-tier daily limit) and the time to the local hour. One API
request per grid cell and calendar year fetches the hourly series, the full responses
are kept as a gzipped raw landing file, and only the hours that match an accident
are loaded into stg_weather_raw. Failed requests are logged, never replaced by
made-up values.
"""

import os
import sys
import gzip
import json
import time
import uuid
import logging
import datetime
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.etl.db import DATA_DIR, get_engine, init_db
from src.etl.standardize import parse_accident_datetime, weather_grid, observation_hour

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("weather_ingest")

OPEN_METEO_BASE = os.getenv("OPEN_METEO_BASE_URL", "https://archive-api.open-meteo.com/v1/archive")
WEATHER_GRID_DEG = float(os.getenv("WEATHER_GRID_DEG", "1.0"))
HOURLY_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "visibility", "weather_code"]
REQUEST_PAUSE_S = float(os.getenv("OPEN_METEO_PAUSE_S", "0.2"))


class RateLimited(Exception):
    pass


def weather_code_to_desc(code) -> str:
    """Translates WMO weather codes to human-readable weather descriptions."""
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall", 77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    if code is None or pd.isna(code):
        return "Unknown"
    return mapping.get(int(code), "Unknown")


def fetch_weather_series(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Queries Open-Meteo for hourly weather at one grid cell over a date range (ISO dates).
    Timestamps come back in UK local time to match DfT accident times.
    Raises RateLimited on HTTP 429 and requests exceptions on other failures.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/London",
    }
    response = requests.get(OPEN_METEO_BASE, params=params, timeout=60)
    if response.status_code == 429:
        raise RateLimited(response.text[:200])
    response.raise_for_status()
    return response.json()


def hourly_frame(payload: dict, lat: float, lon: float) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    frame = pd.DataFrame({
        "weather_timestamp": hourly.get("time", []),
        "temperature_2m": hourly.get("temperature_2m"),
        "precipitation": hourly.get("precipitation"),
        "wind_speed_10m": hourly.get("wind_speed_10m"),
        "visibility": hourly.get("visibility"),
        "weather_code": hourly.get("weather_code"),
    })
    frame["latitude"] = lat
    frame["longitude"] = lon
    return frame


def needed_observations(engine, batch_id: str = None) -> pd.DataFrame:
    """(grid cell, local hour) pairs for staged accidents that do not have weather yet."""
    query = "SELECT accident_date, accident_time, latitude, longitude FROM stg_accidents_raw"
    params = {}
    if batch_id:
        query += " WHERE batch_id = :batch_id"
        params["batch_id"] = batch_id
    acc = pd.read_sql(text(query), con=engine, params=params)

    acc["dt"] = parse_accident_datetime(acc["accident_date"], acc["accident_time"])
    acc = acc.dropna(subset=["dt", "latitude", "longitude"])
    acc = acc[acc["latitude"].between(49.0, 61.5) & acc["longitude"].between(-8.5, 2.5)]
    if acc.empty:
        return pd.DataFrame(columns=["latitude", "longitude", "weather_timestamp"])

    needed = pd.DataFrame({
        "latitude": weather_grid(acc["latitude"], WEATHER_GRID_DEG),
        "longitude": weather_grid(acc["longitude"], WEATHER_GRID_DEG),
        "weather_timestamp": observation_hour(acc["dt"]),
    }).drop_duplicates()

    have = pd.read_sql("SELECT DISTINCT latitude, longitude, weather_timestamp FROM stg_weather_raw", con=engine)
    if not have.empty:
        have["latitude"] = have["latitude"].astype(float).round(4)
        have["longitude"] = have["longitude"].astype(float).round(4)
        needed = needed.merge(have, on=["latitude", "longitude", "weather_timestamp"], how="left", indicator=True)
        needed = needed[needed["_merge"] == "left_only"].drop(columns="_merge")
    return needed


def ingest_weather_for_accidents(batch_id: str = None) -> str:
    """
    Fetches Open-Meteo weather for the staged accidents of one batch (or all staged
    accidents when batch_id is None). Returns the weather batch id.
    """
    engine = get_engine()
    init_db(engine)
    weather_batch_id = f"batch_wea_{uuid.uuid4().hex[:8]}"
    start_time = datetime.datetime.now(datetime.timezone.utc)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"weather_raw_{weather_batch_id}.jsonl.gz"

    needed = needed_observations(engine, batch_id)
    needed["year"] = needed["weather_timestamp"].str.slice(0, 4)
    requests_plan = (
        needed.groupby(["latitude", "longitude", "year"])["weather_timestamp"]
        .agg(["min", "max"]).reset_index()
    )
    logger.info(
        f"{len(needed):,} accident hours need weather across {needed[['latitude', 'longitude']].drop_duplicates().shape[0]} "
        f"grid cells ({WEATHER_GRID_DEG} deg); {len(requests_plan)} API requests planned."
    )

    loaded_rows, failures, rate_limited = 0, [], False
    with gzip.open(raw_file, "wt", encoding="utf-8") as raw_out:
        for i, req in enumerate(requests_plan.itertuples(index=False)):
            lat, lon = float(req.latitude), float(req.longitude)
            start_d, end_d = req.min[:10], req.max[:10]
            try:
                payload = fetch_weather_series(lat, lon, start_d, end_d)
            except RateLimited as e:
                failures.append(f"{lat},{lon} {start_d}..{end_d}: rate limited")
                rate_limited = True
                logger.warning(f"Open-Meteo rate limit reached after {i} requests; remaining cells will be fetched on the next run. {e}")
                break
            except Exception as e:
                failures.append(f"{lat},{lon} {start_d}..{end_d}: {e.__class__.__name__}")
                logger.warning(f"Weather request failed for cell ({lat}, {lon}) {start_d}..{end_d}: {e}")
                continue

            raw_out.write(json.dumps({"latitude": lat, "longitude": lon, "start_date": start_d,
                                      "end_date": end_d, "response": payload}) + "\n")
            series = hourly_frame(payload, lat, lon)
            cell_needed = needed[(needed["latitude"] == lat) & (needed["longitude"] == lon)]
            series = series.merge(cell_needed[["weather_timestamp"]], on="weather_timestamp", how="inner")
            if not series.empty:
                series["weather_description"] = series["weather_code"].map(weather_code_to_desc)
                series["batch_id"] = weather_batch_id
                series.to_sql("stg_weather_raw", con=engine, if_exists="append", index=False)
                loaded_rows += len(series)
            if REQUEST_PAUSE_S:
                time.sleep(REQUEST_PAUSE_S)

    attempted = len(requests_plan)
    if attempted == 0 or not failures:
        status = "SUCCESS"
    elif loaded_rows == 0:
        status = "FAILED"
    else:
        status = "PARTIAL"
    error_summary = None
    if failures:
        error_summary = f"{len(failures)} of {attempted} requests failed" + ("; stopped on rate limit" if rate_limited else "") + \
                        ". " + " | ".join(failures[:10])

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO ingestion_audit (batch_id, source_name, file_or_endpoint, status, records_ingested,
                                             records_rejected, started_at, completed_at, error_message)
                VALUES (:batch_id, :source, :endpoint, :status, :records, :rejected, :started, :completed, :err)
            """),
            {
                "batch_id": weather_batch_id,
                "source": "Open-Meteo Historical Weather API",
                "endpoint": f"{OPEN_METEO_BASE} (raw: {raw_file.name})",
                "status": status,
                "records": loaded_rows,
                "rejected": len(failures),
                "started": start_time,
                "completed": datetime.datetime.now(datetime.timezone.utc),
                "err": error_summary,
            },
        )
    logger.info(f"Weather batch {weather_batch_id}: {status}, {loaded_rows:,} hourly readings staged, {len(failures)} failed requests.")
    return weather_batch_id


if __name__ == "__main__":
    ingest_weather_for_accidents()

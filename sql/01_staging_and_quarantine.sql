-- 01_staging_and_quarantine.sql
-- Ingestion audit, raw landing tables, and rejected record quarantine log

-- Audit table to log each ingestion batch
CREATE TABLE IF NOT EXISTS ingestion_audit (
    batch_id VARCHAR(64) PRIMARY KEY,
    source_name VARCHAR(128) NOT NULL,
    file_or_endpoint VARCHAR(256) NOT NULL,
    status VARCHAR(32) NOT NULL, -- 'SUCCESS', 'FAILED', 'PARTIAL'
    records_ingested INT DEFAULT 0,
    records_rejected INT DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Raw staging table for accident records
CREATE TABLE IF NOT EXISTS stg_accidents_raw (
    accident_index VARCHAR(64),
    accident_date VARCHAR(64),
    accident_time VARCHAR(64),
    latitude NUMERIC(10, 6),
    longitude NUMERIC(10, 6),
    accident_severity VARCHAR(32),
    number_of_vehicles INT,
    number_of_casualties INT,
    speed_limit INT,
    road_type VARCHAR(64),
    light_conditions VARCHAR(128),
    weather_conditions VARCHAR(128),
    road_surface_conditions VARCHAR(128),
    urban_or_rural_area VARCHAR(32),
    local_authority VARCHAR(128),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    batch_id VARCHAR(64)
);

-- Raw staging table for Open-Meteo weather records
-- latitude/longitude are the weather grid cell, weather_timestamp is the local UK hour (ISO, minutes zeroed)
CREATE TABLE IF NOT EXISTS stg_weather_raw (
    weather_timestamp VARCHAR(16),
    latitude NUMERIC(10, 6),
    longitude NUMERIC(10, 6),
    temperature_2m NUMERIC(5, 2),
    precipitation NUMERIC(5, 2),
    wind_speed_10m NUMERIC(5, 2),
    visibility NUMERIC(8, 2),
    weather_code INT,
    weather_description VARCHAR(128),
    batch_id VARCHAR(64),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Quarantine / Error Log for rejected records failing quality validation
CREATE TABLE IF NOT EXISTS quarantine_rejected_records (
    rejection_id SERIAL PRIMARY KEY,
    batch_id VARCHAR(64),
    source_entity VARCHAR(64),
    record_identifier VARCHAR(128),
    rejection_reason VARCHAR(256) NOT NULL,
    raw_payload TEXT,
    rejected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Every prediction served by the API, for monitoring latency, class mix and
-- (once the true severity is known via accident_index) false-negative rates
CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id SERIAL PRIMARY KEY,
    accident_index VARCHAR(64),
    request_payload TEXT,
    predicted_severity_code INT,
    p_fatal NUMERIC(9, 4),
    p_serious NUMERIC(9, 4),
    p_slight NUMERIC(9, 4),
    model_version VARCHAR(64),
    latency_ms NUMERIC(9, 4),
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stg_accidents_batch ON stg_accidents_raw(batch_id);
CREATE INDEX IF NOT EXISTS idx_stg_weather_key ON stg_weather_raw(latitude, longitude, weather_timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_log_index ON prediction_log(accident_index);
CREATE INDEX IF NOT EXISTS idx_quarantine_batch ON quarantine_rejected_records(batch_id);

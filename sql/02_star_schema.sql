-- 02_star_schema.sql
-- Dimensional Model (Star Schema) for Traffic Accident Analytics Warehouse

-- Dim Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INT PRIMARY KEY,              -- YYYYMMDDHH
    full_date DATE NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    month INT NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    day_of_month INT NOT NULL,
    day_of_week INT NOT NULL,              -- 0=Monday, 6=Sunday
    day_name VARCHAR(20) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    hour INT NOT NULL,
    time_of_day VARCHAR(30) NOT NULL       -- 'Morning Rush', 'Midday', 'Evening Rush', 'Night'
);

-- Dim Location
CREATE TABLE IF NOT EXISTS dim_location (
    location_key SERIAL PRIMARY KEY,
    latitude NUMERIC(10, 6) NOT NULL,
    longitude NUMERIC(10, 6) NOT NULL,
    urban_or_rural VARCHAR(32),
    local_authority VARCHAR(128),          -- ONS local authority district code
    region VARCHAR(64),                    -- nation derived from the ONS district code
    UNIQUE (latitude, longitude, urban_or_rural, local_authority)
);

-- Dim Road
CREATE TABLE IF NOT EXISTS dim_road (
    road_key SERIAL PRIMARY KEY,
    road_type VARCHAR(64) NOT NULL,
    speed_limit INT NOT NULL,
    light_conditions VARCHAR(128) NOT NULL,
    road_surface_conditions VARCHAR(128) NOT NULL,
    UNIQUE (road_type, speed_limit, light_conditions, road_surface_conditions)
);

-- Dim Weather
-- Grain: one Open-Meteo hourly observation for a weather grid cell, combined with the
-- police-reported DfT weather condition. weather_nk is the natural key used for upserts.
CREATE TABLE IF NOT EXISTS dim_weather (
    weather_key SERIAL PRIMARY KEY,
    weather_nk VARCHAR(160) NOT NULL UNIQUE,
    grid_latitude NUMERIC(9, 4),
    grid_longitude NUMERIC(9, 4),
    observed_hour VARCHAR(16),             -- local UK hour, ISO format
    weather_condition VARCHAR(128) NOT NULL,   -- DfT police-reported condition
    weather_description VARCHAR(128),      -- Open-Meteo WMO weather code description
    weather_code INT,
    temperature_c NUMERIC(5, 2),
    precipitation_mm NUMERIC(5, 2),
    visibility_m NUMERIC(8, 2),
    wind_speed_kmh NUMERIC(5, 2),
    weather_risk_level VARCHAR(32),        -- 'Low', 'Moderate', 'Severe'
    weather_source VARCHAR(32)             -- 'Open-Meteo' or 'Unavailable'
);

-- Dim Severity
CREATE TABLE IF NOT EXISTS dim_severity (
    severity_key INT PRIMARY KEY,
    severity_code INT NOT NULL,            -- 1: Fatal, 2: Serious, 3: Slight
    severity_name VARCHAR(32) NOT NULL,    -- 'Fatal', 'Serious', 'Slight'
    severity_description TEXT
);

-- Fact Accidents
CREATE TABLE IF NOT EXISTS fact_accidents (
    accident_key SERIAL PRIMARY KEY,
    accident_index VARCHAR(64) UNIQUE NOT NULL,
    date_key INT REFERENCES dim_date(date_key),
    location_key INT REFERENCES dim_location(location_key),
    road_key INT REFERENCES dim_road(road_key),
    weather_key INT REFERENCES dim_weather(weather_key),
    severity_key INT REFERENCES dim_severity(severity_key),
    number_of_vehicles INT DEFAULT 1,
    number_of_casualties INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_accidents(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_location ON fact_accidents(location_key);
CREATE INDEX IF NOT EXISTS idx_fact_severity ON fact_accidents(severity_key);
CREATE INDEX IF NOT EXISTS idx_fact_weather ON fact_accidents(weather_key);

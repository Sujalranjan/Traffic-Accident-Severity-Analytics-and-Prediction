-- 03_analytical_marts.sql
-- Aggregate Marts & Analytical Views for Streamlit / Tableau & ML Features
-- Marts are dropped and rebuilt on every run so they always match fact_accidents.

-- 1. Aggregated Mart: Accidents by Hour and Day of Week
DROP TABLE IF EXISTS agg_hourly_severity_mart;
CREATE TABLE agg_hourly_severity_mart AS
SELECT 
    d.day_name,
    d.day_of_week,
    d.hour,
    d.time_of_day,
    s.severity_name,
    COUNT(f.accident_key) AS accident_count,
    SUM(f.number_of_casualties) AS total_casualties,
    SUM(f.number_of_vehicles) AS total_vehicles
FROM fact_accidents f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_severity s ON f.severity_key = s.severity_key
GROUP BY d.day_name, d.day_of_week, d.hour, d.time_of_day, s.severity_name;

-- 2. Aggregated Mart: Weather and Road Condition Impact
DROP TABLE IF EXISTS agg_weather_road_risk_mart;
CREATE TABLE agg_weather_road_risk_mart AS
SELECT 
    w.weather_condition,
    w.weather_risk_level,
    r.road_type,
    r.road_surface_conditions,
    r.speed_limit,
    s.severity_name,
    COUNT(f.accident_key) AS accident_count,
    SUM(f.number_of_casualties) AS total_casualties,
    ROUND(AVG(f.number_of_casualties), 2) AS avg_casualties_per_accident
FROM fact_accidents f
JOIN dim_weather w ON f.weather_key = w.weather_key
JOIN dim_road r ON f.road_key = r.road_key
JOIN dim_severity s ON f.severity_key = s.severity_key
GROUP BY w.weather_condition, w.weather_risk_level, r.road_type, r.road_surface_conditions, r.speed_limit, s.severity_name;

-- 3. Aggregated Mart: Geospatial Hotspots (binned spatial clusters)
DROP TABLE IF EXISTS agg_location_hotspots_mart;
CREATE TABLE agg_location_hotspots_mart AS
SELECT 
    ROUND(l.latitude, 3) AS lat_binned,
    ROUND(l.longitude, 3) AS lon_binned,
    l.local_authority,
    l.urban_or_rural,
    COUNT(f.accident_key) AS accident_count,
    SUM(CASE WHEN s.severity_name = 'Fatal' THEN 1 ELSE 0 END) AS fatal_count,
    SUM(CASE WHEN s.severity_name = 'Serious' THEN 1 ELSE 0 END) AS serious_count,
    SUM(CASE WHEN s.severity_name = 'Slight' THEN 1 ELSE 0 END) AS slight_count,
    SUM(f.number_of_casualties) AS total_casualties
FROM fact_accidents f
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_severity s ON f.severity_key = s.severity_key
GROUP BY ROUND(l.latitude, 3), ROUND(l.longitude, 3), l.local_authority, l.urban_or_rural
HAVING COUNT(f.accident_key) >= 1;

-- 4. Model-ready View (Denormalized feature view for MLOps training & inference)
DROP VIEW IF EXISTS v_model_training_dataset;
CREATE VIEW v_model_training_dataset AS
SELECT 
    f.accident_index,
    d.full_date,
    d.year,
    d.month,
    d.day_of_week,
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
    w.weather_source,
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
JOIN dim_severity s ON f.severity_key = s.severity_key;

-- 04_postgis.sql
-- PostgreSQL / PostGIS only (skipped on the SQLite fallback).
-- Adds a WGS84 point geometry to dim_location for spatial queries such as
-- ST_DWithin hotspot searches, and keeps it in sync with latitude/longitude.

CREATE EXTENSION IF NOT EXISTS postgis;

ALTER TABLE dim_location ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);

CREATE INDEX IF NOT EXISTS idx_dim_location_geom ON dim_location USING GIST (geom);

UPDATE dim_location
SET geom = ST_SetSRID(ST_MakePoint(longitude::double precision, latitude::double precision), 4326)
WHERE geom IS NULL;

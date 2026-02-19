-- Query templates for Streamlit sidebar inputs.
-- Catalog/Schema: catalog_caiom7nmz_d9oink.renfe_app_data

-- Stations query
SELECT
  station_code,
  station_name,
  lat,
  lon,
  province,
  source
FROM catalog_caiom7nmz_d9oink.renfe_app_data.stations_dim;

-- Trip updates query
SELECT
  trip_id,
  delay_seconds,
  timestamp,
  feed
FROM catalog_caiom7nmz_d9oink.renfe_app_data.trip_updates_rt;

-- Vehicles query
SELECT
  trip_id,
  vehicle_id,
  vehicle_label,
  status,
  stop_id,
  lat,
  lon
FROM catalog_caiom7nmz_d9oink.renfe_app_data.vehicle_positions_rt;

-- Routes query
SELECT
  route_short_name
FROM catalog_caiom7nmz_d9oink.renfe_app_data.routes_dim;

-- Scheduled trips query
SELECT
  trip_id
FROM catalog_caiom7nmz_d9oink.renfe_app_data.trips_dim;

-- Alerts query
SELECT
  id,
  route_count,
  description
FROM catalog_caiom7nmz_d9oink.renfe_app_data.alerts_rt;

-- Incidents query (optional)
SELECT *
FROM catalog_caiom7nmz_d9oink.renfe_app_data.incidents;

-- Atendo query (optional)
SELECT *
FROM catalog_caiom7nmz_d9oink.renfe_app_data.atendo_accessibility;

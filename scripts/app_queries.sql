-- Query templates for Streamlit reference.
-- Catalog/Schema: catalog_caiom7nmz_d9oink.renfe_app_data

-- Stations query
SELECT
  codigo AS station_code,
  descripcion AS station_name,
  latitud AS lat,
  longitud AS lon,
  provincia,
  'estaciones' AS source
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_estaciones;

-- Trip updates query
SELECT
  entity_item.id AS entity_id,
  entity_item.tripUpdate.trip.tripId AS trip_id,
  COALESCE(entity_item.tripUpdate.delay, entity_item.tripUpdate.stopTimeUpdate[0].arrival.delay, 0) AS delay_seconds,
  to_timestamp(from_unixtime(try_cast(src.header.timestamp AS BIGINT))) AS timestamp,
  'Cercanias' AS feed
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_trip_updates src
LATERAL VIEW OUTER explode(src.entity) exp1 AS entity_item
UNION ALL
SELECT
  entity_ld_item.id AS entity_id,
  entity_ld_item.tripUpdate.trip.tripId AS trip_id,
  COALESCE(entity_ld_item.tripUpdate.delay, 0) AS delay_seconds,
  to_timestamp(from_unixtime(try_cast(src_ld.header.timestamp AS BIGINT))) AS timestamp,
  'LD' AS feed
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_trip_updates_ld src_ld
LATERAL VIEW OUTER explode(src_ld.entity) exp2 AS entity_ld_item;

-- Vehicles query
SELECT
  entity_item.vehicle.trip.tripId AS trip_id,
  entity_item.vehicle.vehicle.id AS vehicle_id,
  entity_item.vehicle.vehicle.label AS vehicle_label,
  entity_item.vehicle.currentStatus AS status,
  entity_item.vehicle.stopId AS stop_id,
  entity_item.vehicle.position.latitude AS lat,
  entity_item.vehicle.position.longitude AS lon
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_vehicle_positions src
LATERAL VIEW OUTER explode(src.entity) exp AS entity_item;

-- Routes query
SELECT
  route_id, route_short_name, route_type
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_google_transit_routes;

-- Scheduled trips query
SELECT
  trip_id, route_id, direction_id
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_google_transit_trips;

-- Alerts query
SELECT
  entity_item.id AS id,
  size(entity_item.alert.informedEntity) AS route_count,
  entity_item.alert.descriptionText.translation[0].text AS description
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_alerts src
LATERAL VIEW OUTER explode(src.entity) exp AS entity_item;

-- Incidents query (optional)
SELECT *
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_rfincidentreports_co_noticeresults;

-- Atendo query (optional)
SELECT *
FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_listado_de_estaciones_con_servicio_de_atendo;

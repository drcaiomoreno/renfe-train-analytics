from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Renfe Train Analytics",
    page_icon="🚆",
    layout="wide",
)

TRIP_QUERY_HINT = """
	SELECT
	  entity_item.id AS entity_id,
	  entity_item.tripUpdate.trip.tripId AS trip_id,
	  COALESCE(
	    entity_item.tripUpdate.delay,
	    entity_item.tripUpdate.stopTimeUpdate[0].arrival.delay,
	    0
	  ) AS delay_seconds,
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
	LATERAL VIEW OUTER explode(src_ld.entity) exp2 AS entity_ld_item
"""

GEO_QUERY_HINT = """
	SELECT codigo AS station_code, descripcion AS station_name, latitud AS lat, longitud AS lon, provincia, 'estaciones' AS source
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_estaciones
"""

AVLD_QUERY_HINT = """
	SELECT c_digo AS station_code, descripcion AS station_name, latitud AS lat, longitud AS lon, provincia, 'av_ld' AS source
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_listado_completo_av_ld_md
"""

FEVE_QUERY_HINT = """
	SELECT c_digo AS station_code, descripcion AS station_name, latitud AS lat, longitud AS lon, provincia, 'feve' AS source
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_listado_de_estaciones_feve_2
"""

CERCANIAS_QUERY_HINT = """
	SELECT c_digo AS station_code, descripcion AS station_name, latitud AS lat, longitud AS lon, provincia, 'cercanias_madrid' AS source
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_listado_estaciones_cercanias_madrid
"""

VEHICLES_QUERY_HINT = """
	SELECT
	  entity_item.id AS entity_id,
	  entity_item.vehicle.trip.tripId AS trip_id,
	  entity_item.vehicle.vehicle.id AS vehicle_id,
	  entity_item.vehicle.vehicle.label AS vehicle_label,
	  entity_item.vehicle.currentStatus AS status,
	  entity_item.vehicle.stopId AS stop_id,
	  entity_item.vehicle.position.latitude AS lat,
	  entity_item.vehicle.position.longitude AS lon
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_vehicle_positions src
	LATERAL VIEW OUTER explode(src.entity) exp AS entity_item
"""

ROUTES_QUERY_HINT = """
	SELECT route_id, route_short_name, route_type
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_google_transit_routes
"""

SCHEDULED_TRIPS_QUERY_HINT = """
	SELECT trip_id, route_id, direction_id
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_google_transit_trips
"""

ALERTS_QUERY_HINT = """
	SELECT
	  entity_item.id AS id,
	  size(entity_item.alert.informedEntity) AS route_count,
	  entity_item.alert.descriptionText.translation[0].text AS description
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_alerts src
	LATERAL VIEW OUTER explode(src.entity) exp AS entity_item
"""

INCIDENTS_QUERY_HINT = """
	SELECT *
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_json_rfincidentreports_co_noticeresults
"""

ATENDO_QUERY_HINT = """
	SELECT *
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_listado_de_estaciones_con_servicio_de_atendo
"""

TRAIN_INFO_QUERY_HINT = """
	SELECT *
	FROM catalog_caiom7nmz_d9oink.renfe_app_data.bronze_csv_informacion_trenes
"""

DEFAULT_GENIE_SPACE_ID = "01f10d0ef5311278a6829ada8a43e5b7"
DEFAULT_GENIE_SPACE_ID_2 = "01f10d2386d41be4a33a652f9e3cf521"
DEFAULT_UC_CATALOG = "catalog_caiom7nmz_d9oink"
DEFAULT_UC_SCHEMA = "renfe_app_data"


def _apply_app_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .dbx-header {
            background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 14px;
        }
        .dbx-title {color: #f8fafc; font-size: 1.6rem; font-weight: 700; margin: 0;}
        .dbx-subtitle {color: #cbd5e1; margin-top: 4px; margin-bottom: 0;}
        .dbx-user {
            display: inline-block;
            border: 1px solid #334155;
            border-radius: 999px;
            padding: 6px 12px;
            color: #e2e8f0;
            background: #111827;
            font-size: 0.9rem;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _pick_col(df: pd.DataFrame, options: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    return None


def _clean_query(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_server_hostname(host: str) -> str:
    value = (host or "").strip()
    if value.startswith("https://"):
        value = value[len("https://") :]
    elif value.startswith("http://"):
        value = value[len("http://") :]
    return value.split("/", 1)[0].strip()


def _default_server_hostname() -> str:
    return _normalize_server_hostname(
        os.getenv("DATABRICKS_SERVER_HOSTNAME", "") or os.getenv("DATABRICKS_HOST", "")
    )


def _default_http_path() -> str:
    explicit = os.getenv("DATABRICKS_HTTP_PATH", "").strip()
    if explicit:
        return explicit
    warehouse_id = os.getenv("WAREHOUSE_ID", "").strip()
    if warehouse_id:
        return f"/sql/1.0/warehouses/{warehouse_id}"
    return ""


def _has_app_oauth_credentials() -> bool:
    return bool(os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"))


def _default_genie_space_id() -> str:
    return (os.getenv("GENIE_SPACE_ID", "") or DEFAULT_GENIE_SPACE_ID).strip()


def _default_genie_space_id_2() -> str:
    return (os.getenv("GENIE_SPACE_ID_2", "") or DEFAULT_GENIE_SPACE_ID_2).strip()


@st.cache_data(show_spinner=False, ttl=300)
def _current_user_name() -> str:
    for env_key in ("DATABRICKS_USER", "DATABRICKS_USERNAME", "USER"):
        value = (os.getenv(env_key, "") or "").strip()
        if value:
            return value
    try:
        from databricks.sdk import WorkspaceClient

        me = WorkspaceClient().current_user.me()
        display_name = (getattr(me, "display_name", "") or "").strip()
        user_name = (getattr(me, "user_name", "") or "").strip()
        if display_name:
            return display_name
        if user_name:
            return user_name
    except Exception:
        pass
    return "Unknown User"


def _read_csv_flexible(path: Path, sep: str = ";") -> pd.DataFrame:
    encodings = ("utf-8-sig", "latin1", "cp1252")
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python", on_bad_lines="skip")
            df = _clean_columns(df)
            if len(df.columns) == 1 and ";" in df.columns[0]:
                df = pd.read_csv(
                    path,
                    sep=";",
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
                df = _clean_columns(df)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _read_csv_if_exists(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _clean_columns(pd.read_csv(path, **kwargs))
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _stations_from_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"])

    colmap = {c.upper(): c for c in df.columns}
    code_col = colmap.get("CODIGO") or colmap.get("CÓDIGO") or colmap.get("CODIGO DE ESTACIÓN")
    name_col = colmap.get("DESCRIPCION") or colmap.get("NOMBRE DE LA ESTACION")
    lat_col = colmap.get("LATITUD")
    lon_col = colmap.get("LONGITUD")
    prov_col = colmap.get("PROVINCIA")

    out = pd.DataFrame(
        {
            "station_code": df.get(code_col, pd.Series(dtype=str)).astype(str).str.strip(),
            "station_name": df.get(name_col, pd.Series(dtype=str)).astype(str).str.strip(),
            "lat": _to_float(df.get(lat_col, pd.Series(dtype=float))),
            "lon": _to_float(df.get(lon_col, pd.Series(dtype=float))),
            "province": df.get(prov_col, pd.Series(dtype=str)).astype(str).str.strip(),
            "source": source,
        }
    )
    return out.dropna(subset=["lat", "lon"]).query("lat != 0 and lon != 0")


@st.cache_data(show_spinner=False)
def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    base = Path(data_dir)

    estaciones = _read_csv_flexible(base / "estaciones.csv")
    av_ld = _read_csv_flexible(base / "listado_completo_av_ld_md.csv")
    cercanias = _read_csv_flexible(base / "listado-estaciones-cercanias-madrid.csv")
    feve = _read_csv_flexible(base / "listado-de-estaciones-feve-2.csv")
    atendo = _read_csv_flexible(base / "listado-de-estaciones-con-servicio-de-atendo.csv")

    station_frames = [
        _stations_from_df(estaciones, "General"),
        _stations_from_df(av_ld, "AV/LD"),
        _stations_from_df(cercanias, "Cercanias Madrid"),
        _stations_from_df(feve, "FEVE"),
    ]
    stations = pd.concat(station_frames, ignore_index=True).drop_duplicates(
        subset=["station_code", "lat", "lon", "source"]
    )

    gtfs_base = base / "google_transit"
    gtfs_stops = _read_csv_if_exists(gtfs_base / "stops.txt")
    gtfs_routes = _read_csv_if_exists(gtfs_base / "routes.txt")
    gtfs_trips = _read_csv_if_exists(gtfs_base / "trips.txt")

    if not gtfs_stops.empty:
        gtfs_stops["stop_lat"] = pd.to_numeric(gtfs_stops["stop_lat"], errors="coerce")
        gtfs_stops["stop_lon"] = pd.to_numeric(gtfs_stops["stop_lon"], errors="coerce")

    trip_updates = []
    for feed_name, file_name in [("Cercanias", "trip_updates.json"), ("LD", "trip_updates_LD.json")]:
        feed = _read_json(base / file_name)
        header = feed.get("header", {}) if isinstance(feed, dict) else {}
        feed_timestamp = pd.to_datetime(pd.to_numeric(header.get("timestamp", None), errors="coerce"), unit="s")
        for entity in feed.get("entity", []):
            update = entity.get("tripUpdate", {})
            delay = update.get("delay")
            if delay is None:
                stop_updates = update.get("stopTimeUpdate", [])
                if stop_updates:
                    arr = stop_updates[0].get("arrival", {})
                    dep = stop_updates[0].get("departure", {})
                    delay = arr.get("delay", dep.get("delay"))
            trip_updates.append(
                {
                    "feed": feed_name,
                    "entity_id": entity.get("id"),
                    "trip_id": update.get("trip", {}).get("tripId"),
                    "delay_seconds": pd.to_numeric(delay, errors="coerce"),
                    "timestamp": feed_timestamp,
                }
            )
    trip_updates_df = pd.DataFrame(trip_updates)

    vp = _read_json(base / "vehicle_positions.json")
    vehicles = []
    for entity in vp.get("entity", []):
        vehicle = entity.get("vehicle", {})
        position = vehicle.get("position", {})
        vehicles.append(
            {
                "entity_id": entity.get("id"),
                "trip_id": vehicle.get("trip", {}).get("tripId"),
                "vehicle_id": vehicle.get("vehicle", {}).get("id"),
                "vehicle_label": vehicle.get("vehicle", {}).get("label"),
                "status": vehicle.get("currentStatus"),
                "stop_id": vehicle.get("stopId"),
                "lat": pd.to_numeric(position.get("latitude"), errors="coerce"),
                "lon": pd.to_numeric(position.get("longitude"), errors="coerce"),
            }
        )
    vehicles_df = pd.DataFrame(vehicles).dropna(subset=["lat", "lon"])

    alerts_json = _read_json(base / "alerts.json")
    alerts = []
    for entity in alerts_json.get("entity", []):
        alert = entity.get("alert", {})
        desc = ""
        translations = alert.get("descriptionText", {}).get("translation", [])
        if translations:
            desc = translations[0].get("text", "")
        alerts.append(
            {
                "id": entity.get("id"),
                "route_count": len(alert.get("informedEntity", [])),
                "description": desc,
            }
        )
    alerts_df = pd.DataFrame(alerts)

    incidents_json = _read_json(base / "rfincidentreports_co.noticeresults.json")
    incidents_df = pd.DataFrame(incidents_json if isinstance(incidents_json, list) else [])

    atendo_df = atendo.copy()
    if not atendo_df.empty and len(atendo_df.columns) == 1:
        atendo_df = _read_csv_flexible(base / "listado-de-estaciones-con-servicio-de-atendo.csv")

    return {
        "stations": stations,
        "gtfs_stops": gtfs_stops,
        "gtfs_routes": gtfs_routes,
        "gtfs_trips": gtfs_trips,
        "trip_updates": trip_updates_df,
        "vehicles": vehicles_df,
        "alerts": alerts_df,
        "incidents": incidents_df,
        "atendo": atendo_df,
    }


@st.cache_data(show_spinner=False, ttl=120)
def run_warehouse_query(server_hostname: str, http_path: str, access_token: str, query: str) -> pd.DataFrame:
    from databricks import sql as dbsql
    from databricks.sdk.core import Config

    clean_host = _normalize_server_hostname(server_hostname)
    clean_http_path = (http_path or "").strip()
    token = (access_token or "").strip()

    connect_kwargs: dict[str, Any] = {
        "server_hostname": clean_host,
        "http_path": clean_http_path,
    }
    if token:
        connect_kwargs["access_token"] = token
    else:
        cfg = Config()
        connect_kwargs["server_hostname"] = clean_host or _normalize_server_hostname(cfg.host)
        connect_kwargs["credentials_provider"] = lambda: cfg.authenticate

    with dbsql.connect(**connect_kwargs) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            if cursor.description is None:
                return pd.DataFrame()
            rows = cursor.fetchall()
            columns = [c[0] for c in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def normalize_trip_updates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["feed", "entity_id", "trip_id", "delay_seconds", "timestamp"])

    trip_col = _pick_col(df, ["trip_id", "tripid", "id"])
    delay_col = _pick_col(df, ["delay_seconds", "delay_sec", "delay", "delay_min", "delay_minutes"])
    timestamp_col = _pick_col(df, ["timestamp", "event_time", "ts", "datetime"])
    feed_col = _pick_col(df, ["feed", "service", "line"])

    if not trip_col or not delay_col:
        return pd.DataFrame(columns=["feed", "entity_id", "trip_id", "delay_seconds", "timestamp"])

    out = pd.DataFrame(
        {
            "feed": df[feed_col].astype(str) if feed_col else "Warehouse",
            "entity_id": pd.NA,
            "trip_id": df[trip_col].astype(str),
            "delay_seconds": pd.to_numeric(df[delay_col], errors="coerce"),
            "timestamp": pd.to_datetime(df[timestamp_col], errors="coerce") if timestamp_col else pd.NaT,
        }
    )

    if "min" in delay_col.lower():
        out["delay_seconds"] = out["delay_seconds"] * 60
    return out.dropna(subset=["delay_seconds"])


def normalize_geo_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"])

    lat_col = _pick_col(df, ["lat", "latitude", "stop_lat", "latitud"])
    lon_col = _pick_col(df, ["lon", "lng", "longitude", "stop_lon", "longitud"])
    if not lat_col or not lon_col:
        return pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"])

    code_col = _pick_col(df, ["station_code", "stop_id", "id", "code", "codigo", "codigo_de_estacion"])
    name_col = _pick_col(df, ["station_name", "stop_name", "name", "point_name", "descripcion", "nombre_de_la_estacion"])
    province_col = _pick_col(df, ["province", "region", "state", "provincia"])
    source_col = _pick_col(df, ["source", "feed", "dataset"])

    out = pd.DataFrame(
        {
            "station_code": df[code_col].astype(str) if code_col else "",
            "station_name": df[name_col].astype(str) if name_col else "Warehouse point",
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "province": df[province_col].astype(str) if province_col else "",
            "source": df[source_col].astype(str) if source_col else "Warehouse SQL",
        }
    )
    return out.dropna(subset=["lat", "lon"])


def normalize_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["entity_id", "trip_id", "vehicle_id", "vehicle_label", "status", "stop_id", "lat", "lon"])

    lat_col = _pick_col(df, ["lat", "latitude"])
    lon_col = _pick_col(df, ["lon", "lng", "longitude"])
    if not lat_col or not lon_col:
        return pd.DataFrame(columns=["entity_id", "trip_id", "vehicle_id", "vehicle_label", "status", "stop_id", "lat", "lon"])

    entity_col = _pick_col(df, ["entity_id", "id"])
    trip_col = _pick_col(df, ["trip_id", "tripid"])
    vehicle_id_col = _pick_col(df, ["vehicle_id", "id_vehiculo", "veh_id"])
    vehicle_label_col = _pick_col(df, ["vehicle_label", "label", "name"])
    status_col = _pick_col(df, ["status", "current_status"])
    stop_col = _pick_col(df, ["stop_id", "station_code"])

    out = pd.DataFrame(
        {
            "entity_id": df[entity_col].astype(str) if entity_col else "",
            "trip_id": df[trip_col].astype(str) if trip_col else "",
            "vehicle_id": df[vehicle_id_col].astype(str) if vehicle_id_col else "",
            "vehicle_label": df[vehicle_label_col].astype(str) if vehicle_label_col else "",
            "status": df[status_col].astype(str) if status_col else "",
            "stop_id": df[stop_col].astype(str) if stop_col else "",
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["lat", "lon"])


def normalize_routes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["route_id", "route_short_name", "route_type"])
    route_id_col = _pick_col(df, ["route_id", "id"])
    route_name_col = _pick_col(df, ["route_short_name", "service", "route_name", "line"])
    route_type_col = _pick_col(df, ["route_type", "type"])

    out = pd.DataFrame(
        {
            "route_id": df[route_id_col].astype(str) if route_id_col else "",
            "route_short_name": df[route_name_col].astype(str) if route_name_col else "Unknown",
            "route_type": df[route_type_col].astype(str) if route_type_col else "Unknown",
        }
    )
    return out


def normalize_scheduled_trips(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["trip_id", "route_id", "direction_id"])
    trip_col = _pick_col(df, ["trip_id", "tripid", "id"])
    route_id_col = _pick_col(df, ["route_id"])
    direction_col = _pick_col(df, ["direction_id"])
    if not trip_col:
        trip_col = df.columns[0]
    return pd.DataFrame(
        {
            "trip_id": df[trip_col].astype(str),
            "route_id": df[route_id_col].astype(str) if route_id_col else "",
            "direction_id": pd.to_numeric(df[direction_col], errors="coerce").fillna(0).astype(int)
            if direction_col
            else 0,
        }
    )


def normalize_alerts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "route_count", "description"])
    id_col = _pick_col(df, ["id", "alert_id"])
    route_count_col = _pick_col(df, ["route_count", "affected_routes"])
    desc_col = _pick_col(df, ["description", "text", "message"])

    out = pd.DataFrame(
        {
            "id": df[id_col].astype(str) if id_col else "",
            "route_count": pd.to_numeric(df[route_count_col], errors="coerce").fillna(0).astype(int)
            if route_count_col
            else 0,
            "description": df[desc_col].astype(str) if desc_col else "",
        }
    )
    return out


def empty_data() -> dict[str, pd.DataFrame]:
    return {
        "stations": pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"]),
        "avld": pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"]),
        "feve": pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"]),
        "cercanias_madrid": pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"]),
        "gtfs_stops": pd.DataFrame(),
        "gtfs_routes": pd.DataFrame(columns=["route_short_name"]),
        "gtfs_trips": pd.DataFrame(columns=["trip_id"]),
        "trip_updates": pd.DataFrame(columns=["feed", "entity_id", "trip_id", "delay_seconds", "timestamp"]),
        "vehicles": pd.DataFrame(columns=["entity_id", "trip_id", "vehicle_id", "vehicle_label", "status", "stop_id", "lat", "lon"]),
        "alerts": pd.DataFrame(columns=["id", "route_count", "description"]),
        "incidents": pd.DataFrame(),
        "atendo": pd.DataFrame(),
        "train_info": pd.DataFrame(),
    }


def load_data_from_warehouse(
    server_hostname: str,
    http_path: str,
    access_token: str,
    query_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    data = empty_data()

    stations_sql = _clean_query(query_map.get("stations", ""))
    trips_sql = _clean_query(query_map.get("trip_updates", ""))
    vehicles_sql = _clean_query(query_map.get("vehicles", ""))
    routes_sql = _clean_query(query_map.get("routes", ""))
    scheduled_trips_sql = _clean_query(query_map.get("scheduled_trips", ""))
    alerts_sql = _clean_query(query_map.get("alerts", ""))
    incidents_sql = _clean_query(query_map.get("incidents", ""))
    atendo_sql = _clean_query(query_map.get("atendo", ""))
    avld_sql = _clean_query(query_map.get("avld", ""))
    feve_sql = _clean_query(query_map.get("feve", ""))
    cercanias_sql = _clean_query(query_map.get("cercanias_madrid", ""))
    train_info_sql = _clean_query(query_map.get("train_info", ""))

    def _is_table_not_found_error(exc: Exception) -> bool:
        return "TABLE_OR_VIEW_NOT_FOUND" in str(exc)

    if stations_sql:
        try:
            data["stations"] = normalize_geo_points(run_warehouse_query(server_hostname, http_path, access_token, stations_sql))
        except Exception as exc:
            msg = str(exc).lower()
            if _is_table_not_found_error(exc) and "stations_dim" in msg:
                fallback_sql = f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_csv_estaciones"
                data["stations"] = normalize_geo_points(
                    run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                )
            else:
                raise
    if trips_sql:
        try:
            data["trip_updates"] = normalize_trip_updates(run_warehouse_query(server_hostname, http_path, access_token, trips_sql))
        except Exception as exc:
            if _is_table_not_found_error(exc) and "trip_updates_rt" in str(exc).lower():
                fallback_sql = _clean_query(
                    f"""
                    SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_json_trip_updates
                    UNION ALL
                    SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_json_trip_updates_ld
                    """
                )
                try:
                    data["trip_updates"] = normalize_trip_updates(
                        run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                    )
                except Exception:
                    data["trip_updates"] = pd.DataFrame(
                        columns=["feed", "entity_id", "trip_id", "delay_seconds", "timestamp"]
                    )
            else:
                raise
    if vehicles_sql:
        try:
            data["vehicles"] = normalize_vehicles(
                run_warehouse_query(server_hostname, http_path, access_token, vehicles_sql)
            )
        except Exception as exc:
            if _is_table_not_found_error(exc) and "vehicle_positions_rt" in str(exc).lower():
                fallback_sql = f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_json_vehicle_positions"
                try:
                    data["vehicles"] = normalize_vehicles(
                        run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                    )
                except Exception:
                    data["vehicles"] = pd.DataFrame(
                        columns=["entity_id", "trip_id", "vehicle_id", "vehicle_label", "status", "stop_id", "lat", "lon"]
                    )
            else:
                raise
    if routes_sql:
        try:
            data["gtfs_routes"] = normalize_routes(
                run_warehouse_query(server_hostname, http_path, access_token, routes_sql)
            )
        except Exception as exc:
            if _is_table_not_found_error(exc) and "routes_dim" in str(exc).lower():
                fallback_candidates = [
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_google_transit_routes",
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_fomento_transit_routes",
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_horarios_feve_routes",
                ]
                loaded = False
                for fallback_sql in fallback_candidates:
                    try:
                        data["gtfs_routes"] = normalize_routes(
                            run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                        )
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    data["gtfs_routes"] = pd.DataFrame(columns=["route_id", "route_short_name", "route_type"])
            else:
                raise
    if scheduled_trips_sql:
        try:
            data["gtfs_trips"] = normalize_scheduled_trips(
                run_warehouse_query(server_hostname, http_path, access_token, scheduled_trips_sql)
            )
        except Exception as exc:
            if _is_table_not_found_error(exc) and "trips_dim" in str(exc).lower():
                fallback_candidates = [
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_google_transit_trips",
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_fomento_transit_trips",
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_horarios_feve_trips",
                ]
                loaded = False
                for fallback_sql in fallback_candidates:
                    try:
                        data["gtfs_trips"] = normalize_scheduled_trips(
                            run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                        )
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    data["gtfs_trips"] = pd.DataFrame(columns=["trip_id", "route_id", "direction_id"])
            else:
                raise
    if alerts_sql:
        try:
            data["alerts"] = normalize_alerts(
                run_warehouse_query(server_hostname, http_path, access_token, alerts_sql)
            )
        except Exception as exc:
            if _is_table_not_found_error(exc) and "alerts_rt" in str(exc).lower():
                fallback_sql = f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_json_alerts"
                try:
                    data["alerts"] = normalize_alerts(
                        run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                    )
                except Exception:
                    data["alerts"] = pd.DataFrame(columns=["id", "route_count", "description"])
            else:
                raise
    if incidents_sql:
        try:
            data["incidents"] = run_warehouse_query(server_hostname, http_path, access_token, incidents_sql)
        except Exception as exc:
            if _is_table_not_found_error(exc) and "incidents" in str(exc).lower():
                fallback_sql = f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_json_rfincidentreports_co_noticeresults"
                try:
                    data["incidents"] = run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                except Exception:
                    data["incidents"] = pd.DataFrame()
            else:
                raise
    if atendo_sql:
        try:
            data["atendo"] = run_warehouse_query(server_hostname, http_path, access_token, atendo_sql)
        except Exception as exc:
            if _is_table_not_found_error(exc) and "atendo_accessibility" in str(exc).lower():
                fallback_sql = (
                    f"SELECT * FROM {DEFAULT_UC_CATALOG}.{DEFAULT_UC_SCHEMA}.bronze_csv_listado_de_estaciones_con_servicio_de_atendo"
                )
                try:
                    data["atendo"] = run_warehouse_query(server_hostname, http_path, access_token, fallback_sql)
                except Exception:
                    data["atendo"] = pd.DataFrame()
            else:
                raise
    if avld_sql:
        data["avld"] = normalize_geo_points(run_warehouse_query(server_hostname, http_path, access_token, avld_sql))
    if feve_sql:
        data["feve"] = normalize_geo_points(run_warehouse_query(server_hostname, http_path, access_token, feve_sql))
    if cercanias_sql:
        data["cercanias_madrid"] = normalize_geo_points(
            run_warehouse_query(server_hostname, http_path, access_token, cercanias_sql)
        )
    if train_info_sql:
        data["train_info"] = run_warehouse_query(server_hostname, http_path, access_token, train_info_sql)

    return data


@st.cache_data(show_spinner=False, ttl=300)
def load_table_inventory(server_hostname: str, http_path: str, access_token: str) -> pd.DataFrame:
    sql = f"""
    SELECT table_name, table_type, data_source_format
    FROM {DEFAULT_UC_CATALOG}.information_schema.tables
    WHERE table_schema = '{DEFAULT_UC_SCHEMA}'
    ORDER BY table_name
    """
    inv = run_warehouse_query(server_hostname, http_path, access_token, _clean_query(sql))
    if inv.empty:
        return pd.DataFrame(columns=["table_name", "table_type", "data_source_format", "layer", "domain"])
    out = inv.copy()
    out["layer"] = np.where(out["table_name"].str.startswith("bronze_"), "Bronze", "Curated")
    out["domain"] = out["table_name"].str.replace("bronze_", "", regex=False).str.split("_").str[0]
    return out


def apply_business_filters(
    data: dict[str, pd.DataFrame],
    provinces: list[str],
    feeds: list[str],
    route_types: list[str],
    vehicle_statuses: list[str],
    delay_min_range: tuple[int, int],
) -> dict[str, pd.DataFrame]:
    filtered = {k: v.copy() for k, v in data.items()}

    if provinces and not filtered["stations"].empty and "province" in filtered["stations"].columns:
        filtered["stations"] = filtered["stations"][filtered["stations"]["province"].astype(str).isin(provinces)]
    if provinces and not filtered["avld"].empty and "province" in filtered["avld"].columns:
        filtered["avld"] = filtered["avld"][filtered["avld"]["province"].astype(str).isin(provinces)]

    if feeds and not filtered["trip_updates"].empty and "feed" in filtered["trip_updates"].columns:
        filtered["trip_updates"] = filtered["trip_updates"][filtered["trip_updates"]["feed"].astype(str).isin(feeds)]

    if not filtered["trip_updates"].empty and "delay_seconds" in filtered["trip_updates"].columns:
        lo, hi = delay_min_range
        mins = pd.to_numeric(filtered["trip_updates"]["delay_seconds"], errors="coerce") / 60.0
        filtered["trip_updates"] = filtered["trip_updates"][(mins >= lo) & (mins <= hi)]

    if route_types and not filtered["gtfs_routes"].empty and "route_type" in filtered["gtfs_routes"].columns:
        filtered["gtfs_routes"] = filtered["gtfs_routes"][
            filtered["gtfs_routes"]["route_type"].astype(str).isin(route_types)
        ]

    if vehicle_statuses and not filtered["vehicles"].empty and "status" in filtered["vehicles"].columns:
        filtered["vehicles"] = filtered["vehicles"][filtered["vehicles"]["status"].astype(str).isin(vehicle_statuses)]

    return filtered


def build_station_master(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key in ("stations", "avld", "feve", "cercanias_madrid"):
        df = data.get(key, pd.DataFrame())
        if df.empty:
            continue
        keep_cols = [c for c in ["station_code", "station_name", "lat", "lon", "province", "source"] if c in df.columns]
        if keep_cols:
            frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"])
    out = pd.concat(frames, ignore_index=True)
    out["station_code"] = out["station_code"].astype(str).str.strip()
    out["station_name"] = out["station_name"].astype(str).str.strip()
    out = out.dropna(subset=["lat", "lon"])
    return out.drop_duplicates(subset=["station_code", "station_name", "lat", "lon", "source"])


def render_csv_overview(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Renfe CSV Data Overview")
    stations = build_station_master(data)
    atendo = data["atendo"]
    train_info = data["train_info"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations", f"{stations['station_code'].nunique():,}" if not stations.empty else "0")
    c2.metric("Provinces", f"{stations['province'].astype(str).nunique():,}" if not stations.empty else "0")
    c3.metric("Atendo Stations", f"{len(atendo):,}")
    c4.metric("Train Models", f"{train_info.get('modelo', pd.Series(dtype=str)).astype(str).nunique():,}" if not train_info.empty else "0")

    left, right = st.columns(2)
    with left:
        if not stations.empty and "province" in stations.columns:
            by_prov = (
                stations.assign(province=stations["province"].astype(str).replace("", "Unknown"))
                .groupby("province", as_index=False)
                .size()
                .rename(columns={"size": "stations"})
                .sort_values("stations", ascending=False)
                .head(20)
            )
            st.plotly_chart(px.bar(by_prov, x="province", y="stations", title="Top Provinces by Stations"), use_container_width=True)
        else:
            st.info("No station data available.")
    with right:
        if not train_info.empty and "constructor" in train_info.columns:
            by_builder = (
                train_info.assign(constructor=train_info["constructor"].astype(str).replace("", "Unknown"))
                .groupby("constructor", as_index=False)
                .size()
                .rename(columns={"size": "models"})
                .sort_values("models", ascending=False)
                .head(15)
            )
            st.plotly_chart(px.bar(by_builder, x="constructor", y="models", title="Train Models by Constructor"), use_container_width=True)
        else:
            st.info("No train model data available.")


def render_csv_map(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Stations Map")
    stations = build_station_master(data)
    if stations.empty:
        st.info("No stations with coordinates to display.")
        return
    stations = stations.copy()
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.dropna(subset=["lat", "lon"])
    # Keep map centered on Spain mainland + islands bounds.
    stations = stations[
        (stations["lat"] >= 27.0)
        & (stations["lat"] <= 44.5)
        & (stations["lon"] >= -19.5)
        & (stations["lon"] <= 5.5)
    ]
    if stations.empty:
        st.info("No station coordinates inside Spain bounds.")
        return
    view_state = pdk.ViewState(
        latitude=float(stations["lat"].astype(float).mean()),
        longitude=float(stations["lon"].astype(float).mean()),
        zoom=5.2,
        pitch=0,
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=stations,
        get_position="[lon, lat]",
        get_radius=5000,
        get_fill_color="[255, 140, 0, 140]",
        pickable=True,
    )
    tooltip = {"html": "<b>{station_name}</b><br/>Code: {station_code}<br/>Province: {province}<br/>Source: {source}"}
    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style="light",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        )
    )


def render_csv_accessibility(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Atendo Accessibility")
    atendo = data["atendo"].copy()
    if atendo.empty:
        st.info("No Atendo records available.")
        return
    yes_cols = [c for c in ["aparcamiento", "and_n", "aseos", "vest_bulo"] if c in atendo.columns]
    if yes_cols:
        scores = []
        for c in yes_cols:
            val = atendo[c].astype(str).str.lower().str.contains("si|yes|x|1", regex=True, na=False).sum()
            scores.append({"feature": c, "stations": int(val)})
        feat_df = pd.DataFrame(scores).sort_values("stations", ascending=False)
        st.plotly_chart(px.bar(feat_df, x="feature", y="stations", title="Accessibility Features Coverage"), use_container_width=True)
    delay_col = "tiempo_de_antelacion" if "tiempo_de_antelacion" in atendo.columns else None
    if delay_col:
        dist = atendo[delay_col].astype(str).replace("", "Unknown").value_counts().reset_index(name="count")
        dist.columns = ["lead_time", "count"]
        st.plotly_chart(px.pie(dist, names="lead_time", values="count", title="Atendo Lead Time Distribution"), use_container_width=True)


def render_csv_fleet(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Train Fleet & Specs")
    df = data["train_info"].copy()
    if df.empty:
        st.info("No train information data available.")
        return
    left, right = st.columns(2)
    with left:
        if "velocidad_maxima" in df.columns:
            speeds = pd.to_numeric(
                df["velocidad_maxima"].astype(str).str.extract(r"(\d+[\.,]?\d*)")[0].str.replace(",", ".", regex=False),
                errors="coerce",
            )
            sp = pd.DataFrame({"speed": speeds}).dropna()
            if not sp.empty:
                st.plotly_chart(px.histogram(sp, x="speed", nbins=20, title="Max Speed Distribution"), use_container_width=True)
    with right:
        if "proposito" in df.columns:
            purpose = (
                df["proposito"].astype(str).replace("", "Unknown").value_counts().head(15).reset_index(name="models")
            )
            purpose.columns = ["purpose", "models"]
            st.plotly_chart(px.bar(purpose, x="purpose", y="models", title="Train Purpose Mix"), use_container_width=True)
    show_cols = [c for c in ["modelo", "constructor", "velocidad_maxima", "plazas_sentadas", "proposito"] if c in df.columns]
    if show_cols:
        st.dataframe(df[show_cols].head(100), use_container_width=True, hide_index=True)


def _statement_response_to_df(statement_response: Any) -> pd.DataFrame:
    if statement_response is None or statement_response.manifest is None:
        return pd.DataFrame()
    schema = statement_response.manifest.schema
    columns = [c.name or f"col_{i}" for i, c in enumerate(schema.columns or [])]
    data_array = statement_response.result.data_array if statement_response.result else []
    return pd.DataFrame(data_array or [], columns=columns)


def _run_genie_prompt(
    space_id: str,
    prompt: str,
    conversation_id: str | None = None,
    timeout_seconds: int = 60,
) -> tuple[pd.DataFrame, str | None, str]:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    timeout = datetime.timedelta(seconds=timeout_seconds)
    if conversation_id:
        msg = w.genie.create_message_and_wait(space_id=space_id, conversation_id=conversation_id, content=prompt, timeout=timeout)
    else:
        msg = w.genie.start_conversation_and_wait(space_id=space_id, content=prompt, timeout=timeout)

    conv_id = getattr(msg, "conversation_id", None)
    message_id = getattr(msg, "id", None) or getattr(msg, "message_id", None)
    if not conv_id or not message_id:
        return pd.DataFrame(), conv_id, msg.content or ""

    response = None
    for attachment in msg.attachments or []:
        attachment_id = (
            getattr(attachment, "attachment_id", None)
            or getattr(attachment, "id", None)
            or getattr(getattr(attachment, "query", None), "id", None)
        )
        if getattr(attachment, "query", None) and attachment_id:
            response = w.genie.get_message_query_result_by_attachment(
                space_id=space_id,
                conversation_id=conv_id,
                message_id=message_id,
                attachment_id=attachment_id,
            )
            break
    if response is None:
        response = w.genie.get_message_query_result(
            space_id=space_id,
            conversation_id=conv_id,
            message_id=message_id,
        )
    return _statement_response_to_df(response.statement_response), conv_id, msg.content or ""


def _run_genie_chat_prompt(
    space_id: str,
    prompt: str,
    conversation_id: str | None = None,
    timeout_seconds: int = 60,
) -> tuple[str, str | None]:
    from databricks.sdk import WorkspaceClient

    def _extract_text(msg: Any) -> str:
        parts: list[str] = []
        content = (getattr(msg, "content", None) or "").strip()
        if content:
            parts.append(content)
        for attachment in getattr(msg, "attachments", None) or []:
            text_attachment = getattr(attachment, "text", None)
            text_content = (getattr(text_attachment, "content", None) or "").strip()
            if text_content:
                parts.append(text_content)
        return "\n\n".join(parts).strip()

    w = WorkspaceClient()
    timeout = datetime.timedelta(seconds=timeout_seconds)
    if conversation_id:
        msg = w.genie.create_message_and_wait(space_id=space_id, conversation_id=conversation_id, content=prompt, timeout=timeout)
    else:
        msg = w.genie.start_conversation_and_wait(space_id=space_id, content=prompt, timeout=timeout)

    conv_id = getattr(msg, "conversation_id", None)
    response_text = _extract_text(msg)

    # Some Genie SDK calls return the submitted/user message first. Fetch latest conversation messages
    # and pick the newest non-empty response that differs from the prompt.
    if conv_id and (not response_text or response_text.strip() == prompt.strip()):
        try:
            messages_resp = w.genie.list_conversation_messages(space_id=space_id, conversation_id=conv_id, page_size=20)
            messages = messages_resp.messages or []
            messages_sorted = sorted(
                messages,
                key=lambda m: getattr(m, "created_timestamp", 0) or 0,
                reverse=True,
            )
            for m in messages_sorted:
                candidate = _extract_text(m)
                if candidate and candidate.strip() != prompt.strip():
                    response_text = candidate
                    break
        except Exception:
            pass

    return response_text.strip(), conv_id


def load_data_from_genie(space_id: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    data = empty_data()
    logs: list[str] = []
    conv_id: str | None = None
    prompts = [
        (
            "stations",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.stations_dim "
            "with columns station_code, station_name, lat, lon, province, source.",
            normalize_geo_points,
        ),
        (
            "trip_updates",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.trip_updates_rt "
            "with columns trip_id, delay_seconds, timestamp, feed.",
            normalize_trip_updates,
        ),
        (
            "vehicles",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.vehicle_positions_rt "
            "with columns trip_id, vehicle_id, vehicle_label, status, stop_id, lat, lon.",
            normalize_vehicles,
        ),
        (
            "gtfs_routes",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.routes_dim with column route_short_name.",
            normalize_routes,
        ),
        (
            "gtfs_trips",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.trips_dim with column trip_id.",
            normalize_scheduled_trips,
        ),
        (
            "alerts",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.alerts_rt "
            "with columns id, route_count, description.",
            normalize_alerts,
        ),
        (
            "incidents",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.incidents.",
            lambda df: df,
        ),
        (
            "atendo",
            "Return all rows from catalog_caiom7nmz_d9oink.renfe_app_data.atendo_accessibility.",
            lambda df: df,
        ),
    ]

    for key, prompt, normalizer in prompts:
        df, conv_id, message_text = _run_genie_prompt(space_id=space_id, prompt=prompt, conversation_id=conv_id)
        data[key] = normalizer(df)
        logs.append(f"{key}: {len(data[key]):,} rows")
        if message_text:
            logs.append(f"Genie ({key}): {message_text[:140]}")
    return data, logs


def render_genie_tab(space_id: str, title: str, key_prefix: str) -> None:
    st.subheader(title)

    conv_key = f"{key_prefix}_conversation_id"
    history_key = f"{key_prefix}_history"
    input_key = f"{key_prefix}_input"
    send_key = f"{key_prefix}_send"
    clear_key = f"{key_prefix}_clear"

    if conv_key not in st.session_state:
        st.session_state[conv_key] = None
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.text_input(
            "Ask Genie",
            placeholder="Example: Show top 10 delayed trips",
            key=input_key,
        )
    with col2:
        send = st.button("Send", use_container_width=True, key=send_key)

    if send:
        if not space_id:
            st.error("Genie is not configured for this environment.")
        elif not prompt.strip():
            st.warning("Enter a question for Genie.")
        else:
            try:
                with st.spinner("Querying Genie..."):
                    response_text, conv_id = _run_genie_chat_prompt(
                        space_id=space_id,
                        prompt=prompt.strip(),
                        conversation_id=st.session_state[conv_key],
                    )
                st.session_state[conv_key] = conv_id
                st.session_state[history_key].append(
                    {
                        "prompt": prompt.strip(),
                        "response": response_text,
                    }
                )
            except Exception as exc:
                st.error(str(exc))

    if st.session_state[history_key]:
        if st.button("Clear conversation", key=clear_key):
            st.session_state[conv_key] = None
            st.session_state[history_key] = []
            st.rerun()

    for i, item in enumerate(reversed(st.session_state[history_key]), start=1):
        st.markdown(f"**Q{i}:** {item['prompt']}")
        if item["response"]:
            st.write(item["response"])
        else:
            st.caption("No response returned yet. Try asking again or rephrasing.")


def _metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def render_insights(data: dict[str, pd.DataFrame]) -> None:
    stations = data["stations"]
    trip_updates = data["trip_updates"]
    vehicles = data["vehicles"]
    alerts = data["alerts"]
    incidents = data["incidents"]
    gtfs_routes = data["gtfs_routes"]
    gtfs_trips = data["gtfs_trips"]

    delayed = trip_updates["delay_seconds"].fillna(0)
    delayed_count = int((delayed > 0).sum()) if not trip_updates.empty else 0
    avg_delay_min = (delayed[delayed > 0].mean() / 60) if delayed_count > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _metric("Stations", f"{len(stations):,}")
    with c2:
        _metric("Routes", f"{len(gtfs_routes):,}")
    with c3:
        _metric("Scheduled Trips", f"{len(gtfs_trips):,}")
    with c4:
        _metric("Delayed Trips", f"{delayed_count:,}")
    with c5:
        _metric("Avg Delay", f"{avg_delay_min:.1f} min")
    with c6:
        _metric("Active Vehicles", f"{len(vehicles):,}")

    st.caption(
        f"Active Alerts: {len(alerts):,} | Published Incidents: {len(incidents):,}"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        if not trip_updates.empty:
            plot_df = trip_updates.assign(delay_min=trip_updates["delay_seconds"].fillna(0) / 60)
            fig = px.histogram(
                plot_df,
                x="delay_min",
                nbins=30,
                color="feed",
                title="Delay Distribution (minutes)",
            )
            fig.update_layout(xaxis_title="Delay (min)", yaxis_title="Trips")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trip update data available.")

    with col_right:
        if not stations.empty:
            top_provinces = (
                stations.groupby("province", dropna=False)
                .size()
                .reset_index(name="stations")
                .sort_values("stations", ascending=False)
                .head(15)
            )
            fig = px.bar(
                top_provinces,
                x="stations",
                y="province",
                orientation="h",
                title="Top Provinces by Station Count",
            )
            fig.update_layout(yaxis_title="", xaxis_title="Stations")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No station data available.")


def render_map(data: dict[str, pd.DataFrame]) -> None:
    stations = data["stations"]
    vehicles = data["vehicles"]

    st.subheader("Network Map")
    source_options = sorted(stations["source"].dropna().unique().tolist()) if not stations.empty else []
    selected_sources = st.multiselect("Station layers", source_options, default=source_options)

    map_stations = stations[stations["source"].isin(selected_sources)] if selected_sources else stations.iloc[0:0]
    map_vehicles = vehicles.copy()

    layers = []
    if not map_stations.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=map_stations,
                get_position="[lon, lat]",
                get_radius=4500,
                get_fill_color=[31, 119, 180, 140],
                pickable=True,
            )
        )
    if not map_vehicles.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=map_vehicles,
                get_position="[lon, lat]",
                get_radius=7000,
                get_fill_color=[214, 39, 40, 220],
                pickable=True,
            )
        )

    if layers:
        center_lat = float(pd.concat([map_stations["lat"], map_vehicles["lat"]], ignore_index=True).dropna().mean())
        center_lon = float(pd.concat([map_stations["lon"], map_vehicles["lon"]], ignore_index=True).dropna().mean())
        st.pydeck_chart(
            pdk.Deck(
                map_style="light",
                initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5, pitch=0),
                layers=layers,
                tooltip={
                    "html": "<b>{station_name}</b><br/>{province}<br/>{source}<br/>{vehicle_label}",
                    "style": {"color": "white"},
                },
            ),
            use_container_width=True,
        )
    else:
        st.info("No geospatial points available to render.")


def render_dashboards(data: dict[str, pd.DataFrame]) -> None:
    trip_updates = data["trip_updates"]
    gtfs_routes = data["gtfs_routes"]
    alerts = data["alerts"]
    atendo = data["atendo"]

    left, right = st.columns(2)
    with left:
        if not trip_updates.empty:
            top_delays = (
                trip_updates.sort_values("delay_seconds", ascending=False)
                .head(20)
                .assign(delay_min=lambda df: (df["delay_seconds"] / 60).round(1))
                [["feed", "trip_id", "delay_min", "timestamp"]]
            )
            st.subheader("Top Delayed Trips")
            st.dataframe(top_delays, use_container_width=True, hide_index=True)
        else:
            st.info("No delay leaderboard available.")

    with right:
        if not gtfs_routes.empty:
            route_mix = (
                gtfs_routes["route_short_name"]
                .fillna("Unknown")
                .value_counts()
                .reset_index(name="routes")
                .head(12)
            )
            route_mix.columns = ["service", "routes"]
            fig = px.pie(route_mix, values="routes", names="service", title="Service Mix by Route Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No route catalog available.")

    if not alerts.empty:
        st.subheader("Live Service Alerts")
        st.dataframe(alerts[["id", "route_count", "description"]], use_container_width=True, hide_index=True)

    if not atendo.empty:
        st.subheader("Atendo Accessibility Coverage")
        st.dataframe(atendo.head(50), use_container_width=True, hide_index=True)


def render_insights_plus(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Operational Insights")
    trip_updates = data["trip_updates"]
    vehicles = data["vehicles"]
    alerts = data["alerts"]
    incidents = data["incidents"]

    left, right = st.columns(2)
    with left:
        if not vehicles.empty and "status" in vehicles.columns:
            status_df = (
                vehicles["status"]
                .fillna("UNKNOWN")
                .value_counts()
                .reset_index(name="vehicles")
            )
            status_df.columns = ["status", "vehicles"]
            fig = px.bar(status_df, x="status", y="vehicles", title="Active Vehicles by Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No vehicle status data available.")

    with right:
        if not alerts.empty and "route_count" in alerts.columns:
            fig = px.histogram(alerts, x="route_count", nbins=20, title="Alert Impact Distribution (routes affected)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alert impact data available.")

    if not trip_updates.empty and "delay_seconds" in trip_updates.columns:
        feed_col = "feed" if "feed" in trip_updates.columns else None
        plot_df = trip_updates.assign(delay_min=trip_updates["delay_seconds"].fillna(0) / 60)
        if feed_col:
            fig = px.box(plot_df, x=feed_col, y="delay_min", title="Delay Spread by Service Feed")
        else:
            fig = px.box(plot_df, y="delay_min", title="Delay Spread")
        fig.update_layout(yaxis_title="Delay (min)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No delay spread data available.")

    if not incidents.empty:
        st.subheader("Latest Incident Notes")
        preview_cols = [c for c in ["chipText", "aspect", "paragraph", "link"] if c in incidents.columns]
        if preview_cols:
            st.dataframe(incidents[preview_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.dataframe(incidents.head(20), use_container_width=True, hide_index=True)


def render_analytics_plus(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Analytics")
    stations = data["stations"]
    trip_updates = data["trip_updates"]
    gtfs_routes = data["gtfs_routes"]
    gtfs_trips = data["gtfs_trips"]

    left, right = st.columns(2)
    with left:
        if not stations.empty and {"source", "province"}.issubset(stations.columns):
            source_province = (
                stations.groupby(["source", "province"], dropna=False)
                .size()
                .reset_index(name="stations")
                .sort_values("stations", ascending=False)
                .head(50)
            )
            fig = px.treemap(
                source_province,
                path=["source", "province"],
                values="stations",
                title="Station Coverage by Source and Province",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No station coverage data available.")

    with right:
        if not trip_updates.empty and "delay_seconds" in trip_updates.columns:
            delay = trip_updates["delay_seconds"].fillna(0) / 60
            buckets = pd.cut(
                delay,
                bins=[-1, 0, 5, 15, 30, 10_000],
                labels=["On time", "0-5 min", "5-15 min", "15-30 min", "30+ min"],
            )
            severity = buckets.value_counts().reset_index()
            severity.columns = ["severity", "trips"]
            fig = px.pie(severity, values="trips", names="severity", title="Delay Severity Mix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No delay severity data available.")

    st.subheader("Service Capacity Snapshot")
    c1, c2 = st.columns(2)
    with c1:
        if not gtfs_routes.empty and "route_short_name" in gtfs_routes.columns:
            top_services = (
                gtfs_routes["route_short_name"]
                .fillna("Unknown")
                .value_counts()
                .head(15)
                .reset_index(name="routes")
            )
            top_services.columns = ["service", "routes"]
            fig = px.bar(top_services, x="service", y="routes", title="Top Service Labels by Route Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No route service data available.")
    with c2:
        if not gtfs_trips.empty and "trip_id" in gtfs_trips.columns:
            st.metric("Total Scheduled Trips", f"{gtfs_trips['trip_id'].nunique():,}")
        else:
            st.info("No scheduled trip catalog available.")


def render_operations_dashboard(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Operations Command Center")
    trips = data["trip_updates"].copy()
    vehicles = data["vehicles"].copy()
    routes = data["gtfs_routes"].copy()
    scheduled = data["gtfs_trips"].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Live Trip Updates", f"{len(trips):,}")
    c2.metric("Live Vehicle Positions", f"{len(vehicles):,}")
    c3.metric("Scheduled Trips", f"{len(scheduled):,}")

    if not trips.empty:
        t = trips.copy()
        t["delay_seconds"] = pd.to_numeric(t["delay_seconds"], errors="coerce").fillna(0)
        t["delay_min"] = (t["delay_seconds"] / 60.0).round(2)
        t["timestamp"] = pd.to_datetime(t.get("timestamp"), errors="coerce")
        t["hour"] = t["timestamp"].dt.hour.fillna(-1).astype(int)
        by_hour = t.groupby("hour", as_index=False)["delay_min"].mean()
        fig_hour = px.line(by_hour, x="hour", y="delay_min", markers=True, title="Average Delay by Hour")
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("No trip update data available.")

    col_left, col_right = st.columns(2)
    with col_left:
        if not vehicles.empty and "status" in vehicles.columns:
            status_df = (
                vehicles.assign(status=vehicles["status"].astype(str).replace("", "UNKNOWN"))
                .groupby("status", as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            fig_status = px.bar(status_df, x="status", y="count", title="Vehicle Status Distribution")
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("Vehicle status data not available.")

    with col_right:
        if not routes.empty and "route_short_name" in routes.columns:
            top_routes = (
                routes.assign(route_short_name=routes["route_short_name"].astype(str))
                .groupby("route_short_name", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
                .head(15)
            )
            fig_routes = px.bar(
                top_routes.sort_values("count"),
                x="count",
                y="route_short_name",
                orientation="h",
                title="Top Route Families",
            )
            st.plotly_chart(fig_routes, use_container_width=True)
        else:
            st.info("Route data not available.")


def render_table_inventory(table_inventory: pd.DataFrame) -> None:
    st.subheader("Unity Catalog Table Inventory")
    if table_inventory.empty:
        st.warning("Could not retrieve table inventory from Unity Catalog.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tables", f"{len(table_inventory):,}")
    c2.metric("Bronze Tables", f"{int((table_inventory['layer'] == 'Bronze').sum()):,}")
    c3.metric("Domains", f"{table_inventory['domain'].nunique():,}")

    by_domain = (
        table_inventory.groupby(["domain", "layer"], as_index=False)
        .size()
        .rename(columns={"size": "tables"})
    )
    fig_domain = px.bar(by_domain, x="domain", y="tables", color="layer", barmode="group", title="Tables by Domain")
    st.plotly_chart(fig_domain, use_container_width=True)
    st.dataframe(table_inventory, use_container_width=True, hide_index=True)


def _build_ml_dataset(
    trip_updates: pd.DataFrame,
    gtfs_trips: pd.DataFrame,
    gtfs_routes: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> pd.DataFrame:
    if trip_updates.empty or "delay_seconds" not in trip_updates.columns:
        return pd.DataFrame()

    df = trip_updates.copy()
    df["delay_seconds"] = pd.to_numeric(df["delay_seconds"], errors="coerce")
    df = df.dropna(subset=["delay_seconds"])
    if df.empty:
        return pd.DataFrame()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)

    df["hour"] = ts.dt.hour.fillna(0).astype(int)
    df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.month.fillna(1).astype(int)
    df["time_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)
    df["feed"] = df.get("feed", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    trip_id = df.get("trip_id", pd.Series("", index=df.index)).fillna("").astype(str)
    df["trip_id"] = trip_id

    route_lookup = pd.DataFrame(columns=["trip_id", "route_id", "direction_id"])
    if not gtfs_trips.empty and "trip_id" in gtfs_trips.columns:
        route_lookup = gtfs_trips[["trip_id"]].copy()
        route_lookup["trip_id"] = route_lookup["trip_id"].astype(str)
        if "route_id" in gtfs_trips.columns:
            route_lookup["route_id"] = gtfs_trips["route_id"].astype(str)
        else:
            route_lookup["route_id"] = ""
        if "direction_id" in gtfs_trips.columns:
            route_lookup["direction_id"] = pd.to_numeric(gtfs_trips["direction_id"], errors="coerce").fillna(0).astype(int)
        else:
            route_lookup["direction_id"] = 0
    df = df.merge(route_lookup, how="left", on="trip_id")

    if "route_id" not in df.columns:
        df["route_id"] = ""
    df["route_id"] = df["route_id"].fillna("").astype(str)
    if "direction_id" not in df.columns:
        df["direction_id"] = 0
    df["direction_id"] = pd.to_numeric(df["direction_id"], errors="coerce").fillna(0).astype(int)

    route_type_lookup = pd.DataFrame(columns=["route_id", "route_type", "route_short_name"])
    if not gtfs_routes.empty and "route_id" in gtfs_routes.columns:
        route_type_lookup = gtfs_routes[["route_id"]].copy()
        route_type_lookup["route_id"] = route_type_lookup["route_id"].astype(str)
        if "route_type" in gtfs_routes.columns:
            route_type_lookup["route_type"] = gtfs_routes["route_type"].astype(str)
        else:
            route_type_lookup["route_type"] = "Unknown"
        if "route_short_name" in gtfs_routes.columns:
            route_type_lookup["route_short_name"] = gtfs_routes["route_short_name"].astype(str)
        else:
            route_type_lookup["route_short_name"] = "Unknown"
    df = df.merge(route_type_lookup, how="left", on="route_id")
    df["route_type"] = df.get("route_type", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    df["route_short_name"] = df.get("route_short_name", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)

    vehicle_features = pd.DataFrame(columns=["trip_id", "has_vehicle_position", "vehicle_status_mode", "vehicle_records"])
    if not vehicles.empty and "trip_id" in vehicles.columns:
        v = vehicles.copy()
        v["trip_id"] = v["trip_id"].fillna("").astype(str)
        status_col = "status" if "status" in v.columns else None
        if status_col:
            status_mode = (
                v.groupby("trip_id")[status_col]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
                .rename("vehicle_status_mode")
            )
        else:
            status_mode = pd.Series(dtype=str, name="vehicle_status_mode")
        counts = v.groupby("trip_id").size().rename("vehicle_records")
        vehicle_features = pd.concat([status_mode, counts], axis=1).reset_index()
        vehicle_features["vehicle_status_mode"] = vehicle_features["vehicle_status_mode"].fillna("UNKNOWN").astype(str)
        vehicle_features["vehicle_records"] = pd.to_numeric(vehicle_features["vehicle_records"], errors="coerce").fillna(0).astype(int)
        vehicle_features["has_vehicle_position"] = (vehicle_features["vehicle_records"] > 0).astype(int)
    df = df.merge(vehicle_features, how="left", on="trip_id")
    df["vehicle_status_mode"] = df.get("vehicle_status_mode", pd.Series("UNKNOWN", index=df.index)).fillna("UNKNOWN").astype(str)
    df["vehicle_records"] = pd.to_numeric(df.get("vehicle_records", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    df["has_vehicle_position"] = pd.to_numeric(df.get("has_vehicle_position", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)

    return df[
        [
            "feed",
            "day_of_week",
            "hour",
            "is_weekend",
            "month",
            "time_bucket",
            "route_type",
            "route_short_name",
            "direction_id",
            "has_vehicle_position",
            "vehicle_status_mode",
            "vehicle_records",
            "delay_seconds",
        ]
    ]


def render_ml_tab(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("ML Root Causes & Delay Prediction")
    ml_df = _build_ml_dataset(data["trip_updates"], data["gtfs_trips"], data["gtfs_routes"], data["vehicles"])
    if ml_df.empty or len(ml_df) < 30:
        st.info("Not enough trip update rows to train a model (need at least 30 rows with delay_seconds).")
        return

    features = [
        "feed",
        "day_of_week",
        "hour",
        "is_weekend",
        "month",
        "time_bucket",
        "route_type",
        "route_short_name",
        "direction_id",
        "has_vehicle_position",
        "vehicle_status_mode",
        "vehicle_records",
    ]
    X = ml_df[features]
    y = ml_df["delay_seconds"]

    if y.nunique() <= 1:
        st.info("Model training skipped because all delay values are identical.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    categorical_features = ["feed", "time_bucket", "route_type", "route_short_name", "vehicle_status_mode"]
    numeric_features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        "direction_id",
        "has_vehicle_position",
        "vehicle_records",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=400, max_depth=16, random_state=42, n_jobs=-1)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MAE", f"{mae/60:.2f} min")
    with c2:
        st.metric("RMSE", f"{rmse/60:.2f} min")
    with c3:
        st.metric("R²", f"{r2:.3f}")

    st.subheader("Root Cause Drivers (Feature Importance)")
    transformed_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    rf_importance = model.named_steps["regressor"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": transformed_feature_names, "importance": rf_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top_importance = importance_df.head(20).copy()
    fig_imp = px.bar(
        top_importance.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Importances (Random Forest)",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Group encoded features (e.g., cat__route_type_3) into root categories for easier interpretation.
    def _feature_group(name: str) -> str:
        if name.startswith("cat__"):
            raw = name.replace("cat__", "")
            for base in categorical_features:
                if raw.startswith(f"{base}_"):
                    return base
            return raw
        if name.startswith("num__"):
            return name.replace("num__", "")
        return name

    grouped = (
        importance_df.assign(group=importance_df["feature"].map(_feature_group))
        .groupby("group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    fig_group = px.bar(
        grouped.head(12).sort_values("importance", ascending=True),
        x="importance",
        y="group",
        orientation="h",
        title="Importance by Root Variable",
    )
    st.plotly_chart(fig_group, use_container_width=True)

    # Permutation importance (model-agnostic check) for root variables.
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
        perm_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": perm.importances_mean,
            }
        ).sort_values("importance", ascending=False)
        st.caption("Permutation importance (higher means stronger contribution to predictive accuracy).")
        st.dataframe(perm_df, use_container_width=True, hide_index=True)
    except Exception:
        st.caption("Permutation importance unavailable for this run.")

    eval_df = pd.DataFrame({"actual_delay_min": y_test / 60, "predicted_delay_min": preds / 60})
    fig = px.scatter(
        eval_df,
        x="actual_delay_min",
        y="predicted_delay_min",
        title="Model Evaluation: Actual vs Predicted Delay",
    )
    fig.update_layout(xaxis_title="Actual Delay (min)", yaxis_title="Predicted Delay (min)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predict a New Delay")
    route_types = sorted(ml_df["route_type"].dropna().astype(str).unique().tolist())
    route_type_val = st.selectbox("Route type", options=route_types, index=0 if route_types else None)
    feeds = sorted(ml_df["feed"].dropna().astype(str).unique().tolist())
    feed_val = st.selectbox("Feed", options=feeds, index=0 if feeds else None)
    vehicle_statuses = sorted(ml_df["vehicle_status_mode"].dropna().astype(str).unique().tolist())
    vehicle_status_val = st.selectbox(
        "Vehicle status (mode for similar trips)",
        options=vehicle_statuses,
        index=0 if vehicle_statuses else None,
    )
    route_short_names = sorted(ml_df["route_short_name"].dropna().astype(str).unique().tolist())
    route_short_name_val = st.selectbox(
        "Route short name",
        options=route_short_names,
        index=0 if route_short_names else None,
    )
    hour_val = st.slider("Hour of day", min_value=0, max_value=23, value=8)
    day_val = st.selectbox("Day of week", options=[0, 1, 2, 3, 4, 5, 6], index=0, format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    weekend_val = 1 if day_val >= 5 else 0
    month_val = st.slider("Month", min_value=1, max_value=12, value=1)
    direction_val = st.selectbox("Direction ID", options=[0, 1], index=0)
    has_vehicle_val = st.selectbox("Has live vehicle position", options=[0, 1], index=1)
    vehicle_records_val = st.slider("Vehicle records count", min_value=0, max_value=10, value=1)
    time_bucket_val = "night" if hour_val <= 5 else "morning" if hour_val <= 11 else "afternoon" if hour_val <= 17 else "evening"

    if st.button("Predict Delay"):
        sample = pd.DataFrame(
            [
                {
                    "feed": feed_val,
                    "route_type": route_type_val,
                    "route_short_name": route_short_name_val,
                    "hour": hour_val,
                    "day_of_week": day_val,
                    "is_weekend": weekend_val,
                    "month": month_val,
                    "time_bucket": time_bucket_val,
                    "direction_id": direction_val,
                    "has_vehicle_position": has_vehicle_val,
                    "vehicle_status_mode": vehicle_status_val,
                    "vehicle_records": vehicle_records_val,
                }
            ]
        )
        pred_seconds = float(model.predict(sample)[0])
        st.success(f"Predicted delay: {pred_seconds/60:.1f} minutes ({pred_seconds:.0f} seconds)")

def main() -> None:
    _apply_app_style()

    user_name = _current_user_name()
    left, right = st.columns([0.78, 0.22])
    with left:
        st.markdown(
            """
            <div class="dbx-header">
              <p class="dbx-title">Renfe Train Analytics</p>
              <p class="dbx-subtitle">Operational insights, maps, dashboards powered by Databricks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div style='text-align:right; padding-top: 10px;'><span class='dbx-user'>{user_name}</span></div>",
            unsafe_allow_html=True,
        )

    genie_space_id = _default_genie_space_id()
    db_host = _default_server_hostname()
    db_http_path = _default_http_path()
    db_token = os.getenv("DATABRICKS_TOKEN", "")
    query_map = {
        "stations": GEO_QUERY_HINT,
        "avld": AVLD_QUERY_HINT,
        "feve": FEVE_QUERY_HINT,
        "cercanias_madrid": CERCANIAS_QUERY_HINT,
        "atendo": ATENDO_QUERY_HINT,
        "train_info": TRAIN_INFO_QUERY_HINT,
    }

    refresh_sql = st.button("Refresh Data", type="primary")

    if "warehouse_data" not in st.session_state:
        st.session_state["warehouse_data"] = empty_data()
    if "warehouse_error" not in st.session_state:
        st.session_state["warehouse_error"] = ""
    if "warehouse_loaded" not in st.session_state:
        st.session_state["warehouse_loaded"] = False
    must_reload = refresh_sql or not st.session_state["warehouse_loaded"]
    if must_reload:
        st.session_state["warehouse_error"] = ""
        has_token = bool((db_token or "").strip())
        if not db_host or not db_http_path:
            st.session_state["warehouse_error"] = (
                "Missing Databricks SQL connection settings (server hostname or HTTP path)."
            )
            st.session_state["warehouse_data"] = empty_data()
            st.session_state["warehouse_loaded"] = False
        elif not has_token and not _has_app_oauth_credentials():
            st.session_state["warehouse_error"] = (
                "Missing Databricks SQL credentials. Provide DATABRICKS_TOKEN, or run inside Databricks Apps "
                "with DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET injected."
            )
            st.session_state["warehouse_data"] = empty_data()
            st.session_state["warehouse_loaded"] = False
        else:
            try:
                st.session_state["warehouse_data"] = load_data_from_warehouse(db_host, db_http_path, db_token, query_map)
                st.session_state["warehouse_loaded"] = True
            except Exception as exc:
                st.session_state["warehouse_error"] = str(exc)
                st.session_state["warehouse_data"] = empty_data()
                st.session_state["warehouse_loaded"] = False

    data = st.session_state["warehouse_data"]
    warehouse_error = st.session_state["warehouse_error"]

    table_inventory = pd.DataFrame()
    if not warehouse_error:
        try:
            table_inventory = load_table_inventory(db_host, db_http_path, db_token)
        except Exception:
            table_inventory = pd.DataFrame()

    station_master = build_station_master(data)
    st.sidebar.header("Business Filters")
    province_options = (
        sorted(station_master["province"].dropna().astype(str).unique().tolist())
        if not station_master.empty and "province" in station_master.columns
        else []
    )
    source_options = (
        sorted(station_master["source"].dropna().astype(str).unique().tolist())
        if not station_master.empty and "source" in station_master.columns
        else []
    )
    constructor_options = (
        sorted(data["train_info"]["constructor"].dropna().astype(str).unique().tolist())
        if not data["train_info"].empty and "constructor" in data["train_info"].columns
        else []
    )

    selected_provinces = st.sidebar.multiselect("Province", province_options)
    selected_sources = st.sidebar.multiselect("Station Source", source_options)
    selected_constructors = st.sidebar.multiselect("Train Constructor", constructor_options)
    delay_range = (0, 180)

    filtered_data = apply_business_filters(
        data=data,
        provinces=selected_provinces,
        feeds=[],
        route_types=[],
        vehicle_statuses=[],
        delay_min_range=delay_range,
    )
    if selected_sources:
        for key in ("stations", "avld"):
            if not filtered_data[key].empty and "source" in filtered_data[key].columns:
                filtered_data[key] = filtered_data[key][filtered_data[key]["source"].astype(str).isin(selected_sources)]
    if selected_constructors and not filtered_data["train_info"].empty and "constructor" in filtered_data["train_info"].columns:
        filtered_data["train_info"] = filtered_data["train_info"][
            filtered_data["train_info"]["constructor"].astype(str).isin(selected_constructors)
        ]

    tab_overview, tab_map, tab_accessibility, tab_fleet, tab_inventory, tab_genie = st.tabs(
        [
            "Overview",
            "Maps",
            "Accessibility",
            "Fleet",
            "Table Inventory",
            "Train Expert",
        ]
    )
    with tab_overview:
        if warehouse_error:
            st.error(warehouse_error)
        render_csv_overview(filtered_data)
    with tab_map:
        if warehouse_error:
            st.error(warehouse_error)
        render_csv_map(filtered_data)
    with tab_accessibility:
        if warehouse_error:
            st.error(warehouse_error)
        render_csv_accessibility(filtered_data)
    with tab_fleet:
        if warehouse_error:
            st.error(warehouse_error)
        render_csv_fleet(filtered_data)
    with tab_inventory:
        if warehouse_error:
            st.error(warehouse_error)
        render_table_inventory(table_inventory)
    with tab_genie:
        render_genie_tab(genie_space_id, "Train Expert", "genie1")


if __name__ == "__main__":
    main()

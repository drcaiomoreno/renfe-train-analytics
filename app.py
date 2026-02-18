from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="Renfe Train Analytics",
    page_icon="🚆",
    layout="wide",
)

TRIP_QUERY_HINT = """-- Return columns like: trip_id, delay_seconds, timestamp (optional), feed (optional)
-- Example:
-- SELECT trip_id, delay_seconds, event_time AS timestamp
-- FROM catalog.schema.train_delay_events
-- WHERE event_time >= current_timestamp() - INTERVAL 2 HOURS
-- LIMIT 2000
"""

GEO_QUERY_HINT = """-- Return columns like: lat, lon, station_name (optional), province (optional), source (optional)
-- Example:
-- SELECT latitude AS lat, longitude AS lon, station_name, province
-- FROM catalog.schema.station_activity
-- LIMIT 5000
"""


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

    with dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
    ) as connection:
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

    lat_col = _pick_col(df, ["lat", "latitude", "stop_lat"])
    lon_col = _pick_col(df, ["lon", "lng", "longitude", "stop_lon"])
    if not lat_col or not lon_col:
        return pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"])

    code_col = _pick_col(df, ["station_code", "stop_id", "id", "code"])
    name_col = _pick_col(df, ["station_name", "stop_name", "name", "point_name"])
    province_col = _pick_col(df, ["province", "region", "state"])
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
                .reset_index()
                .rename(columns={"route_short_name": "service", "count": "routes"})
                .head(12)
            )
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


def main() -> None:
    st.title("Renfe Train Analytics App")
    st.caption("Analytics insights, maps, and operational dashboards using files from the local data folder.")

    data_dir = st.sidebar.text_input("Data folder", value="data")
    if not Path(data_dir).exists():
        st.error(f"Data folder not found: {data_dir}")
        st.stop()

    data = load_data(data_dir)

    st.sidebar.divider()
    st.sidebar.subheader("Databricks SQL Warehouse")
    sql_enabled = st.sidebar.checkbox("Enable SQL Warehouse integration", value=False)
    db_host = st.sidebar.text_input(
        "Server hostname",
        value=os.getenv("DATABRICKS_SERVER_HOSTNAME", ""),
        disabled=not sql_enabled,
    )
    db_http_path = st.sidebar.text_input(
        "HTTP path",
        value=os.getenv("DATABRICKS_HTTP_PATH", ""),
        disabled=not sql_enabled,
    )
    db_token = st.sidebar.text_input(
        "Access token",
        value=os.getenv("DATABRICKS_TOKEN", ""),
        type="password",
        disabled=not sql_enabled,
    )
    trip_sql = st.sidebar.text_area(
        "Trip query (optional)",
        value=st.session_state.get("trip_sql", TRIP_QUERY_HINT),
        height=150,
        disabled=not sql_enabled,
    )
    geo_sql = st.sidebar.text_area(
        "Geo query (optional)",
        value=st.session_state.get("geo_sql", GEO_QUERY_HINT),
        height=140,
        disabled=not sql_enabled,
    )
    st.session_state["trip_sql"] = trip_sql
    st.session_state["geo_sql"] = geo_sql
    refresh_sql = st.sidebar.button("Refresh warehouse data", disabled=not sql_enabled)

    if "warehouse_trip_updates" not in st.session_state:
        st.session_state["warehouse_trip_updates"] = pd.DataFrame(
            columns=["feed", "entity_id", "trip_id", "delay_seconds", "timestamp"]
        )
    if "warehouse_geo_points" not in st.session_state:
        st.session_state["warehouse_geo_points"] = pd.DataFrame(
            columns=["station_code", "station_name", "lat", "lon", "province", "source"]
        )
    if "warehouse_error" not in st.session_state:
        st.session_state["warehouse_error"] = ""

    if refresh_sql:
        st.session_state["warehouse_error"] = ""
        trip_df = pd.DataFrame(columns=["feed", "entity_id", "trip_id", "delay_seconds", "timestamp"])
        geo_df = pd.DataFrame(columns=["station_code", "station_name", "lat", "lon", "province", "source"])

        if not db_host or not db_http_path or not db_token:
            st.session_state["warehouse_error"] = "Missing Databricks SQL credentials."
        else:
            try:
                clean_trip_query = _clean_query(trip_sql)
                clean_geo_query = _clean_query(geo_sql)
                if clean_trip_query:
                    trip_raw = run_warehouse_query(db_host, db_http_path, db_token, clean_trip_query)
                    trip_df = normalize_trip_updates(trip_raw)
                if clean_geo_query:
                    geo_raw = run_warehouse_query(db_host, db_http_path, db_token, clean_geo_query)
                    geo_df = normalize_geo_points(geo_raw)
            except Exception as exc:
                st.session_state["warehouse_error"] = str(exc)

        st.session_state["warehouse_trip_updates"] = trip_df
        st.session_state["warehouse_geo_points"] = geo_df

    warehouse_trip_updates = st.session_state["warehouse_trip_updates"]
    warehouse_geo_points = st.session_state["warehouse_geo_points"]
    warehouse_error = st.session_state["warehouse_error"]

    if not warehouse_trip_updates.empty:
        data["trip_updates"] = pd.concat([data["trip_updates"], warehouse_trip_updates], ignore_index=True)
    if not warehouse_geo_points.empty:
        data["stations"] = pd.concat([data["stations"], warehouse_geo_points], ignore_index=True)

    tab_insights, tab_map, tab_dash = st.tabs(["Analytics Insights", "Maps", "Dashboards"])
    with tab_insights:
        render_insights(data)
    with tab_map:
        render_map(data)
    with tab_dash:
        render_dashboards(data)
        st.subheader("Warehouse Integration Status")
        if not sql_enabled:
            st.info("Warehouse integration is disabled. Enable it in the sidebar.")
        elif warehouse_error:
            st.error(warehouse_error)
        else:
            st.success(
                f"Warehouse rows loaded: trips={len(warehouse_trip_updates):,}, geo points={len(warehouse_geo_points):,}"
            )
        if sql_enabled:
            st.caption("Use the sidebar to update credentials and refresh SQL-backed datasets.")


if __name__ == "__main__":
    main()

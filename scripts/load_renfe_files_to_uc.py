#!/usr/bin/env python3
"""
Load Renfe app source files from `data/` into Unity Catalog Delta tables.

Target schema:
  catalog_caiom7nmz_d9oink.renfe_app_data

Run in Databricks (recommended):
  python scripts/load_renfe_files_to_uc.py --data-dir /Workspace/Repos/<user>/<repo>/data
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


DEFAULT_CATALOG = "catalog_caiom7nmz_d9oink"
DEFAULT_SCHEMA = "renfe_app_data"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _sanitize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", ascii_name).strip("_").lower()
    if not safe:
        safe = "col"
    if safe[0].isdigit():
        safe = f"col_{safe}"
    return safe


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for raw in out.columns:
        base = _sanitize_column_name(str(raw))
        idx = seen.get(base, 0)
        col = base if idx == 0 else f"{base}_{idx}"
        seen[base] = idx + 1
        new_cols.append(col)
    out.columns = new_cols
    return out


def _read_csv_flexible(path: Path, sep: str = ";") -> pd.DataFrame:
    encodings = ("utf-8-sig", "latin1", "cp1252")
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python", on_bad_lines="skip")
            df = _clean_columns(df)
            if len(df.columns) == 1 and ";" in df.columns[0]:
                df = pd.read_csv(path, sep=";", encoding=encoding, engine="python", on_bad_lines="skip")
                df = _clean_columns(df)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


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


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_stations(data_dir: Path) -> pd.DataFrame:
    estaciones = _read_csv_flexible(data_dir / "estaciones.csv")
    av_ld = _read_csv_flexible(data_dir / "listado_completo_av_ld_md.csv")
    cercanias = _read_csv_flexible(data_dir / "listado-estaciones-cercanias-madrid.csv")
    feve = _read_csv_flexible(data_dir / "listado-de-estaciones-feve-2.csv")
    frames = [
        _stations_from_df(estaciones, "General"),
        _stations_from_df(av_ld, "AV/LD"),
        _stations_from_df(cercanias, "Cercanias Madrid"),
        _stations_from_df(feve, "FEVE"),
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["station_code", "lat", "lon", "source"])


def load_trip_updates(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feed_name, file_name in [("Cercanias", "trip_updates.json"), ("LD", "trip_updates_LD.json")]:
        payload = _read_json(data_dir / file_name)
        header = payload.get("header", {}) if isinstance(payload, dict) else {}
        feed_ts = pd.to_datetime(pd.to_numeric(header.get("timestamp", None), errors="coerce"), unit="s")
        for entity in payload.get("entity", []):
            update = entity.get("tripUpdate", {})
            delay = update.get("delay")
            if delay is None:
                stop_updates = update.get("stopTimeUpdate", [])
                if stop_updates:
                    arr = stop_updates[0].get("arrival", {})
                    dep = stop_updates[0].get("departure", {})
                    delay = arr.get("delay", dep.get("delay"))
            rows.append(
                {
                    "feed": feed_name,
                    "entity_id": entity.get("id"),
                    "trip_id": update.get("trip", {}).get("tripId"),
                    "delay_seconds": pd.to_numeric(delay, errors="coerce"),
                    "timestamp": feed_ts,
                }
            )
    return pd.DataFrame(rows)


def load_vehicle_positions(data_dir: Path) -> pd.DataFrame:
    payload = _read_json(data_dir / "vehicle_positions.json")
    rows: list[dict[str, Any]] = []
    for entity in payload.get("entity", []):
        vehicle = entity.get("vehicle", {})
        pos = vehicle.get("position", {})
        rows.append(
            {
                "entity_id": entity.get("id"),
                "trip_id": vehicle.get("trip", {}).get("tripId"),
                "vehicle_id": vehicle.get("vehicle", {}).get("id"),
                "vehicle_label": vehicle.get("vehicle", {}).get("label"),
                "status": vehicle.get("currentStatus"),
                "stop_id": vehicle.get("stopId"),
                "lat": pd.to_numeric(pos.get("latitude"), errors="coerce"),
                "lon": pd.to_numeric(pos.get("longitude"), errors="coerce"),
                "event_timestamp": pd.to_datetime(
                    pd.to_numeric(vehicle.get("timestamp"), errors="coerce"), unit="s"
                ),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"])


def load_alerts(data_dir: Path) -> pd.DataFrame:
    payload = _read_json(data_dir / "alerts.json")
    rows: list[dict[str, Any]] = []
    for entity in payload.get("entity", []):
        alert = entity.get("alert", {})
        desc = ""
        translations = alert.get("descriptionText", {}).get("translation", [])
        if translations:
            desc = translations[0].get("text", "")
        rows.append(
            {
                "id": entity.get("id"),
                "route_count": len(alert.get("informedEntity", [])),
                "description": desc,
            }
        )
    return pd.DataFrame(rows)


def load_incidents(data_dir: Path) -> pd.DataFrame:
    payload = _read_json(data_dir / "rfincidentreports_co.noticeresults.json")
    return pd.DataFrame(payload if isinstance(payload, list) else [])


def load_atendo(data_dir: Path) -> pd.DataFrame:
    return _read_csv_flexible(data_dir / "listado-de-estaciones-con-servicio-de-atendo.csv")


def load_routes(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "google_transit" / "routes.txt"
    if not path.exists():
        return pd.DataFrame(columns=["route_short_name"])
    return _clean_columns(pd.read_csv(path))


def load_scheduled_trips(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "google_transit" / "trips.txt"
    if not path.exists():
        return pd.DataFrame(columns=["trip_id"])
    return _clean_columns(pd.read_csv(path))


def write_table(spark: SparkSession, pdf: pd.DataFrame, full_table_name: str) -> None:
    if pdf.empty and len(pdf.columns) == 0:
        raise ValueError(f"No columns found for {full_table_name}; source parser returned an empty schema.")
    safe_pdf = _sanitize_columns(pdf)
    sdf = spark.createDataFrame(safe_pdf)
    (
        sdf.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_table_name)
    )
    count = spark.table(full_table_name).count()
    print(f"Loaded {full_table_name}: {count} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory containing source files")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--create-catalog",
        action="store_true",
        help="Create catalog if it does not exist (requires metastore default storage or managed location).",
    )
    parser.add_argument(
        "--catalog-managed-location",
        default="",
        help="Managed location to use when creating catalog, e.g. abfss://<container>@<account>.dfs.core.windows.net/<path>",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    catalog_exists = spark.sql(f"SHOW CATALOGS LIKE '{args.catalog}'").count() > 0
    if not catalog_exists:
        if not args.create_catalog:
            raise RuntimeError(
                f"Catalog {args.catalog} does not exist. Create it in UI with Default Storage, "
                "or rerun with --create-catalog and optionally --catalog-managed-location."
            )
        create_catalog_sql = f"CREATE CATALOG IF NOT EXISTS {args.catalog}"
        if args.catalog_managed_location:
            create_catalog_sql += f" MANAGED LOCATION '{args.catalog_managed_location}'"
        try:
            spark.sql(create_catalog_sql)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create catalog. If Default Storage is not configured at metastore level, "
                "provide --catalog-managed-location or create the catalog from UI."
            ) from exc
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema}")

    prefix = f"{args.catalog}.{args.schema}"
    write_table(spark, load_stations(data_dir), f"{prefix}.stations_dim")
    write_table(spark, load_trip_updates(data_dir), f"{prefix}.trip_updates_rt")
    write_table(spark, load_vehicle_positions(data_dir), f"{prefix}.vehicle_positions_rt")
    write_table(spark, load_routes(data_dir), f"{prefix}.routes_dim")
    write_table(spark, load_scheduled_trips(data_dir), f"{prefix}.trips_dim")
    write_table(spark, load_alerts(data_dir), f"{prefix}.alerts_rt")
    write_table(spark, load_incidents(data_dir), f"{prefix}.incidents")
    write_table(spark, load_atendo(data_dir), f"{prefix}.atendo_accessibility")

    # Convenience views matching app hint names.
    spark.sql(f"CREATE OR REPLACE VIEW {prefix}.stations_view AS SELECT * FROM {prefix}.stations_dim")
    spark.sql(f"CREATE OR REPLACE VIEW {prefix}.routes_view AS SELECT * FROM {prefix}.routes_dim")
    spark.sql(f"CREATE OR REPLACE VIEW {prefix}.trips_view AS SELECT * FROM {prefix}.trips_dim")

    print("All Renfe app tables loaded successfully.")


if __name__ == "__main__":
    main()

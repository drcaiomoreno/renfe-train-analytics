#!/usr/bin/env python3
"""
Auto Loader ingestion for all files in UC Volume:
  /Volumes/catalog_caiom7nmz_d9oink/renfe_app_data/renfe_app_data_files

This script ingests files into Delta Bronze tables using Databricks Auto Loader
best practices:
  - cloudFiles source with schema tracking
  - per-stream checkpointing
  - rescued data column for malformed records
  - availableNow trigger for backfill + incremental consistency
  - source metadata columns for lineage
"""

from __future__ import annotations

import argparse
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, functions as F


DEFAULT_CATALOG = "catalog_caiom7nmz_d9oink"
DEFAULT_SCHEMA = "renfe_app_data"
DEFAULT_SOURCE_PATH = "/Volumes/catalog_caiom7nmz_d9oink/renfe_app_data/renfe_app_data_files"
DEFAULT_STATE_BASE = "dbfs:/tmp/renfe_app_autoloader_state"


@dataclass(frozen=True)
class IngestionSpec:
    table_name: str
    source_glob: str
    source_format: str
    options: dict[str, str]
    description: str


def _sanitize_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "dataset"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def _sanitize_spark_columns(df: DataFrame) -> DataFrame:
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for c in df.columns:
        base = _sanitize_name(c)
        idx = seen.get(base, 0)
        col = base if idx == 0 else f"{base}_{idx}"
        seen[base] = idx + 1
        new_cols.append(col)
    return df.toDF(*new_cols)


def _csv_options_top_level() -> dict[str, str]:
    return {
        "header": "true",
        "sep": ";",
        "quote": '"',
        "escape": '"',
        "encoding": "ISO-8859-1",
        "cloudFiles.inferColumnTypes": "false",
        "cloudFiles.schemaEvolutionMode": "rescue",
        "rescuedDataColumn": "_rescued_data",
    }


def _csv_options_gtfs() -> dict[str, str]:
    return {
        "header": "true",
        "sep": ",",
        "quote": '"',
        "escape": '"',
        "cloudFiles.inferColumnTypes": "false",
        "cloudFiles.schemaEvolutionMode": "rescue",
        "rescuedDataColumn": "_rescued_data",
    }


def _json_options() -> dict[str, str]:
    return {
        "cloudFiles.inferColumnTypes": "true",
        "cloudFiles.schemaEvolutionMode": "addNewColumns",
        "rescuedDataColumn": "_rescued_data",
        "multiLine": "true",
    }


def _spec_for_file(rel_path: str) -> IngestionSpec | None:
    p = Path(rel_path)
    parent = p.parent.as_posix()
    stem = _sanitize_name(p.stem)
    ext = p.suffix.lower()

    if ext == ".json":
        return IngestionSpec(
            table_name=f"bronze_json_{stem}",
            source_glob=rel_path,
            source_format="json",
            options=_json_options(),
            description=f"JSON source file: {rel_path}",
        )

    if ext == ".csv":
        return IngestionSpec(
            table_name=f"bronze_csv_{stem}",
            source_glob=rel_path,
            source_format="csv",
            options=_csv_options_top_level(),
            description=f"Top-level CSV source file: {rel_path}",
        )

    if ext == ".txt" and parent in {"google_transit", "fomento_transit", "horarios-feve"}:
        return IngestionSpec(
            table_name=f"bronze_{_sanitize_name(parent)}_{stem}",
            source_glob=rel_path,
            source_format="csv",
            options=_csv_options_gtfs(),
            description=f"GTFS TXT source file: {rel_path}",
        )

    if ext in {".pb", ".xlsx"}:
        return IngestionSpec(
            table_name=f"bronze_binary_{stem}",
            source_glob=rel_path,
            source_format="binaryFile",
            options={},
            description=f"Binary source file: {rel_path}",
        )

    return None


def _build_specs(source_base: str) -> list[IngestionSpec]:
    files = _list_relative_files(source_base)
    specs: list[IngestionSpec] = []
    for rel in files:
        spec = _spec_for_file(rel)
        if spec:
            specs.append(spec)
    return specs


def _with_lineage_columns(df: DataFrame, pattern: str, fmt: str) -> DataFrame:
    return (
        df.withColumn("_ingest_ts", F.current_timestamp())
        .withColumn("_source_file_path", F.col("_metadata.file_path"))
        .withColumn("_source_pattern", F.lit(pattern))
        .withColumn("_source_format", F.lit(fmt))
    )


def _ingest_spec(
    spark: SparkSession,
    source_base: str,
    state_base: str,
    catalog: str,
    schema: str,
    spec: IngestionSpec,
) -> None:
    target_table = f"{catalog}.{schema}.{_sanitize_name(spec.table_name)}"
    schema_loc = f"{state_base}/schemas/{_sanitize_name(spec.table_name)}"
    checkpoint_loc = f"{state_base}/checkpoints/{_sanitize_name(spec.table_name)}"

    source_path = f"{source_base.rstrip('/')}/{spec.source_glob}"
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", spec.source_format)
        .option("cloudFiles.includeExistingFiles", "true")
        .option("cloudFiles.schemaLocation", schema_loc)
    )
    for k, v in spec.options.items():
        reader = reader.option(k, v)

    df = reader.load(source_path)
    df = _sanitize_spark_columns(df)
    df = _with_lineage_columns(df, spec.source_glob, spec.source_format)

    query = (
        df.writeStream.format("delta")
        .option("checkpointLocation", checkpoint_loc)
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(target_table)
    )
    query.awaitTermination()

    count = spark.table(target_table).count()
    print(f"[OK] {target_table}: {count} rows ({spec.description})")


def _list_relative_files(source_base: str) -> list[str]:
    base = Path(source_base)
    if not base.exists():
        return []
    return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())


def _validate_pattern_coverage(source_base: str, specs: list[IngestionSpec]) -> None:
    files = _list_relative_files(source_base)
    if not files:
        print(f"[WARN] No files found under {source_base}")
        return

    uncovered = []
    patterns = [s.source_glob for s in specs]
    for rel in files:
        if not any(fnmatch.fnmatch(rel, p) for p in patterns):
            uncovered.append(rel)
    if uncovered:
        print("[WARN] The following files are not covered by ingestion specs:")
        for f in uncovered:
            print(f"  - {f}")
    else:
        print("[OK] All files are covered by ingestion specs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH, help="UC Volume source root path.")
    parser.add_argument("--state-base", default=DEFAULT_STATE_BASE, help="Base path for Auto Loader schema/checkpoints.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--create-catalog",
        action="store_true",
        help="Create catalog if missing (optional; requires metastore storage root or managed location).",
    )
    parser.add_argument(
        "--catalog-managed-location",
        default=None,
        help="Managed location URI used only with --create-catalog.",
    )
    parser.add_argument("--skip-coverage-check", action="store_true", help="Skip file-pattern coverage validation.")
    args = parser.parse_args()

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    if args.create_catalog:
        if args.catalog_managed_location:
            spark.sql(
                f"CREATE CATALOG IF NOT EXISTS {args.catalog} "
                f"MANAGED LOCATION '{args.catalog_managed_location}'"
            )
        else:
            spark.sql(f"CREATE CATALOG IF NOT EXISTS {args.catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema}")

    specs = _build_specs(args.source_path)
    if not specs:
        raise RuntimeError(f"No supported files found under {args.source_path}")
    if not args.skip_coverage_check:
        _validate_pattern_coverage(args.source_path, specs)

    for spec in specs:
        _ingest_spec(
            spark=spark,
            source_base=args.source_path,
            state_base=args.state_base,
            catalog=args.catalog,
            schema=args.schema,
            spec=spec,
        )

    print("Auto Loader ingestion completed.")


if __name__ == "__main__":
    main()

# renfe-train-analytics

Databricks App for Renfe analytics sourced from Databricks SQL Warehouse.

YouTube Video: https://www.youtube.com/watch?v=UxIbwCSlQy0

## Features

- Analytics insights:
  - Station, route, trip, delay, and vehicle KPIs
  - Delay distribution and station concentration charts
- Maps:
  - Geospatial station network layers (General, AV/LD, Cercanias, FEVE)
  - Live vehicle position layer
- Dashboards:
  - Top delayed trips
  - Route service mix
  - Live service alerts
  - Atendo accessibility coverage preview
  - Warehouse load status

## Files added for the app

- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `app.yaml`: Databricks App startup command

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Databricks SQL Warehouse configuration (required)

The app reads dashboard/map data from SQL queries configured in the sidebar.

Provide credentials via sidebar or environment variables:

- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_TOKEN`

In Databricks Apps deployment, `DATABRICKS_TOKEN` can be left empty if app OAuth credentials
are injected and the SQL Warehouse resource is attached.

### Genie integration

The app supports loading datasets through Genie. Default room:
`https://adb-7405618360625211.11.azuredatabricks.net/genie/rooms/01f10d0ef5311278a6829ada8a43e5b7?o=7405618360625211`

Second Genie tab room:
`https://adb-7405618360625211.11.azuredatabricks.net/genie/rooms/01f10d2386d41be4a33a652f9e3cf521?o=7405618360625211`

In the sidebar, enable `Use Genie room for data loading`.

Provide queries for these datasets:

- **Stations query** returning columns like:
  - `station_code` (optional)
  - `station_name`
  - `lat`
  - `lon`
  - `province` (optional)
  - `source` (optional)
- **Trip updates query** returning columns like:
  - `trip_id`
  - `delay_seconds` (or `delay_min`/`delay_minutes`)
  - `timestamp` (optional)
  - `feed` (optional)
- **Vehicles query** returning columns like:
  - `trip_id` (optional)
  - `vehicle_id` (optional)
  - `vehicle_label` (optional)
  - `status` (optional)
  - `stop_id` (optional)
  - `lat`
  - `lon`
- **Routes query** with at least one route/service label column.
- **Scheduled trips query** with one row per scheduled trip.
- **Alerts query** returning `id`, `description`, and optional `route_count`.
- **Incidents query** (optional).
- **Atendo query** (optional).

## Deploy with Databricks Asset Bundles

This repo now includes `databricks.yml` with an app resource:

- Resource key: `renfe_train_analytics_app`
- Targets: `dev` (default) and `prod`
- App source path: project root (`.`), using `app.yaml`

### 1. Authenticate CLI

```bash
databricks auth login --host https://<your-workspace-host>
```

### 2. Validate bundle

```bash
databricks bundle validate -t dev
```

### 3. Deploy app resources

```bash
databricks bundle deploy -t dev
```

### 4. Start/update the app deployment

```bash
databricks bundle run renfe_train_analytics_app -t dev
```

### 5. Get app URL/status

```bash
databricks bundle summary -t dev
```

## Load Source Files Into Unity Catalog

Scripts are available under `scripts/` to ingest all datasets used by the app into:
`catalog_caiom7nmz_d9oink.renfe_app_data`

Run on Databricks (cluster or serverless notebook with PySpark), reading directly from UC Volume:

```bash
python scripts/load_renfe_files_to_uc.py \
  --source-path /Volumes/catalog_caiom7nmz_d9oink/renfe_app_data/renfe_app_data_files \
  --catalog catalog_caiom7nmz_d9oink \
  --schema renfe_app_data
```

If the catalog does not exist and your metastore has no default storage root configured, use:

```bash
python scripts/load_renfe_files_to_uc.py \
  --source-path /Volumes/catalog_caiom7nmz_d9oink/renfe_app_data/renfe_app_data_files \
  --catalog catalog_caiom7nmz_d9oink \
  --schema renfe_app_data \
  --create-catalog \
  --catalog-managed-location "abfss://<container>@<storage-account>.dfs.core.windows.net/<path>"
```

Tables created:

- `stations_dim`
- `trip_updates_rt`
- `vehicle_positions_rt`
- `routes_dim`
- `trips_dim`
- `alerts_rt`
- `incidents`
- `atendo_accessibility`

Use `scripts/app_queries.sql` as copy/paste query templates for the Streamlit app sidebar.

### Run Loader As Bundle Job

The bundle includes job resource key: `load_renfe_to_uc_job`

Deploy resources:

```bash
databricks bundle deploy -t dev
```

Run the loader job:

```bash
databricks bundle run load_renfe_to_uc_job -t dev
```

If needed, override cluster settings at deploy time:

```bash
databricks bundle deploy -t dev \
  --var loader_spark_version=15.4.x-scala2.12 \
  --var loader_node_type_id=Standard_DS3_v2
```

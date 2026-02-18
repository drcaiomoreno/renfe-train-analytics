# renfe-train-analytics

Databricks App for Renfe analytics built from files in `data/`.

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
  - Databricks SQL Warehouse integration status

## Files added for the app

- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `app.yaml`: Databricks App startup command

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads data from `data/` by default (configurable in the sidebar).

### Optional Databricks SQL Warehouse integration

You can enrich local files with live Delta/warehouse data from the sidebar.

Provide credentials via sidebar or environment variables:

- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_TOKEN`

Then enable integration and run:

- **Trip query** returning columns like:
  - `trip_id`
  - `delay_seconds` (or `delay_min`/`delay_minutes`)
  - `timestamp` (optional)
  - `feed` (optional)
- **Geo query** returning columns like:
  - `lat`
  - `lon`
  - `station_name` (optional)
  - `province` (optional)
  - `source` (optional)

Successful results are merged into the existing Insights, Maps, and Dashboards views.

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

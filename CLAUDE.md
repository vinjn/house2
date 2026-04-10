# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web app for analyzing Silicon Valley real estate market trends using Redfin public data. It visualizes median prices, inventory, and other metrics across 20 Silicon Valley cities, and includes a Ridge regression price prediction feature.

## Running the App

```bash
# Install dependencies
uv sync
# or
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Rebuild the DuckDB database from a raw TSV (if redfin_cache is present)
python build_db.py
```

No test suite or linter is configured for this project.

## Architecture

**Data flow**: S3 (Redfin public TSV) → local gzip cache → DuckDB → Streamlit UI

All application logic lives in `app.py` (~470 lines). `build_db.py` is a standalone utility used to manually recreate `redfin.duckdb` from the decompressed TSV.

### Key sections of `app.py`

| Lines | Component | Description |
|-------|-----------|-------------|
| 53–88 | Cache freshness | Compares S3 ETag/LastModified against local cache |
| 92–162 | `load_sv_data()` | Main data loader: S3 sync → gzip decompress → DuckDB build; decorated with `@st.cache_data` |
| 179–209 | Sidebar filters | City, date range, metric, property type selections |
| 227–247 | KPI cards | Latest metric values per selected city |
| 251–269 | Trend chart | Plotly line chart of metric over time |
| 271–305 | YoY chart | Year-over-year % change bar chart |
| 307–338 | Heatmap | Multi-metric normalized comparison grid |
| 340–460 | Price prediction | Ridge regression with polynomial + sine/cosine seasonal features; 3–24 month horizon |
| 462–469 | Raw data export | Expandable filtered data table |

### Data layer

- **Source**: `s3://redfin-public-data/redfin_market_tracker/city_market_tracker.tsv000.gz` (unsigned access via boto3)
- **Local cache**: `redfin_cache.gz` (compressed) and `redfin_cache` (decompressed TSV)
- **Database**: `redfin.duckdb` — queried read-only during app runtime; all column names are lowercased on ingest
- Download uses 8 concurrent threads with 8 MB chunks; decompression streams in 4 MB chunks

### Prediction model

Ridge regression (alpha=1.0) trained on the full history for a selected city/metric. Features: polynomial degree-2 time trend + 12-month sine/cosine seasonality. Confidence bands are ±1.96σ of training residuals.

## Tech Stack

| Library | Purpose |
|---------|---------|
| Streamlit ≥1.55 | Web UI |
| DuckDB ≥1.5 | Columnar query engine |
| Pandas ≥2.3 | Aggregation / DataFrame ops |
| Plotly ≥6.6 | Interactive charts |
| scikit-learn ≥1.8 | Ridge regression & PolynomialFeatures |
| boto3 ≥1.42 | S3 data download (unsigned) |

Python 3.12 is required (see `.python-version`).

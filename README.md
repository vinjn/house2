# Silicon Valley Real Estate Analyzer

A Streamlit web app for analyzing Silicon Valley real estate market trends using Redfin public data. Visualizes median prices, inventory, and other metrics across 20 Silicon Valley cities, with a Ridge regression price prediction feature.

## Features

- **Trend charts** — median price and other metrics over time per city
- **Year-over-year comparison** — % change bar charts
- **Heatmap** — multi-metric normalized comparison across cities
- **Price prediction** — 3–24 month forecast using Ridge regression with polynomial + seasonal features
- **Raw data export** — filterable data table download

## Setup

Requires Python 3.12.

```bash
# Install dependencies
uv sync
# or
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Data is downloaded automatically from Redfin's public S3 bucket on first run and cached locally.

## Rebuilding the Database

If you have a raw TSV cache (`redfin_cache`), you can rebuild the DuckDB database manually:

```bash
python build_db.py
```

## Architecture

**Data flow**: S3 → local gzip cache → DuckDB → Streamlit UI

| Library | Purpose |
|---------|---------|
| Streamlit | Web UI |
| DuckDB | Columnar query engine |
| Pandas | Aggregation / DataFrame ops |
| Plotly | Interactive charts |
| scikit-learn | Ridge regression & PolynomialFeatures |
| boto3 | S3 data download (unsigned) |

Data source: `s3://redfin-public-data/redfin_market_tracker/city_market_tracker.tsv000.gz`

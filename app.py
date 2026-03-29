import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gzip
import time
import threading
from pathlib import Path
import boto3
import boto3.s3.transfer as s3transfer
from botocore import UNSIGNED
from botocore.config import Config
import duckdb

st.set_page_config(page_title="Silicon Valley Housing Trends", layout="wide")

REDFIN_CITY_URL = (
    "https://redfin-public-data.s3.us-west-2.amazonaws.com"
    "/redfin_market_tracker/city_market_tracker.tsv000.gz"
)

SILICON_VALLEY_CITIES = [
    "San Jose", "Palo Alto", "Mountain View", "Sunnyvale", "Santa Clara",
    "Cupertino", "Los Altos", "Menlo Park", "Redwood City", "San Mateo",
    "Foster City", "Milpitas", "Campbell", "Los Gatos", "Saratoga",
    "Fremont", "Newark", "Union City", "Hayward", "East Palo Alto",
]

METRICS = {
    "Median Sale Price": "median_sale_price",
    "Median List Price": "median_list_price",
    "Homes Sold": "homes_sold",
    "New Listings": "new_listings",
    "Inventory": "inventory",
    "Days on Market": "median_dom",
    "Sale-to-List Ratio": "avg_sale_to_list",
    "Price Drop %": "price_drops",
}


S3_BUCKET = "redfin-public-data"
S3_KEY = "redfin_market_tracker/city_market_tracker.tsv000.gz"
CACHE_FILE = Path(__file__).parent / "redfin_cache.gz"
UNZIPPED_FILE = Path(__file__).parent / "redfin_cache"
DB_FILE = Path(__file__).parent / "redfin.duckdb"

s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))


def _is_cache_fresh() -> bool:
    """Return True if local cache matches the S3 object's ETag/LastModified."""
    if not CACHE_FILE.exists():
        return False
    head = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
    remote_mtime = head["LastModified"].timestamp()
    local_mtime = CACHE_FILE.stat().st_mtime
    return local_mtime >= remote_mtime


def _build_db(bar=None):
    """Decompress gz cache and build DuckDB table with lowercase column names."""
    if bar:
        bar.progress(0.0, text="Decompressing cache…")
    with gzip.open(CACHE_FILE, "rb") as src, open(UNZIPPED_FILE, "wb") as dst:
        chunk_size = 4 * 1024 * 1024
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)

    if bar:
        bar.progress(0.6, text="Building DuckDB…")
    cache_path = str(UNZIPPED_FILE).replace("\\", "/")
    con = duckdb.connect(str(DB_FILE))
    con.execute(f"""
        CREATE OR REPLACE TABLE redfin AS
        SELECT * FROM read_csv('{cache_path}', delim='\t', header=true, quote='"')
    """)
    cols = con.execute("DESCRIBE redfin").fetchdf()["column_name"].tolist()
    lower_select = ", ".join(f'"{c}" AS {c.lower()}' for c in cols)
    con.execute(f"CREATE OR REPLACE TABLE redfin AS SELECT {lower_select} FROM redfin")
    con.close()
    if bar:
        bar.progress(1.0, text="Database ready.")


@st.cache_data(show_spinner=False)
def load_sv_data():
    bar = st.progress(0, text="Checking for updates…")

    need_download = not _is_cache_fresh()

    if need_download:
        head = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
        total = head["ContentLength"]
        seen = [0]
        error = [None]

        def progress_callback(bytes_transferred):
            seen[0] += bytes_transferred

        def do_download():
            try:
                transfer_config = s3transfer.TransferConfig(max_concurrency=8, multipart_chunksize=8 * 1024 * 1024)
                s3.download_file(
                    Bucket=S3_BUCKET,
                    Key=S3_KEY,
                    Filename=str(CACHE_FILE),
                    Callback=progress_callback,
                    Config=transfer_config,
                )
            except Exception as e:
                error[0] = e

        bar.progress(0, text="New data available. Downloading…")
        t = threading.Thread(target=do_download, daemon=True)
        t.start()
        while t.is_alive():
            pct = min(seen[0] / total, 1.0) if total else 0
            bar.progress(pct * 0.8, text=f"Downloading… {seen[0] / 1e6:.1f} / {total / 1e6:.1f} MB (8 threads)")
            time.sleep(0.2)
        t.join()
        if error[0]:
            raise error[0]

    # Rebuild DB if it doesn't exist or cache is newer
    db_stale = (
        not DB_FILE.exists()
        or (CACHE_FILE.exists() and CACHE_FILE.stat().st_mtime > DB_FILE.stat().st_mtime)
    )
    if db_stale:
        _build_db(bar)
    else:
        bar.progress(0, text="Loading from database…")

    # Query only Silicon Valley cities from DuckDB
    cities_sql = ", ".join(f"'{c}'" for c in SILICON_VALLEY_CITIES)
    con = duckdb.connect(str(DB_FILE), read_only=True)
    sv = con.execute(f"""
        SELECT period_begin, city, state_code, property_type,
               median_sale_price, median_list_price, homes_sold,
               new_listings, inventory, median_dom,
               avg_sale_to_list, price_drops
        FROM redfin
        WHERE city IN ({cities_sql})
          AND state_code = 'CA'
        ORDER BY period_begin
    """).df()
    con.close()

    sv["period_begin"] = pd.to_datetime(sv["period_begin"])
    for col in ["median_sale_price", "median_list_price", "homes_sold",
                "new_listings", "inventory", "median_dom",
                "avg_sale_to_list", "price_drops"]:
        if col in sv.columns:
            sv[col] = pd.to_numeric(sv[col], errors="coerce")
    bar.empty()
    return sv


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Silicon Valley Housing Market Trends")
st.caption("Data source: Redfin Data Center · Updated weekly")

try:
    sv = load_sv_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if sv.empty:
    st.error("No Silicon Valley data found after filtering.")
    st.stop()

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    available_cities = sorted(sv["city"].unique())
    selected_cities = st.multiselect(
        "Cities",
        options=available_cities,
        default=[c for c in [
            "Los Altos", "Los Gatos", "Santa Clara", "Sunnyvale",
            "Palo Alto", "Mountain View", "Cupertino",
            "Campbell", "Saratoga",
        ] if c in available_cities],
    )

    min_date = sv["period_begin"].min().date().replace(year=2023, month=1, day=1)
    max_date = sv["period_begin"].max().date()
    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(max(min_date, min_date.replace(year=2025, month=1, day=1)), max_date),
        format="YYYY-MM",
    )

    metric_label = st.selectbox("Primary metric", list(METRICS.keys()))
    property_types = sv["property_type"].dropna().unique().tolist()
    pt_options = ["All"] + sorted(property_types)
    default_pt = "Single Family Residential" if "Single Family Residential" in pt_options else "All"
    sel_type = st.selectbox("Property type", pt_options, index=pt_options.index(default_pt))

if not selected_cities:
    st.warning("Select at least one city in the sidebar.")
    st.stop()

# ── Apply filters ─────────────────────────────────────────────────────────────
df = sv[sv["city"].isin(selected_cities)].copy()
df = df[(df["period_begin"].dt.date >= date_range[0]) & (df["period_begin"].dt.date <= date_range[1])]
if sel_type != "All":
    df = df[df["property_type"] == sel_type]

metric_col = METRICS[metric_label]
if metric_col not in df.columns:
    st.error(f"Column '{metric_col}' not available in this dataset.")
    st.stop()

df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

# ── KPI cards (latest values, all SV combined) ────────────────────────────────
latest = (
    df.sort_values("period_begin")
    .groupby("city")
    .last()
    .reset_index()
)

st.subheader("Latest snapshot")
kpi_cols = st.columns(min(len(selected_cities), 5))
for i, row in latest.iterrows():
    col_idx = i % len(kpi_cols)
    val = row.get(metric_col)
    if pd.notna(val):
        if "price" in metric_col:
            display = f"${val:,.0f}"
        elif "ratio" in metric_col or "drop" in metric_col:
            display = f"{val:.1%}" if val < 10 else f"{val:.1f}%"
        else:
            display = f"{val:,.0f}"
        kpi_cols[col_idx].metric(row["city"], display)

st.divider()

# ── Main trend chart ──────────────────────────────────────────────────────────
st.subheader(f"{metric_label} over time")
agg = (
    df.groupby(["period_begin", "city"])[metric_col]
    .median()
    .reset_index()
)

fig = px.line(
    agg,
    x="period_begin",
    y=metric_col,
    color="city",
    labels={"period_begin": "Date", metric_col: metric_label, "city": "City"},
    template="plotly_white",
)
fig.update_traces(line_width=2)
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)

# ── YoY comparison bar chart ──────────────────────────────────────────────────
st.subheader("Year-over-year change (%)")
latest_period = df["period_begin"].max()
yoy_period = latest_period - pd.DateOffset(years=1)

latest_vals = (
    df[df["period_begin"] == latest_period]
    .groupby("city")[metric_col].median()
)
prior_vals = (
    df[(df["period_begin"] >= yoy_period - pd.Timedelta(days=15)) &
       (df["period_begin"] <= yoy_period + pd.Timedelta(days=15))]
    .groupby("city")[metric_col].median()
)
yoy = ((latest_vals - prior_vals) / prior_vals * 100).dropna().reset_index()
yoy.columns = ["city", "yoy_pct"]
yoy = yoy.sort_values("yoy_pct")

if not yoy.empty:
    fig2 = px.bar(
        yoy,
        x="yoy_pct",
        y="city",
        orientation="h",
        color="yoy_pct",
        color_continuous_scale=["#d73027", "#fee090", "#4575b4"],
        color_continuous_midpoint=0,
        labels={"yoy_pct": "YoY Change (%)", "city": "City"},
        template="plotly_white",
    )
    fig2.update_coloraxes(showscale=False)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Insufficient data for year-over-year comparison.")

# ── Multi-metric heatmap ──────────────────────────────────────────────────────
st.subheader("Multi-metric comparison (latest month)")
available_metrics = {k: v for k, v in METRICS.items() if v in df.columns}
heat_data = {}
for label, col in available_metrics.items():
    vals = df[df["period_begin"] == latest_period].groupby("city")[col].median()
    if not vals.empty:
        heat_data[label] = vals

if heat_data:
    heat_df = pd.DataFrame(heat_data).loc[selected_cities].dropna(how="all")
    # Normalize each column 0-1 for heatmap coloring
    norm_df = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())

    fig3 = go.Figure(
        data=go.Heatmap(
            z=norm_df.values,
            x=norm_df.columns.tolist(),
            y=norm_df.index.tolist(),
            colorscale="RdYlGn",
            text=heat_df.round(1).values,
            texttemplate="%{text}",
            hovertemplate="%{y} · %{x}<br>Value: %{text}<extra></extra>",
        )
    )
    fig3.update_layout(
        height=max(300, 60 * len(norm_df)),
        xaxis_tickangle=-30,
        template="plotly_white",
        margin=dict(l=120),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("Raw data"):
    show_cols = ["period_begin", "city", "property_type"] + list(available_metrics.values())
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df[show_cols].sort_values(["city", "period_begin"], ascending=[True, False]),
        use_container_width=True,
    )

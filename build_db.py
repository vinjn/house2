#!/usr/bin/env python3
"""Build redfin.duckdb from redfin_cache (uncompressed TSV)."""
import duckdb
from pathlib import Path

CACHE_FILE = Path(__file__).parent / "redfin_cache"
DB_FILE = Path(__file__).parent / "redfin.duckdb"


def build():
    if not CACHE_FILE.exists():
        print(f"Cache file not found: {CACHE_FILE}")
        return

    size_gb = CACHE_FILE.stat().st_size / 1e9
    print(f"Reading {CACHE_FILE.name} ({size_gb:.1f} GB)…")

    # DuckDB requires forward slashes on Windows
    cache_path = str(CACHE_FILE).replace("\\", "/")

    con = duckdb.connect(str(DB_FILE))
    con.execute(f"""
        CREATE OR REPLACE TABLE redfin AS
        SELECT * FROM read_csv(
            '{cache_path}',
            delim='\t',
            header=true,
            quote='"'
        )
    """)

    # Lowercase all column names so app.py can use lowercase references
    cols = con.execute("DESCRIBE redfin").fetchdf()["column_name"].tolist()
    lower_select = ", ".join(f'"{c}" AS {c.lower()}' for c in cols)
    con.execute(f"CREATE OR REPLACE TABLE redfin AS SELECT {lower_select} FROM redfin")

    count = con.execute("SELECT COUNT(*) FROM redfin").fetchone()[0]
    db_size_mb = DB_FILE.stat().st_size / 1e6
    print(f"Done. {count:,} rows → {DB_FILE.name} ({db_size_mb:.0f} MB)")
    con.close()


if __name__ == "__main__":
    build()

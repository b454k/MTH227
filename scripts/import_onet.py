"""Import all O*NET SQL dump files from ``onet_sql`` into DuckDB.

This is the base O*NET database build step. It executes each SQL file in
filename order and writes the resulting tables to ``data/duckdb/onet.duckdb``.
"""

from pathlib import Path
import duckdb

DB_PATH = Path("data/duckdb/onet.duckdb")
SQL_FOLDER = Path("onet_sql")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

if not SQL_FOLDER.exists():
    raise FileNotFoundError(f"SQL folder not found: {SQL_FOLDER}")

sql_files = sorted(SQL_FOLDER.glob("*.sql"))

if not sql_files:
    raise FileNotFoundError(f"No .sql files found in: {SQL_FOLDER}")

con = duckdb.connect(str(DB_PATH))

for file in sql_files:
    print(f"\nImporting: {file.name}")

    sql = file.read_text(encoding="utf-8", errors="ignore")

    try:
        con.execute(sql)
        print("OK")
    except Exception as e:
        print(f"FAILED: {file.name}")
        print(e)
        break

tables = con.execute("SHOW TABLES").fetchall()
print(f"\nTotal tables imported: {len(tables)}")
for table in tables[:20]:
    print("-", table[0])

con.close()

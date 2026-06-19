"""Create or open the project DuckDB file used for imported O*NET tables."""

import duckdb

con = duckdb.connect("data/duckdb/onet.duckdb")

print("Database created")

import pandas as pd
import duckdb

def load_csv(path: str):
    return pd.read_csv(path)

def load_parquet(path: str):
    return pd.read_parquet(path)

def load_sql(query: str, db_path: str):
    con = duckdb.connect(db_path)
    return con.execute(query).fetchdf()
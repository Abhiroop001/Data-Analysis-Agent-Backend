# app/utils/schema_utils.py
import pandas as pd

def dataframe_schema_summary(df: pd.DataFrame) -> dict:
    """Return a compact schema + simple stats summary."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    stats = {}
    for col in df.columns:
        try:
            stats[col] = {
                "n_missing": int(df[col].isnull().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col].update({
                    "min": float(df[col].min(skipna=True)),
                    "max": float(df[col].max(skipna=True)),
                    "mean": float(df[col].mean(skipna=True)),
                    "std": float(df[col].std(skipna=True)),
                })
        except Exception as e:
            stats[col] = {"error": str(e)}
    return {"schema": schema, "stats": stats, "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}

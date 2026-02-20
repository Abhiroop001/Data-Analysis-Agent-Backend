# app/agents/visualization_agent.py
import pandas as pd
import plotly.express as px
import plotly.io as pio
from typing import List, Dict, Any

def numeric_histograms_json(df: pd.DataFrame, max_columns: int = 4) -> List[Dict[str, Any]]:
    numeric = df.select_dtypes("number").columns.tolist()
    chosen = numeric[:max_columns]
    out = []
    for col in chosen:
        fig = px.histogram(df, x=col, nbins=40, title=f"Distribution: {col}")
        out.append({"column": col, "plotly_json": pio.to_json(fig)})
    return out

def correlation_heatmap_json(df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any] | None:
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return None
    corr = num.corr().abs().unstack().sort_values(ascending=False)
    # prepare a small heatmap for top_k columns by variance
    top_cols = num.var().sort_values(ascending=False).head(top_k).index.tolist()
    fig = px.imshow(num[top_cols].corr(), text_auto=True, title="Correlation heatmap")
    import plotly.io as pio
    return {"plotly_json": pio.to_json(fig), "columns": top_cols}

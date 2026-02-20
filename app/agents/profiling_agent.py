# app/agents/profiling_agent.py
from pathlib import Path
import pandas as pd
import tempfile

try:
    from ydata_profiling import ProfileReport  # ydata-profiling
    _HAS_PROFILE = True
except Exception:
    _HAS_PROFILE = False

def generate_profile_html(df: pd.DataFrame, out_dir: str = "reports") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"profile_{hash(df.shape) % (10**6)}.html"

    if _HAS_PROFILE:
        profile = ProfileReport(df, explorative=True)
        profile.to_file(out_path)
        return str(out_path)
    else:
        # fallback: small HTML summary
        html = ["<html><body><h1>Quick Profile</h1>"]
        html.append(f"<p>Rows: {df.shape[0]} Columns: {df.shape[1]}</p>")
        html.append("<h2>Top columns</h2><ul>")
        for col in list(df.columns)[:20]:
            dtype = str(df[col].dtype)
            nmiss = int(df[col].isnull().sum())
            html.append(f"<li>{col} — {dtype} — missing: {nmiss}</li>")
        html.append("</ul></body></html>")
        out_path.write_text("\n".join(html))
        return str(out_path)

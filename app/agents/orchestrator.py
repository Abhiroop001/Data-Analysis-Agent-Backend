# app/agents/orchestrator.py
from typing import Dict, Any
import pandas as pd
from app.scaledown.client import compress_schema_and_stats_from_df
from app.agents.profiling_agent import generate_profile_html
from app.agents.visualization_agent import numeric_histograms_json, correlation_heatmap_json
from app.agents.insight_agent import basic_insights, detect_anomalies
from app.agents.automl_agent import recommend_model

def run_pipeline(df: pd.DataFrame, target: str | None = None) -> Dict[str, Any]:
    """
    Run the full EDA pipeline and return a serializable dict.
    """
    # 1. scaledown compression on schema/stats
    scaledown_result = compress_schema_and_stats_from_df(df)

    # 2. profile (HTML) - saved to reports/
    profile_path = generate_profile_html(df, out_dir="reports")

    # 3. visualizations (json)
    histos = numeric_histograms_json(df, max_columns=4)
    heat = correlation_heatmap_json(df, top_k=8)

    # 4. insights
    insights = basic_insights(df)

    # 5. anomalies
    anomalies = detect_anomalies(df)

    # 6. automl recommendation (optional - requires target)
    automl = None
    if target:
        automl = recommend_model(df, target=target)

    return {
        "scaledown": scaledown_result,
        "profile_html": profile_path,
        "visualizations": {"histograms": histos, "heatmap": heat},
        "insights": insights,
        "anomalies": anomalies,
        "automl": automl
    }

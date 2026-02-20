# app/agents/insight_agent.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Dict, Any, List

def basic_insights(df: pd.DataFrame) -> List[str]:
    insights = []
    insights.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    missing_total = int(df.isnull().sum().sum())
    insights.append(f"Total missing values: {missing_total}")
    # top 5 columns by missing %
    miss_pct = (df.isnull().mean() * 100).sort_values(ascending=False).head(5)
    for col, pct in miss_pct.items():
        insights.append(f"Missing: {col} — {pct:.2f}%")
    # skewed numeric cols
    nums = df.select_dtypes("number")
    for col in nums.columns[:5]:
        vals = nums[col].dropna()
        if len(vals) > 2:
            skew = float(vals.skew())
            if abs(skew) > 2:
                insights.append(f"Highly skewed numeric column: {col} (skew={skew:.2f})")
    return insights

def detect_anomalies(df: pd.DataFrame, max_samples: int = 5000) -> Dict[str, Any]:
    """
    Use IsolationForest on numeric columns to flag anomalies.
    Returns a dict: {col: [rows indices that are anomalous (sampled index)]}
    """
    nums = df.select_dtypes("number").copy()
    if nums.shape[1] == 0 or nums.shape[0] == 0:
        return {"note": "no numeric columns or empty dataframe"}
    # sample to keep runtime manageable
    if nums.shape[0] > max_samples:
        sample_idx = np.random.choice(nums.index, max_samples, replace=False)
        nums_sample = nums.loc[sample_idx]
    else:
        sample_idx = nums.index
        nums_sample = nums

    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    try:
        iso.fit(nums_sample.fillna(0))
        preds = iso.predict(nums_sample.fillna(0))  # -1 anomaly, 1 normal
        anomalies = list(nums_sample.index[preds == -1].tolist())
        return {"n_checked": len(nums_sample), "n_anomalies": len(anomalies), "anomaly_indices": anomalies}
    except Exception as e:
        return {"error": str(e)}

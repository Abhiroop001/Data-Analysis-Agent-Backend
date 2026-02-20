# app/agents/automl_agent.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
from typing import Tuple, Dict, Any

def infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        # numeric targets: regression if many unique values
        if y.nunique() > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"

def recommend_model(df: pd.DataFrame, target: str, quick: bool = True) -> Dict[str, Any]:
    if target not in df.columns:
        return {"error": "target not in dataframe"}
    df = df.copy()
    y = df[target]
    X = df.drop(columns=[target])
    # simple preprocessing
    X = X.select_dtypes(include=[np.number]).fillna(0)  # very simple baseline: numeric only
    if X.shape[1] == 0:
        return {"error": "no numeric features for baseline recommendation"}

    task = infer_task_type(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task == "classification":
        # encode y if needed
        if not pd.api.types.is_numeric_dtype(y_train):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train.astype(str))
            y_test = le.transform(y_test.astype(str))
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = float(model.score(X_test, y_test))
        return {"task": "classification", "model": "RandomForestClassifier", "score": score}
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = float(r2_score(y_test, preds))
        return {"task": "regression", "model": "RandomForestRegressor", "r2": score}

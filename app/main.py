from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import pandas as pd
from app.agents.orchestrator import run_pipeline
from app.config import settings
import uvicorn

app = FastAPI(title="Automated Data Science Agent")

class RunPayload(BaseModel):
    prompt: Optional[str] = None
    data: Optional[Dict[str, Any]] = None 
    target: Optional[str] = None

@app.post("/run-eda")
def run_eda(payload: RunPayload):
    if not payload.data:
        raise HTTPException(status_code=400, detail="No data provided")
    try:
        if isinstance(payload.data, dict):
            df = pd.DataFrame(payload.data)
        else:
            df = pd.DataFrame(payload.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not parse data to DataFrame: {str(e)}")

    result = run_pipeline(df, target=payload.target)
    return {
        "insights": result["insights"],
        "profile_html": result["profile_html"],
        "visualizations": result["visualizations"],
        "anomalies": result["anomalies"],
        "scaledown": result["scaledown"],
        "automl": result["automl"]
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

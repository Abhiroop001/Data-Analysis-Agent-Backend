from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import pandas as pd
import os
from app.agents.orchestrator import run_pipeline

app = FastAPI(title="Automated Data Science Agent")

# CORS (Required for deployed frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunPayload(BaseModel):
    prompt: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    target: Optional[str] = None

@app.post("/run-eda")
def run_eda(payload: RunPayload):

    if not payload.data:
        raise HTTPException(status_code=400, detail="No data provided")

    try:
        df = pd.DataFrame(payload.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse data: {str(e)}")

    result = run_pipeline(df, target=payload.target)

    return {
        "insights": result.get("insights"),
        "profile_html": result.get("profile_html"),
        "visualizations": result.get("visualizations"),
        "anomalies": result.get("anomalies"),
        "scaledown": result.get("scaledown"),
        "automl": result.get("automl"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )

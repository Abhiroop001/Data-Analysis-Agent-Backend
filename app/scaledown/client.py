# app/scaledown/client.py
import json
import gzip
import base64
from typing import Any, Dict
from app.config import settings
from app.utils.schema_utils import dataframe_schema_summary

# Try to import a real scaledown client if installed
try:
    from scaledown import ScaleDownClient  # type: ignore
    _HAS_SCALEDOWN = True
except Exception:
    _HAS_SCALEDOWN = False

class ScaleDownWrapper:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        if _HAS_SCALEDOWN:
            self.client = ScaleDownClient(api_key=api_key)
        else:
            self.client = None

    def compress(self, schema: dict, statistics: dict) -> Dict[str, Any]:
        """
        If a real ScaleDown client is available, use it.
        Otherwise use a deterministic fallback that gzips JSON and reports compression ratio.
        """
        if self.client:
            # If library available, call its compress / API (example placeholder)
            try:
                return self.client.compress(schema=schema, statistics=statistics)
            except Exception as e:
                return {"error": f"scaledown client error: {str(e)}"}
        # Fallback
        payload = {"schema": schema, "statistics": statistics}
        raw = json.dumps(payload).encode("utf-8")
        compressed = gzip.compress(raw, compresslevel=6)
        ratio = round(1 - (len(compressed) / max(len(raw), 1)), 3)
        return {
            "method": "gzip-fallback",
            "original_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "compression_ratio": float(ratio),
            "compressed_b64": base64.b64encode(compressed).decode("utf-8")
        }

# single shared instance
scaledown_client = ScaleDownWrapper(api_key=settings.scaledown_api_key)

def compress_schema_and_stats_from_df(df):
    summary = dataframe_schema_summary(df)
    return scaledown_client.compress(schema=summary["schema"], statistics=summary["stats"])

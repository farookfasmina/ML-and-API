from __future__ import annotations
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

# ---- logging ----
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("iris_api")

MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_meta.json")
_model = None
_meta = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _meta
    try:
        _model = joblib.load(MODEL_PATH)
        log.info("Model loaded")
        _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        log.info("Metadata loaded")
    except Exception as e:
        log.exception("Startup load failed: %s", e)
        raise
    yield

app = FastAPI(
    title="Iris ML Model API",
    description="Predict iris species from four measurements (cm)",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- schemas ----
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="cm")
    sepal_width:  float = Field(..., gt=0, description="cm")
    petal_length: float = Field(..., gt=0, description="cm")
    petal_width:  float = Field(..., gt=0, description="cm")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float | None = None

class BatchInput(BaseModel):
    items: List[PredictionInput]

class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]

# ---- helpers ----
def _to_feature_array(pi: PredictionInput) -> np.ndarray:
    order = _meta.get("features", ["sepal_length","sepal_width","petal_length","petal_width"])
    vals = [getattr(pi, f) for f in order]
    return np.array(vals, dtype=float)

def _predict_one(pi: PredictionInput) -> PredictionOutput:
    x = _to_feature_array(pi).reshape(1, -1)
    try:
        idx = int(_model.predict(x)[0])
        label = _meta["class_names"][idx]
        conf = None
        if hasattr(_model, "predict_proba"):
            conf = float(np.max(_model.predict_proba(x)[0]))
        return PredictionOutput(prediction=label, confidence=conf)
    except Exception as e:
        log.exception("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

# ---- endpoints ----
@app.get("/")
def health():
    return {"status": "healthy", "message": "Iris API running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(body: PredictionInput):
    return _predict_one(body)

@app.get("/model-info")
def model_info():
    return _meta

# bonus: batch + tiny test page
@app.post("/predict-batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    try:
        return BatchOutput(predictions=[_predict_one(i) for i in batch.items])
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Batch error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <!doctype html><meta charset="utf-8">
    <h2>Iris Predictor</h2>
    <label>sepal_length <input id="sl" type="number" step="0.01" value="5.1"></label><br>
    <label>sepal_width  <input id="sw" type="number" step="0.01" value="3.5"></label><br>
    <label>petal_length <input id="pl" type="number" step="0.01" value="1.4"></label><br>
    <label>petal_width  <input id="pw" type="number" step="0.01" value="0.2"></label><br>
    <button onclick="go()">Predict</button>
    <pre id="out"></pre>
    <script>
      async function go(){
        const payload = {
          sepal_length: parseFloat(sl.value),
          sepal_width:  parseFloat(sw.value),
          petal_length: parseFloat(pl.value),
          petal_width:  parseFloat(pw.value)
        };
        const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        out.textContent = JSON.stringify(await r.json(), null, 2);
      }
    </script>
    """

from typing import Dict, List, Any
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier


# In container, code lives under /app/code and models under /app/models
CODE_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = CODE_ROOT / "models" / "titanic_model.cbm"
META_PATH = CODE_ROOT / "models" / "metadata.json"


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    target_names: List[str]


def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


app = FastAPI(title="Titanic Survival API", version="1.0.0")
model = None
meta = None


@app.on_event("startup")
def startup_event():
    global model, meta
    model, meta = load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"feature_names": meta["feature_names"], "target_names": meta.get("target_names", ["Died", "Survived"]), "model": meta.get("model", "CatBoostClassifier"), "dataset": meta.get("dataset", "titanic")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    feature_names: List[str] = meta["feature_names"]
    missing = [f for f in feature_names if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    vals = []
    for name in feature_names:
        v = req.features[name]
        if name == "Sex":
            s = str(v).strip().lower()
            if s in ["f", "female"]:
                vals.append("female")
            elif s in ["m", "male"]:
                vals.append("male")
            else:
                raise HTTPException(status_code=400, detail="Sex must be M/F")
        else:
            vals.append(float(v))
    row = [vals]
    proba_surv = float(model.predict_proba(row)[0][1])
    pred = int(proba_surv >= 0.5)
    return PredictResponse(prediction=pred, probability=proba_surv, target_names=meta.get("target_names", ["Died", "Survived"]))

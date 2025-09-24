from typing import Dict, List, Any
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier


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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = [[
        req.Pclass,
        req.Sex,
        float(req.Age),
        float(req.Fare),
    ]]

    proba_surv = float(model.predict_proba(row)[0][1])
    pred = int(proba_surv >= 0.5)
    return PredictResponse(prediction=pred, probability=proba_surv)

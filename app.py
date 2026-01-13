import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="MediMind ML API", version="1.0.0")

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "medimind-ml-api"}


# -----------------------------
# Load artifacts
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ML model artifacts (for /predict)
rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "mlb.pkl"))
sym_feature_names = joblib.load(os.path.join(BASE_DIR, "sym_feature_names.pkl"))
vital_cols = joblib.load(os.path.join(BASE_DIR, "vital_cols.pkl"))

# Suggestion artifacts (for /suggest)
SUGG_PATH = os.path.join(BASE_DIR, "suggestion_data.pkl")
if os.path.exists(SUGG_PATH):
    _sugg = joblib.load(SUGG_PATH)
    COOCCUR: Dict[str, Dict[str, int]] = _sugg.get("cooccur", {})
    SYM_FREQ: Dict[str, int] = _sugg.get("symptomFreq", {})
    DIS_SYM: Dict[str, Dict[str, int]] = _sugg.get("diseaseSymCount", {})
    DIS_COUNT: Dict[str, int] = _sugg.get("diseaseCount", {})
else:
    COOCCUR, SYM_FREQ, DIS_SYM, DIS_COUNT = {}, {}, {}, {}

DEFAULT_VITALS = {
    "Age": 22,
    "Gender": 1,
    "bp_sys": 120,
    "bp_dia": 80,
    "hr": 80,
    "rr": 18,
    "blood_sugar": 100,
    "temp_c": 37.0,
    "spo2": 98,
}


def _clean_symptom(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if not s or s in {"none", "nan", "null"}:
        return ""
    # collapse multiple spaces
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _normalize_symptoms(symptoms: List[str]) -> List[str]:
    out = []
    for s in symptoms or []:
        ss = _clean_symptom(s)
        if ss:
            out.append(ss)
    # unique preserving order
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _normalize_vitals(vitals: Optional[Dict]) -> Dict[str, float]:
    vit = dict(DEFAULT_VITALS)
    if vitals:
        vit.update(vitals)

    # numeric coercion with fallback
    for k, default in DEFAULT_VITALS.items():
        try:
            vit[k] = float(vit[k])
        except Exception:
            vit[k] = float(default)

    # Gender as 0/1
    vit["Gender"] = 1.0 if vit["Gender"] >= 0.5 else 0.0
    return vit


# -----------------------------
# /predict
# -----------------------------
class PredictRequest(BaseModel):
    symptoms: List[str]
    vitals: Dict


@app.post("/predict")
def predict(data: PredictRequest):
    sym_list = _normalize_symptoms(data.symptoms)

    if len(sym_list) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 symptoms")

    sym_vec = mlb.transform([sym_list])[0]
    X_sym = pd.DataFrame([sym_vec], columns=sym_feature_names)

    vitals = _normalize_vitals(data.vitals)
    X_v = pd.DataFrame([vitals], columns=vital_cols)

    X = pd.concat([X_v, X_sym], axis=1)

    proba = rf_model.predict_proba(X)[0]
    classes = rf_model.classes_
    top3 = np.argsort(proba)[::-1][:3]

    results = []
    for i in top3:
        results.append({
            "disease": str(classes[i]),
            "probability": round(float(proba[i]) * 100, 2)
        })
    return results


# -----------------------------
# /suggest
# -----------------------------
class DiseaseProb(BaseModel):
    disease: str
    prob: float  # accepts 0..1 OR 0..100


class SuggestRequest(BaseModel):
    selectedSymptoms: List[str]
    topDiseases: Optional[List[DiseaseProb]] = None
    limit: int = 12


@app.post("/suggest")
def suggest(req: SuggestRequest):
    if not COOCCUR and not SYM_FREQ:
        # suggestion_data.pkl missing
        raise HTTPException(status_code=500, detail="Suggestion artifacts not found on server")

    selected = set(_normalize_symptoms(req.selectedSymptoms))
    limit = max(1, min(int(req.limit or 12), 50))

    score: Dict[str, float] = {}

    # (A) Co-occurrence score (weight 0.6)
    for s in selected:
        nxt = COOCCUR.get(s)
        if not nxt:
            continue
        for cand, cnt in nxt.items():
            if cand in selected:
                continue
            score[cand] = score.get(cand, 0.0) + 0.6 * float(cnt)

    # (B) Disease-driven score (weight 0.4)
    if req.topDiseases:
        for dp in req.topDiseases:
            d = (dp.disease or "").strip()
            if not d:
                continue
            p = float(dp.prob)
            # accept 0..100 from client (Android currently gets 0..100)
            if p > 1.0:
                p = p / 100.0
            if p <= 0:
                continue

            dcnt = int(DIS_COUNT.get(d, 0) or 0)
            smap = DIS_SYM.get(d)
            if dcnt <= 0 or not smap:
                continue

            for cand, cnt in smap.items():
                if cand in selected:
                    continue
                pSymGivenD = float(cnt) / float(dcnt)
                # multiply by 100 to roughly match cooccur scale
                score[cand] = score.get(cand, 0.0) + 0.4 * (p * pSymGivenD * 100.0)

    # fallback: popular symptoms
    if not score:
        popular_sorted = sorted(SYM_FREQ.keys(), key=lambda x: SYM_FREQ.get(x, 0), reverse=True)
        return popular_sorted[:limit]

    out = sorted(score.keys(), key=lambda x: score.get(x, 0.0), reverse=True)
    return out[:limit]

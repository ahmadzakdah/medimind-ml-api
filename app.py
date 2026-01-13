import os

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
@app.get("/")
def root():
    return {"status": "ok", "service": "medimind-ml-api"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# تحميل النموذج والأدوات
rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "mlb.pkl"))
sym_feature_names = joblib.load(os.path.join(BASE_DIR, "sym_feature_names.pkl"))
vital_cols = joblib.load(os.path.join(BASE_DIR, "vital_cols.pkl"))

class PredictRequest(BaseModel):
    symptoms: list[str]
    vitals: dict

@app.post("/predict")
def predict(data: PredictRequest):

    # تجهيز الأعراض
    sym_list = [s.strip().lower() for s in data.symptoms]
    sym_vec = mlb.transform([sym_list])[0]
    X_sym = pd.DataFrame([sym_vec], columns=sym_feature_names)

    # تجهيز الفايتلز
    X_v = pd.DataFrame([data.vitals], columns=vital_cols)

    # دمج
    X = pd.concat([X_v, X_sym], axis=1)

    # التنبؤ
    proba = rf_model.predict_proba(X)[0]
    classes = rf_model.classes_

    top3 = np.argsort(proba)[::-1][:3]

    results = []
    for i in top3:
        results.append({
            "disease": classes[i],
            "probability": round(float(proba[i]) * 100, 2)
        })

    return results

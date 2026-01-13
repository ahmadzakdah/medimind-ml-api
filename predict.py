import joblib
import pandas as pd
import numpy as np

rf_model = joblib.load("rf_model.pkl")
mlb = joblib.load("mlb.pkl")
sym_feature_names = joblib.load("sym_feature_names.pkl")
vital_cols = joblib.load("vital_cols.pkl")

def predict_top3(symptoms, vitals_dict):
    # normalize symptoms
    sym_list = []
    for s in symptoms:
        ss = str(s).strip().lower()
        if ss and ss not in ["none", "nan", "null"]:
            sym_list.append(ss)

    sym_vec = mlb.transform([sym_list])[0]
    X_sym = pd.DataFrame([sym_vec], columns=sym_feature_names)

    X_v = pd.DataFrame([vitals_dict], columns=vital_cols)

    X = pd.concat([X_v, X_sym], axis=1)

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

# Example run
if __name__ == "__main__":
    test_symptoms = ["fever", "cough", "fatigue"]
    test_vitals = {"Age": 22, "Gender": 1, "bp_sys": 118, "bp_dia": 75, "hr": 88,
                   "rr": 18, "blood_sugar": 105, "temp_c": 38.2, "spo2": 97}

    print(predict_top3(test_symptoms, test_vitals))

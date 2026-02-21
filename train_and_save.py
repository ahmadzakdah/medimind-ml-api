import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1) Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\LENOVO\Desktop\Dataset\dataset.csv")

symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
vital_cols = ["Age","Gender","bp_sys","bp_dia","hr","rr","blood_sugar","temp_c","spo2"]
target_col = "Disease"

# -----------------------------
# 2) Build symptom_list
# -----------------------------
symptoms_matrix = df[symptom_cols].where(pd.notna(df[symptom_cols]), None)

def row_to_symptom_list(row):
    clean = []
    for v in row:
        if v is None:
            continue
        s = str(v).strip().lower()
        if s in ["none", "nan", "null", ""]:
            continue
        clean.append(s)
    return clean

df["symptom_list"] = symptoms_matrix.apply(row_to_symptom_list, axis=1)

# -----------------------------
# 3) Clean vitals
# -----------------------------
df_clean = df.copy()
df_clean["Gender"] = (
    df_clean["Gender"].astype(str).str.strip().str.lower()
    .map({"male": 1, "female": 0})
)
df_clean["Gender"] = df_clean["Gender"].fillna(df_clean["Gender"].mode()[0]).astype(int)

for col in vital_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# -----------------------------
# 4) Deduplicate by symptom sets
# -----------------------------
df_clean["symptom_tuple"] = df_clean["symptom_list"].apply(lambda x: tuple(sorted(x)))
df_nodup = df_clean.drop_duplicates(subset=["symptom_tuple"]).copy()

# -----------------------------
# 5) Encode symptoms + combine with vitals
# -----------------------------
mlb = MultiLabelBinarizer()
X_sym = mlb.fit_transform(df_nodup["symptom_list"])
sym_feature_names = [f"sym_{s}" for s in mlb.classes_]
X_sym_df = pd.DataFrame(X_sym, columns=sym_feature_names, index=df_nodup.index)

X_vitals = df_nodup[vital_cols].copy()
X = pd.concat([X_vitals, X_sym_df], axis=1)
y = df_nodup[target_col].astype(str)

# -----------------------------
# 6) Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 7) Train RandomForest
# -----------------------------
rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_final.fit(X_train, y_train)

# -----------------------------
# 8) Calibrate probabilities
# -----------------------------
rf_calibrated = CalibratedClassifierCV(rf_final, method='isotonic', cv=3)
rf_calibrated.fit(X_train, y_train)

# -----------------------------
# 9) Evaluate on test set
# -----------------------------
y_pred = rf_calibrated.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# 10) Save artifacts
# -----------------------------
joblib.dump(rf_calibrated, "rf_model_calibrated.pkl")
joblib.dump(mlb, "mlb.pkl")
joblib.dump(sym_feature_names, "sym_feature_names.pkl")
joblib.dump(vital_cols, "vital_cols.pkl")

print("✅ Training + Calibration done & files saved!")
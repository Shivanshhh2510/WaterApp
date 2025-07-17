import streamlit as st
import csv
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

@st.cache_data
def load_data():
    # Load CSV via Python csv into NumPy arrays
    with open('water_potability.csv', newline='') as f:
        reader = csv.DictReader(f)
        feats = reader.fieldnames[:-1]  # all except "Potability"
        rows = list(reader)
    X = np.array([[float(r[f]) if r[f] != '' else np.nan for f in feats]
                  for r in rows])
    y = np.array([int(r['Potability']) for r in rows])
    return X, y, feats

@st.cache_resource
def train_model(X, y):
    # 1) Impute missing values (median)
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    # 2) Log-transform skewed columns by index
    #    Here cols 2,4,7 correspond to Solids,Sulfate,Trihalomethanes
    skew_idx = [2, 4, 7]
    X_imp[:, skew_idx] = np.log1p(X_imp[:, skew_idx])
    # 3) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    # 4) Balance
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)
    # 5) Train RF
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_bal, y_bal)
    return imp, scaler, skew_idx, rf

def main():
    st.set_page_config(page_title="💧 Water Potability Predictor")
    st.title("💧 Water Potability Predictor")

    # Load and train
    X, y, feats = load_data()
    imp, scaler, skew_idx, model = train_model(X, y)

    # Prepare slider defaults/ranges from raw X
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    meds = np.nanmedian(X, axis=0)

    # Build inputs
    user_vals = []
    for i, f in enumerate(feats):
        val = st.number_input(
            f.replace('_',' ').title(),
            float(mins[i]), float(maxs[i]), float(meds[i]),
            step=(maxs[i]-mins[i])/100
        )
        user_vals.append(val)

    if st.button("Predict Potability"):
        inp = np.array([user_vals])
        inp_imp = imp.transform(inp)
        inp_imp[:, skew_idx] = np.log1p(inp_imp[:, skew_idx])
        inp_scaled = scaler.transform(inp_imp)

        proba = model.predict_proba(inp_scaled)[0,1]
        verdict = "SAFE 💚" if proba >= 0.5 else "UNSAFE 🚩"
        st.markdown(f"## **{verdict}**  (Prob: {proba:.1%})")

        st.subheader("Feature Importances")
        imps = model.feature_importances_
        st.bar_chart({feats[i]: imps[i] for i in range(len(feats))})

if __name__=="__main__":
    main()

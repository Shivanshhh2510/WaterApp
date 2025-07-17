import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

@st.cache_data
def load_data():
    return pd.read_csv('water_potability.csv')

@st.cache_resource
def train_model(df):
    # 1) Impute
    imputer = SimpleImputer(strategy='median')
    X = df.drop('Potability', axis=1)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2) Log-transform skewed features
    skewed = ['Solids','Sulfate','Trihalomethanes']
    for c in skewed:
        X_imp[c] = np.log1p(X_imp[c])

    # 3) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # 4) SMOTE
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, df['Potability'])

    # 5) Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_bal, y_bal)

    return imputer, scaler, skewed, rf

def main():
    st.set_page_config(page_title="💧 Water Potability Predictor")
    st.title("💧 Water Potability Predictor")

    df = load_data()
    feats = df.columns.drop('Potability')
    mins, maxs, defaults = df.min(), df.max(), df.median()

    imputer, scaler, skewed, model = train_model(df)

    user_vals = {}
    for f in feats:
        user_vals[f] = st.number_input(
            f.replace('_',' ').title(),
            float(mins[f]), float(maxs[f]), float(defaults[f]),
            step=(maxs[f]-mins[f])/100
        )

    if st.button("Predict Potability"):
        inp = pd.DataFrame([user_vals])
        inp_imp = pd.DataFrame(imputer.transform(inp), columns=feats)
        for c in skewed:
            inp_imp[c] = np.log1p(inp_imp[c])
        inp_scaled = scaler.transform(inp_imp)

        proba = model.predict_proba(inp_scaled)[0,1]
        verdict = "SAFE 💚" if proba>=0.5 else "UNSAFE 🚩"
        st.markdown(f"## **{verdict}** (Prob: {proba:.1%})")

        st.subheader("Feature Importances")
        imps = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        st.bar_chart(imps)

if __name__=="__main__":
    main()

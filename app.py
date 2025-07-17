import streamlit as st
import pandas as pd
import numpy as np
import joblib

def main():
    # —— 1) Load your saved preprocessing + model objects ——
    imputer     = joblib.load('imputer.joblib')
    skewed_cols = joblib.load('skewed_cols.joblib')
    scaler      = joblib.load('scaler.joblib')
    rf_model    = joblib.load('rf_model.joblib')

    # —— 2) Load raw CSV to get slider ranges & order ——
    df_raw   = pd.read_csv('water_potability.csv')
    features = df_raw.columns.drop('Potability')
    defaults = df_raw.median()
    mins     = df_raw.min()
    maxs     = df_raw.max()

    # —— 3) Streamlit page setup ——
    st.set_page_config(page_title="💧 Water Potability Predictor", layout="centered")
    st.title("💧 Water Potability Predictor")
    st.write("Adjust the water-quality metrics below, then click **Predict Potability**.")

    # —— 4) Build sliders for each feature ——
    user_vals = {}
    for feat in features:
        user_vals[feat] = st.number_input(
            label=feat.replace('_',' ').title(),
            min_value=float(mins[feat]),
            max_value=float(maxs[feat]),
            value=float(defaults[feat]),
            format="%.2f"
        )

    # —— 5) Run inference when button clicked ——
    if st.button("Predict Potability"):
        # a) Create a one-row DataFrame
        input_df = pd.DataFrame([user_vals])

        # b) Impute missing values
        input_imp = pd.DataFrame(
            imputer.transform(input_df),
            columns=features
        )

        # c) Log-transform the skewed columns
        for c in skewed_cols:
            input_imp[c] = np.log1p(input_imp[c])

        # d) Scale all features
        input_scaled = scaler.transform(input_imp)

        # e) Predict probability of “safe” water
        proba_safe = rf_model.predict_proba(input_scaled)[0,1]
        verdict   = "SAFE 💚" if proba_safe >= 0.5 else "UNSAFE 🚩"

        # f) Display the verdict
        st.markdown(f"## **{verdict}**  (Probability: {proba_safe:.1%})")

        # g) Show feature importances
        st.subheader("Feature Importances")
        imps = pd.Series(
            rf_model.feature_importances_,
            index=features
        ).sort_values(ascending=False)
        st.bar_chart(imps)

if __name__ == "__main__":
    main()

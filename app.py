import streamlit as st
import csv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# —— 1) Basic logistic regression in NumPy ——
class LogisticRegressionND:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.weights) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# —— 2) Load data into DataFrame & NumPy ——
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('water_potability.csv')
    X = df.drop('Potability', axis=1).values
    y = df['Potability'].values
    feats = list(df.columns.drop('Potability'))
    return df, X, y, feats

# —— 3) Train-model (impute, scale, oversample, fit) ——
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    # a) Impute missing with median
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    # b) Log-transform skewed cols (Solids=2, Sulfate=4, Trihalomethanes=7)
    skew_idx = [2, 4, 7]
    X_imp[:, skew_idx] = np.log1p(X_imp[:, skew_idx])
    # c) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    # d) SMOTE
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)
    # e) Train logistic regression
    model = LogisticRegressionND(lr=0.1, n_iters=1000)
    model.fit(X_bal, y_bal)
    # f) Compute safe ranges (5th-95th percentiles)
    return imp, scaler, skew_idx, model

# —— 4) Main app ——
def main():
    st.set_page_config(page_title="💧 Water Potability Predictor", layout="wide")
    st.title("💧 Water Potability Predictor")

    # Load data & model
    df, X, y, feats = load_data()
    imp, scaler, skew_idx, model = train_model(X, y)

    # Compute slider ranges & safe ranges
    mins = df[feats].min()
    maxs = df[feats].max()
    meds = df[feats].median()
    safe5, safe95 = df[df['Potability']==1][feats].quantile(0.05), df[df['Potability']==1][feats].quantile(0.95)

    # Sidebar inputs
    st.sidebar.header("Input Water-Quality Metrics")
    user_vals = {}
    for f in feats:
        user_vals[f] = st.sidebar.slider(
            label=f.replace('_',' ').title(),
            min_value=float(mins[f]),
            max_value=float(maxs[f]),
            value=float(meds[f]),
            step=1.0,
            format="%.2f",
            help=f"Typical safe range: {safe5[f]:.2f} to {safe95[f]:.2f}"
        )
    st.sidebar.markdown("---")
    if st.sidebar.button("Predict Potability"):
        # Prepare input
        inp = np.array([[user_vals[f] for f in feats]])
        # Impute, transform, scale
        inp_imp = imp.transform(inp)
        inp_imp[:, skew_idx] = np.log1p(inp_imp[:, skew_idx])
        inp_scaled = scaler.transform(inp_imp)
        # Predict
        prob = model.predict_proba(inp_scaled)[0]
        verdict = "SAFE 💚" if prob >= 0.5 else "UNSAFE 🚩"
        # Metrics
        st.metric(label="Prediction", value=verdict, delta=f"{prob:.1%} safe chance")
        # Feature importances
        imps = np.abs(model.weights)
        fig = pd.Series(imps, index=feats).sort_values().plot(kind='barh', title="Feature Importances")
        st.plotly_chart(fig)

    # Data Exploration Tab
    tab1, tab2 = st.tabs(["Prediction","Data Exploration"])
    with tab2:
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature to explore", feats)
        hist_data = [df[df['Potability']==1][feature], df[df['Potability']==0][feature]]
        labels = ['Safe','Unsafe']
        fig2 = st.session_state.get('hist_plot', None)
        fig2 = pd.DataFrame({ 'Safe':hist_data[0], 'Unsafe':hist_data[1]}).plot(kind='hist', alpha=0.5, bins=30)
        st.plotly_chart(fig2)

if __name__ == "__main__":
    main()

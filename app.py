import streamlit as st
import csv
import numpy as np

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

# —— 2) Load CSV into NumPy arrays ——  
@st.cache_data(show_spinner=False)
def load_data():
    with open('water_potability.csv', newline='') as f:
        reader = csv.DictReader(f)
        feats = reader.fieldnames[:-1]
        rows = list(reader)
    X = np.array([[float(r[f]) if r[f] != '' else np.nan for f in feats]
                  for r in rows])
    y = np.array([int(r['Potability']) for r in rows])
    return X, y, feats

# —— 3) Train-model (impute, scale, oversample, fit) ——  
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    # a) Impute missing values with median
    medians = np.nanmedian(X, axis=0)
    inds_nan = np.where(np.isnan(X))
    X[inds_nan] = np.take(medians, inds_nan[1])

    # b) Log-transform skewed cols by index 
    #    (Solids idx=2, Sulfate=4, Trihalomethanes=7)
    skew_idx = [2, 4, 7]
    X[:, skew_idx] = np.log1p(X[:, skew_idx])

    # c) Scale features
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    X_scaled = (X - means) / stds

    # d) Simple oversampling of minority class
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx1) < len(idx0):
        extra = np.random.choice(idx1, size=len(idx0)-len(idx1), replace=True)
        X_bal = np.vstack([X_scaled, X_scaled[extra]])
        y_bal = np.concatenate([y, y[extra]])
    else:
        extra = np.random.choice(idx0, size=len(idx1)-len(idx0), replace=True)
        X_bal = np.vstack([X_scaled, X_scaled[extra]])
        y_bal = np.concatenate([y, y[extra]])

    # e) Fit logistic regression
    model = LogisticRegressionND(lr=0.1, n_iters=1000)
    model.fit(X_bal, y_bal)

    return medians, means, stds, skew_idx, model

# —— 4) App UI & inference ——  
def main():
    st.set_page_config(page_title="💧 Water Potability Predictor", layout="centered")
    st.title("💧 Water Potability Predictor")

    X, y, feats = load_data()
    medians, means, stds, skew_idx, model = train_model(X.copy(), y)

    # slider ranges
    mins    = np.nanmin(X, axis=0)
    maxs    = np.nanmax(X, axis=0)
    defaults= medians

    user_vals = []
    for i, f in enumerate(feats):
        val = st.number_input(
            f.replace('_',' ').title(),
            float(mins[i]), float(maxs[i]), float(defaults[i]),
            step=(maxs[i]-mins[i])/100
        )
        user_vals.append(val)

    if st.button("Predict Potability"):
        inp = np.array([user_vals], dtype=float)

        # impute missing values by column median
        mask = np.isnan(inp)
        cols = np.where(mask)[1]               # get column indices of NaNs
        inp[mask] = medians[cols]              # fill NaNs with corresponding medians

        # log1p on skewed
        inp[:, skew_idx] = np.log1p(inp[:, skew_idx])

        # scale
        inp_scaled = (inp - means) / stds

        proba = model.predict_proba(inp_scaled)[0]
        verdict = "SAFE 💚" if proba >= 0.5 else "UNSAFE 🚩"
        st.markdown(f"## **{verdict}**  (Prob: {proba:.1%})")

        # show “importances” (absolute weights)
        st.subheader("Feature Importances (|weights|)")
        imps = np.abs(model.weights)
        st.bar_chart({feats[i]: imps[i] for i in range(len(feats))})

if __name__=="__main__":
    main()

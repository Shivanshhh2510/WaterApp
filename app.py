import streamlit as st
import csv
import numpy as np

# ── 1) Simple logistic regression in NumPy ──────────────────────────────────────
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
            linear = X @ self.weights + self.bias
            y_hat  = self.sigmoid(linear)
            dw = (X.T @ (y_hat - y)) / n_samples
            db = np.sum(y_hat - y)   / n_samples
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)


# ── 2) Load CSV ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    with open("water_potability.csv", newline="") as f:
        reader = csv.DictReader(f)
        feats  = reader.fieldnames[:-1]
        rows   = list(reader)

    X = np.array([[float(r[f]) if r[f] else np.nan for f in feats] for r in rows])
    y = np.array([int(r["Potability"]) for r in rows])
    return X, y, feats


# ── 3) Train model (impute → log1p → scale → oversample → fit) ────────────────
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    med = np.nanmedian(X, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = med[nan_cols]

    skew_idx = [2, 4, 7]               # Solids, Sulfate, Trihalomethanes
    X[:, skew_idx] = np.log1p(X[:, skew_idx])

    mu, sigma = X.mean(0), X.std(0)
    Xs = (X - mu) / sigma

    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(idx1) < len(idx0):
        extra = np.random.choice(idx1, len(idx0) - len(idx1), replace=True)
    else:
        extra = np.random.choice(idx0, len(idx1) - len(idx0), replace=True)

    X_bal = np.vstack([Xs, Xs[extra]])
    y_bal = np.concatenate([y,  y[extra]])

    clf = LogisticRegressionND()
    clf.fit(X_bal, y_bal)
    return med, mu, sigma, skew_idx, clf


# ── 4) Safe-range helper (5th–95th pct. of safe samples) ───────────────────────
@st.cache_data
def safe_ranges(X, y):
    safe = X[y == 1]
    return np.nanpercentile(safe, 5, 0), np.nanpercentile(safe, 95, 0)


# ── 5) Streamlit App ───────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="💧 Water Potability Predictor", layout="wide")
    st.title("💧 Water Potability Predictor")

    X, y, feats = load_data()
    med, mu, sig, sk_idx, model = train_model(X.copy(), y)
    lo_safe, hi_safe = safe_ranges(X, y)

    mins, maxs = np.nanmin(X, 0), np.nanmax(X, 0)

    # ── Sidebar controls ────────────────────────────────────────────────────
    st.sidebar.header("🔧 Adjust metrics")
    preset1, preset2 = st.sidebar.columns(2)
    placeholder_vals = med.copy()      # default medians

    if preset1.button("💧 Tap-water"):
        placeholder_vals = med.copy()
    if preset2.button("🎲 Random safe"):
        placeholder_vals = X[y == 1][np.random.randint(sum(y == 1))]

    user_vals = []
    for i, f in enumerate(feats):
        help_txt = f"Safe 5-95%: {lo_safe[i]:.2f} – {hi_safe[i]:.2f}"
        user_vals.append(
            st.sidebar.slider(
                f.replace('_', ' ').title(),
                float(mins[i]), float(maxs[i]), float(placeholder_vals[i]),
                step=1.0, format="%.2f", help=help_txt,
            )
        )

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Predict Potability"):

        # ── Pre-process single sample ───────────────────────────────────────
        inp = np.array([user_vals])
        nan_mask = np.isnan(inp)
        if nan_mask.any():
            inp[nan_mask] = med[nan_mask[0]]

        inp[:, sk_idx] = np.log1p(inp[:, sk_idx])
        inp_scaled = (inp - mu) / sig

        # ── Predict ────────────────────────────────────────────────────────
        prob = model.predict_proba(inp_scaled)[0]
        verdict = "SAFE 💚" if prob >= 0.5 else "UNSAFE 🚩"

        st.metric("Prediction", verdict, f"{prob*100:.1f}% safe")
        st.progress(int(prob * 100))

        # ── Feature importances plot ───────────────────────────────────────
        st.subheader("Feature Importances (|weights|)")
        imps = np.abs(model.weights)
        st.bar_chart({feats[i]: imps[i] for i in range(len(feats))})

        # ── Expandable raw weights table ───────────────────────────────────
        with st.expander("🔍 Detailed weights"):
            st.write(
                "\n".join(
                    f"{feats[i]}: {model.weights[i]:+.4f}"
                    for i in np.argsort(-imps)
                )
            )


if __name__ == "__main__":
    main()

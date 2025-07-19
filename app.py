import streamlit as st
import csv
import numpy as np
import random

# —— 1) Logistic Regression (NumPy) ——
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
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.weights) + self.bias)

# —— 2) Load CSV Data ——
@st.cache_data(show_spinner=False)
def load_data():
    with open('water_potability.csv', newline='') as f:
        reader = csv.DictReader(f)
        feats = reader.fieldnames[:-1]
        rows = list(reader)
    X = np.array([[float(r[f]) if r[f] else np.nan for f in feats] for r in rows])
    y = np.array([int(r['Potability']) for r in rows])
    return X, y, feats

# —— 3) Model Training ——
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    med = np.nanmedian(X, axis=0)
    X[np.isnan(X)] = np.take(med, np.where(np.isnan(X))[1])
    skew_idx = [2, 4, 7]
    X[:, skew_idx] = np.log1p(X[:, skew_idx])
    mu, sigma = X.mean(0), X.std(0)
    Xs = (X - mu) / sigma

    id0, id1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(id1) < len(id0):
        extra = np.random.choice(id1, len(id0) - len(id1), replace=True)
    else:
        extra = np.random.choice(id0, len(id1) - len(id0), replace=True)
    Xb = np.vstack([Xs, Xs[extra]])
    yb = np.concatenate([y, y[extra]])

    model = LogisticRegressionND()
    model.fit(Xb, yb)
    return med, mu, sigma, skew_idx, model

# —— 4) Safe Range Reference ——
@st.cache_data
def safe_ranges(X, y):
    safe = X[y == 1]
    return np.nanpercentile(safe, 5, 0), np.nanpercentile(safe, 95, 0)

# —— 5) Random Facts ——
def get_random_fact():
    facts = [
        "💧 Only 1% of Earth's water is drinkable.",
        "💦 Boiling kills most bacteria in water.",
        "🚱 High sulfate levels can cause a bitter taste.",
        "🧪 Chloramines disinfect water, but excess isn't healthy.",
        "🌍 pH between 6.5 and 8.5 is ideal for drinking water."
    ]
    return random.choice(facts)

# —— 6) Suggest Improvement ——
def suggest_improvements(user_vals, low_safe, hi_safe, feats):
    suggestions = []
    for i, val in enumerate(user_vals):
        if val < low_safe[i]:
            suggestions.append(f"⬆️ Increase **{feats[i]}** to at least {low_safe[i]:.2f}")
        elif val > hi_safe[i]:
            suggestions.append(f"⬇️ Decrease **{feats[i]}** to below {hi_safe[i]:.2f}")
    return suggestions

# —— 7) App UI ——
def main():
    st.set_page_config(page_title='💧 Water Potability Predictor', layout='wide')
    st.title('💧 Water Potability Predictor')

    X, y, feats = load_data()
    med, mu, sig, sk_idx, model = train_model(X.copy(), y)
    low_safe, hi_safe = safe_ranges(X, y)
    mins, maxs = np.nanmin(X, 0), np.nanmax(X, 0)

    st.sidebar.header("🔧 Adjust Water Quality Features")
    presets_col1, presets_col2 = st.sidebar.columns(2)

    placeholder_vals = med.copy()

    if presets_col1.button("💧 Tap-water"):
        placeholder_vals = med.copy()

    if presets_col2.button("🎲 Random Safe"):
        safe_sample = X[y == 1][np.random.randint(sum(y == 1))]
        placeholder_vals = safe_sample.copy()

    user_vals = []
    for i, f in enumerate(feats):
        help_text = f"Safe range: {low_safe[i]:.2f} – {hi_safe[i]:.2f}"
        val = st.sidebar.slider(
            label=f.replace("_", " ").title(),
            min_value=float(mins[i]),
            max_value=float(maxs[i]),
            value=float(placeholder_vals[i]),
            step=1.0,
            format="%.2f",
            help=help_text
        )
        user_vals.append(val)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"🧠 **Did You Know?**\n\n{get_random_fact()}")

    predict = st.sidebar.button("🚀 Predict Potability")

    if predict:
        inp = np.array([user_vals])
        nan_mask = np.isnan(inp)
        if nan_mask.any():
            inp[nan_mask] = med[nan_mask[0]]
        inp[:, sk_idx] = np.log1p(inp[:, sk_idx])
        inp_scaled = (inp - mu) / sig
        prob = model.predict_proba(inp_scaled)[0]
        verdict = "SAFE 💚" if prob >= 0.5 else "UNSAFE 🚩"

        st.metric("Prediction", verdict, f"{prob*100:.1f}% safe")
        st.progress(int(prob * 100))

        st.subheader("📊 Feature Importances (|weights|)")
        imps = np.abs(model.weights)
        st.bar_chart({feats[i]: imps[i] for i in range(len(feats))})

        with st.expander("📥 Download Prediction"):
            st.download_button(
                label="Download Result as Text",
                file_name="prediction.txt",
                mime="text/plain",
                data=f"Prediction: {verdict}\nProbability: {prob*100:.2f}%"
            )

        if verdict == "UNSAFE 🚩":
            st.warning("💡 Suggestions to Improve Potability")
            suggestions = suggest_improvements(user_vals, low_safe, hi_safe, feats)
            for s in suggestions:
                st.markdown(f"- {s}")

    # —— 8) CSV Batch Prediction ——
    st.markdown("---")
    st.subheader("📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file (same structure as water_potability.csv)", type="csv")
    if uploaded_file is not None:
        raw = np.genfromtxt(uploaded_file, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        raw[:, sk_idx] = np.log1p(raw[:, sk_idx])
        raw_scaled = (raw - mu) / sig
        probs = model.predict_proba(raw_scaled)
        for i, p in enumerate(probs):
            label = "SAFE 💚" if p >= 0.5 else "UNSAFE 🚩"
            st.write(f"Sample {i+1}: **{label}** ({p*100:.1f}% safe)")

if __name__ == '__main__':
    main()

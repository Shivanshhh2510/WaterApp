import streamlit as st
import csv
import numpy as np
import pandas as pd

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

# —— 2) Load CSV ——
@st.cache_data(show_spinner=False)
def load_data():
    with open('water_potability.csv', newline='') as f:
        reader = csv.DictReader(f)
        feats = reader.fieldnames[:-1]
        rows = list(reader)
    X = np.array([[float(r[f]) if r[f] else np.nan for f in feats] for r in rows])
    y = np.array([int(r['Potability']) for r in rows])
    return X, y, feats

# —— 3) Train model ——
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    med = np.nanmedian(X, axis=0)
    X[np.isnan(X)] = np.take(med, np.where(np.isnan(X))[1])
    skew_idx = [2, 4, 7]
    X[:, skew_idx] = np.log1p(X[:, skew_idx])
    mu, sigma = X.mean(0), X.std(0)
    Xs = (X-mu)/sigma
    # naive balance
    id0, id1 = np.where(y==0)[0], np.where(y==1)[0]
    if len(id1)<len(id0):
        extra = np.random.choice(id1, len(id0)-len(id1), True)
    else:
        extra = np.random.choice(id0, len(id1)-len(id0), True)
    Xb = np.vstack([Xs, Xs[extra]]); yb=np.concatenate([y, y[extra]])
    model = LogisticRegressionND()
    model.fit(Xb, yb)
    return med, mu, sigma, skew_idx, model

# —— 4) Helper: safe ranges ——
@st.cache_data
def safe_ranges(X, y):
    safe = X[y==1]
    return np.nanpercentile(safe,5,0), np.nanpercentile(safe,95,0)

# —— 5) Main App ——
def main():
    st.set_page_config(page_title='💧 Water Potability Predictor', layout='wide')
    st.title('💧 Water Potability Predictor')

    X, y, feats     = load_data()
    med, mu, sig, sk_idx, model = train_model(X.copy(), y)
    low_safe, hi_safe = safe_ranges(X, y)

    mins, maxs = np.nanmin(X,0), np.nanmax(X,0)

    # ——— Sidebar inputs ———
    st.sidebar.header('🔧 Adjust metrics')
    presets_col1, presets_col2 = st.sidebar.columns(2)
    placeholder_vals = med.copy()
    if presets_col1.button('💧 Tap‑water'):
        placeholder_vals = med.copy()
    if presets_col2.button('🎲 Random safe'):
        safe_row = X[y==1][np.random.randint(sum(y==1))]
        placeholder_vals = safe_row.copy()

    tooltips = {
        "ph": "Measures acidity (ideal is ~7).",
        "Hardness": "Minerals (calcium, magnesium). Too high = scaling.",
        "Solids": "Total dissolved solids in ppm.",
        "Chloramines": "Used for disinfection; too much may be harmful.",
        "Sulfate": "Naturally occurring; excess may affect taste.",
        "Conductivity": "Water’s ability to conduct electricity.",
        "Organic_carbon": "Natural organic matter.",
        "Trihalomethanes": "By-products of chlorination; can be harmful.",
        "Turbidity": "Water clarity (NTU units)."
    }

    user_vals = []
    for i,f in enumerate(feats):
        help_txt = tooltips.get(f, f"Typical safe range: {low_safe[i]:.2f}–{hi_safe[i]:.2f}")
        user_vals.append(
            st.sidebar.slider(f.replace('_',' ').title(),
                               float(mins[i]), float(maxs[i]), float(placeholder_vals[i]),
                               step=1.0, format="%.2f", help=help_txt)
        )

    st.sidebar.markdown('---')
    if st.sidebar.button('🔄 Reset All to Tap‑Water'):
        st.experimental_rerun()

    predict = st.sidebar.button('🚀 Predict Potability')

    if predict:
        inp = np.array([user_vals])
        nan_mask = np.isnan(inp)
        if nan_mask.any():
            inp[nan_mask] = med[nan_mask[0]]
        inp[:, sk_idx] = np.log1p(inp[:, sk_idx])
        inp_scaled = (inp - mu)/sig
        prob = model.predict_proba(inp_scaled)[0]
        verdict = 'SAFE 💚' if prob>=0.5 else 'UNSAFE 🚩'

        st.metric('Prediction', verdict, f"{prob*100:.1f}% safe")
        st.progress(int(prob*100))

        st.subheader('Feature Importances (|weights|)')
        imps = np.abs(model.weights)
        st.bar_chart({feats[i]: imps[i] for i in range(len(feats))})

        with st.expander('📌 Feature Influence on this Prediction'):
            influences = inp_scaled[0] * model.weights
            infl_df = pd.DataFrame({"Feature": feats, "Influence": influences})
            infl_df = infl_df.sort_values(by="Influence", ascending=False)
            st.bar_chart(infl_df.set_index("Feature"))

        with st.expander('🔍 Detailed weights'):
            st.write('\n'.join([f"{feats[i]}: {model.weights[i]:+.4f}" for i in np.argsort(-imps)]))

        st.subheader("🚨 Out-of-Range Features")
        for i, val in enumerate(user_vals):
            if val < low_safe[i] or val > hi_safe[i]:
                st.warning(f"🔴 {feats[i]} is outside safe range! ({val:.2f})")

        with st.expander("📊 Safe Range Reference Table"):
            safe_df = pd.DataFrame({
                "Feature": feats,
                "Safe Min (5%)": low_safe,
                "Safe Max (95%)": hi_safe,
                "Median": med
            })
            st.dataframe(safe_df.style.format(precision=2))

        st.download_button("⬇️ Download Result", f"Water Status: {verdict}\nProbability: {prob*100:.2f}%\n", file_name="potability_result.txt")

if __name__=='__main__':
    main()

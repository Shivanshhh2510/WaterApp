# 💧 Water Potability Predictor

This Streamlit-based web application predicts whether a given sample of water is **safe (potable)** or **unsafe (non-potable)** for human consumption using a **custom Logistic Regression model** implemented in **NumPy**.

---

## 📌 Features

- 🧪 **Interactive Feature Input**  
  Adjust water quality parameters like pH, solids, chloramines, etc., using sliders and get instant predictions.

- 🧠 **Logistic Regression (Built from Scratch)**  
  Model trained using NumPy — no scikit-learn or external ML libraries used.

- 📊 **Prediction Confidence & Feature Importance**  
  Get a clear probability (%) of potability and understand which features influenced the result most.

- 💡 **Actionable Suggestions**  
  If the sample is unsafe, the app suggests how to improve it (e.g., increase pH, reduce solids).

- 📚 **Educational Add-ons**  
  - Random water facts  
  - Quick water quiz  
  - Feature explanations

---

## 🔧 How to Run the App Locally

```bash
1. Clone the Repository
git clone https://github.com/your-username/water-potability-predictor.git
cd water-potability-predictor

2. Install Dependencies
pip install -r requirements.txt

⚠️ If you face issues related to pandas or heavy dependencies, install just the essentials:
pip install streamlit numpy

3. Run the Streamlit App
streamlit run app.py

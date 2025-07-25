# -*- coding: utf-8 -*-
"""waterquality.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lZdT5_WAbGoWWPdDxqUImUIUysDg6_py
"""

import pandas as pd

df = pd.read_csv('/content/water_potability.csv')

print("Dataset shape:", df.shape)
print("\nColumn info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

from sklearn.impute import SimpleImputer

median_imputer = SimpleImputer(strategy='median')

cols_to_impute = ['ph', 'Sulfate', 'Trihalomethanes']
df[cols_to_impute] = median_imputer.fit_transform(df[cols_to_impute])

print("Missing values after imputation:")
print(df[cols_to_impute].isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Features to visualize (exclude target)
features = ['ph', 'Hardness', 'Solids', 'Chloramines',
            'Sulfate', 'Conductivity', 'Organic_carbon',
            'Trihalomethanes', 'Turbidity']

# 1) Histograms with KDE
plt.figure(figsize=(16, 12))
for i, col in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 2) Boxplots for outlier detection
plt.figure(figsize=(16, 12))
for i, col in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Create a copy and log1p-transform the skewed cols
df_log = df.copy()
for col in ['Solids', 'Sulfate', 'Trihalomethanes']:
    df_log[col] = np.log1p(df_log[col])

# 2) Plot the transformed distributions
plt.figure(figsize=(12, 4))
for i, col in enumerate(['Solids', 'Sulfate', 'Trihalomethanes'], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df_log[col], kde=True, bins=30)
    plt.title(f'Log1p {col}')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X = df_log.drop('Potability', axis=1)
y = df_log['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_proba))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

X = df_log.drop('Potability', axis=1)
y = df_log['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC AUC:", roc_auc_score(y_test, y_proba_rf))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6,4))
importances.plot(kind='bar')
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After  SMOTE:", y_train_sm.value_counts())

rf_sm = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf_sm.fit(X_train_sm, y_train_sm)

y_pred_sm  = rf_sm.predict(X_test)
y_proba_sm = rf_sm.predict_proba(X_test)[:, 1]

print("\n--- SMOTE + RF Classification Report ---")
print(classification_report(y_test, y_pred_sm))
print("SMOTE + RF ROC AUC:", roc_auc_score(y_test, y_proba_sm))

cm_sm = confusion_matrix(y_test, y_pred_sm)
plt.figure(figsize=(4,3))
sns.heatmap(cm_sm, annot=True, fmt="d", cmap="Blues")
plt.title("SMOTE + RF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

scale_pw = y_train.value_counts()[0] / y_train.value_counts()[1]
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pw,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_clf.fit(X_train_sm, y_train_sm)

y_pred_xgb  = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost ROC AUC:", roc_auc_score(y_test, y_proba_xgb))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(4,3))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

skewed_cols = ['Solids', 'Sulfate', 'Trihalomethanes']

X_full = df.drop('Potability', axis=1)
y_full = df['Potability']

imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X_full), columns=X_full.columns)
joblib.dump(imputer, 'imputer.joblib')


X_imp[skewed_cols] = np.log1p(X_imp[skewed_cols])
joblib.dump(skewed_cols, 'skewed_cols.joblib')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
joblib.dump(scaler, 'scaler.joblib')

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y_full)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_bal, y_bal)
joblib.dump(rf, 'rf_model.joblib')

print("Saved: imputer.joblib, skewed_cols.joblib, scaler.joblib, rf_model.joblib")

!pip install streamlit pyngrok

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# 
# def main():
#     imputer     = joblib.load('imputer.joblib')
#     skewed_cols = joblib.load('skewed_cols.joblib')
#     scaler      = joblib.load('scaler.joblib')
#     rf_model    = joblib.load('rf_model.joblib')
# 
#     df_raw   = pd.read_csv('water_potability.csv')
#     features = df_raw.columns.drop('Potability')
#     defaults = df_raw.median()
#     mins     = df_raw.min()
#     maxs     = df_raw.max()
# 
#     st.set_page_config(page_title="💧 Water Potability Predictor", layout="centered")
#     st.title("💧 Water Potability Predictor")
#     st.write("Adjust metrics below and click **Predict**.")
# 
#     user_vals = {}
#     for feat in features:
#         user_vals[feat] = st.number_input(
#             feat,
#             min_value=float(mins[feat]),
#             max_value=float(maxs[feat]),
#             value=float(defaults[feat]),
#             format="%.2f"
#         )
# 
#     if st.button("Predict Potability"):
#         inp = pd.DataFrame([user_vals])
#         inp_imp = pd.DataFrame(imputer.transform(inp), columns=inp.columns)
#         for c in skewed_cols:
#             inp_imp[c] = np.log1p(inp_imp[c])
#         inp_scaled = scaler.transform(inp_imp)
#         proba = rf_model.predict_proba(inp_scaled)[0,1]
#         verdict = "SAFE 💚" if proba >= 0.5 else "UNSAFE 🚩"
#         st.markdown(f"### **{verdict}**  (Prob: {proba:.1%})")
#         st.subheader("Feature Importances")
#         imps = pd.Series(rf_model.feature_importances_, index=features).sort_values()
#         st.bar_chart(imps)
# 
# if __name__=="__main__":
#     main()
#
# Beyond the Naked Eye — Thermography ML

Machine-learning pipeline for **infrared thermography**:
* **Regression:** estimate surface temperature from thermal imagery
* **Classification:** flag normal vs. anomalous patterns
  
This README is purposefully rewritten and **does not reuse text** from my academic submission.

---

## Why this repo exists

I wanted a clean, reusable implementation for working with IR images—separate from my academic write-up—so others can run experiments, tweak features, and compare baselines without digging through a report.

---

## What’s inside

* **Preprocessing & features:** intensity stats + simple texture descriptors (GLCM/HOG ready)
* **Two tasks, one pipeline:** shared features → split into regression and classification tracks
* **Scriptable CLI:** train/evaluate from the terminal; notebooks only for EDA
* **Reproducibility:** pinned dependencies and deterministic seeds

---

## Repo layout

```
.
├─ src/                # scripts & helpers (dataset, features, train, and evaluate)
├─ data/               # see format below
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## Data format

**Rows**: 1,020  Columns: 35  Types: 32 numeric, 3 categorical

**Categoricals**: Gender, Age (bins), Ethnicity

**Environment**: T_atm (°C), Humidity (%), Distance (camera-to-subject), T_offset1

**Thermal ROI features (examples**): Max1R13_1, Max1L13_1, aveAllR13_1, aveAllL13_1,
T_RC1 / T_LC1 (+ dry/wet/max variants), canthiMax1, canthi4Max1,
T_FHCC1, T_FHRC1, T_FHLC1, T_FHBC1, T_FHTC1, T_FH_Max1, T_FHC_Max1, T_Max1

**Reference oral temps**: T_OR1, T_OR_Max1, aveOralF, aveOralM

### Notes
• Units are in °C.
• Distance has 2 missing values in the example table; others are complete.
• Column names reflect facial regions/conditions captured by IR measurements.

## Targets & tasks

Default regression target: T_OR1 (oral reference reading).
You can switch to T_OR_Max1, T_Max1, or any numeric column via CLI.

Default classification label: binary fever-style screen derived from the chosen target
(e.g., threshold at 37.5 °C). Threshold is passed as a flag; no label is stored in the CSV.

---

## Quick start

Follow these steps to run the full pipeline locally:

1️⃣ Set up the environment

<pre>
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt </pre>

**Required packages include**:
> pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, tensorflow, imbalanced-learn

---

2️⃣ Load & prepare the dataset

The notebook uses a tabular dataset with ~1,020 rows of thermal and demographic data. You can:

- Download or generate infrared_thermography_temperature.csv
- Perform one-hot encoding on categorical variables: Gender, Age, Ethnicity
- Drop target leakage columns as needed (e.g., aveOralM if predicting aveOralF)

<pre>
import pandas as pd
df = pd.read_csv("infrared_thermography_temperature.csv")
df.dropna(inplace=True)

df = pd.get_dummies(df, columns=["Gender", "Age", "Ethnicity"], drop_first=True) </pre>

---

3️⃣ Train your model (example: XGBoost regression)
<pre> 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X = df.drop(columns=["aveOralF", "aveOralM"])   # Drop targets
y = df["aveOralF"]                               # Choose one target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train) </pre>

---

4️⃣ Evaluate the model
<pre> 
from sklearn.metrics import mean_squared_error, mean_absolute_error

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)

print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}") </pre>

---

5️⃣ Visualize feature importance
<pre> 
import matplotlib.pyplot as plt
import pandas as pd

importance = model.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({"feature": feat_names, "importance": importance}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_df["feature"], feat_df["importance"])
plt.xlabel("Feature Importance")
plt.title("XGBoost - Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show() </pre>

---

## Baselines you can expect

Tabular features: uses the numeric columns above + one-hot encodes Gender/Age/Ethnicity.
Algorithms: Random Forest (regression & classification) as strong, interpretable baselines.
Metrics: MAE/RMSE for regression; Accuracy/Macro-F1 for classification.
Outliers & QC: extremely large Distance values are flagged; you can clip or winsorize via a flag.

## Extensions you can try

- Replace RF with XGBoost/LightGBM; add Optuna tuning
- Calibrate thresholds per subgroup (Age/Ethnicity)
- Add CNNs if you have raw thermal images
- Explainability (per-feature importance; SHAP)

## Academic-integrity note

This repository is a clean implementation meant for reproducibility and collaboration.
It does not copy narrative text from my coursework; code and documentation here are rewritten and reorganized for public release.

---

## License

MIT — use and adapt with attribution.

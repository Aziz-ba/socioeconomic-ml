# 🧑‍💼 Predicting Socio-Economic Status - a Multi-Model ML Project

An end-to-end **machine-learning project** on a realistic dataset of **100,000 people** (Metropolitan France). The goal: build a predictive model for an (undisclosed) target variable describing a person's **socio-economic status**, by combining several data sources - main demographics, jobs, employment type, sports registration, and geography.

The project is deliberately structured as a **progression of models**, each adding a new signal, so you can see how much each data source contributes.

---

## 🎯 Runnable baseline & results

Beyond the notebooks, a single script trains a **Random Forest regressor** on the bundled learning set and reports honest metrics:

```bash
pip install -r requirements.txt
python src/baseline.py
```

The script (`src/baseline.py`) loads `data/learn_dataset.csv`, drops identifier columns (`UNIQUE_ID`, `insee_code`), one-hot encodes categoricals via `pd.get_dummies(drop_first=True)`, and trains a `RandomForestRegressor(n_estimators=200)` on an 80/20 train/test split.

**Result** (80/20 split, 50k rows, 67 encoded features):

| Metric | Value |
|--------|-------|
| R² | **0.68** |
| MAE | 5.49 |
| RMSE | 7.19 |

> The target is a continuous **socio-economic score** (0-92, mean 52.7).

### 💡 Insight - what drives the score
**Age and sex dominate.** `CURRENT_AGE` (0.46) and `SEX` (0.20) alone account for ~two-thirds of the model's feature importance; occupation, activity sector, degree and family type fill in the rest. A simple tree-based baseline already explains **~68% of the variance** - a strong reference point before the more elaborate models in the notebooks.

---

## 🧱 Data

The dataset is split into a **learning set (50,001)** and a **test set (49,999)**, described across multiple CSV files that must be **joined**:

| Group | Files | Content |
|-------|-------|---------|
| Main | `learn_dataset.csv`, `test_dataset.csv` | Core description + target (learn only) |
| Jobs | `*_job.csv` | Job details for employees |
| Employment | `*_Emp_type.csv` | Type of employment |
| Sports | `*_sport.csv` | Sport-club registration |
| Reference | `code_*.csv`, `regions.csv`, `departments.csv`, `city_*.csv` | Lookup / geography tables |

All files are in [`data/`](data/); the full brief is in [`SUBJECT.txt`](SUBJECT.txt).

---

## 🔬 Modeling progression

| Notebook | Focus |
|----------|-------|
| `01_exploration_preprocessing` | EDA, cleaning, joining sources (baseline features) |
| `02_modeling` | Training & evaluation framework |
| `03_model2_geography_employment` | Add geography + employment signal |
| `04_model3_geodistance` | Engineer geographic-distance features |
| `05_model4_geo_sport` | Add sport-registration signal |
| `06_model5_population` | Add commune-population signal |
| `07_departements_grouping` | Group departments to reduce cardinality |
| `08_occupation_grouping` | Group occupations to reduce cardinality |

Each step is a controlled experiment: add a feature group, measure the effect.

---

## 🚀 Run it

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the baseline model**:
   ```bash
   python src/baseline.py
   ```
   Trains the Random Forest regressor on `data/learn_dataset.csv` and prints R², MAE, RMSE, and the top 10 feature importances.

3. **Explore the progression notebooks**:
   ```bash
   jupyter notebook notebooks/01_exploration_preprocessing.ipynb
   ```
   Work through the notebooks in order (`01` to `08`). Outputs were cleared to keep the repo light - re-run to regenerate them.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

Python · pandas · numpy · scikit-learn · Jupyter

---

## 📚 What this project demonstrates

- Joining and reconciling **many messy real-world tables** into one modeling dataset
- **Feature engineering** across demographics, geography, employment and lifestyle
- A disciplined, **incremental modeling methodology** (measure each signal's contribution)
- Handling **high-cardinality categoricals** via grouping

---

## 📄 License

Released under the [MIT License](LICENSE).

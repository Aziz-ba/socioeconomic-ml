# 🧑‍💼 Predicting Socio-Economic Status — a Multi-Model ML Project

An end-to-end **machine-learning project** on a realistic dataset of **100,000 people** (Metropolitan France). The goal: build a predictive model for an (undisclosed) target variable describing a person's **socio-economic status**, by combining several data sources — main demographics, jobs, employment type, sports registration, and geography.

The project is deliberately structured as a **progression of models**, each adding a new signal, so you can see how much each data source contributes.

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

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_exploration_preprocessing.ipynb
```

Work through the notebooks in order. Outputs were cleared to keep the repo light — re-run to regenerate them.

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

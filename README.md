# Predicting Student Performance: A Dual Approach Using Classification and Regression Models

A data mining course project that applies machine learning to predict student academic performance using two datasets and two modeling paradigms — classification and regression.

---

## Project Overview

| Notebook | Dataset | Task | Target | Best Model |
|---|---|---|---|---|
| `DATA1.ipynb` | Dataset1.csv (2,392 students) | Classification | Grade Class (A–F) | SVM — 75% accuracy |
| `DATA2.ipynb` | Dataset2.csv (493 students) | Regression | Overall GPA | Random Forest — R² = 0.89 |

---

## Datasets

**Dataset 1 — High School Students**  
Features: Age, Gender, Ethnicity, Parental Education, Weekly Study Time, Absences, Tutoring, Parental Support, Extracurricular activities, Sports, Music, Volunteering.  
Target: `GradeClass` (A, B, C, D, F). Strongest predictor: `Absences` (correlation = 0.73).

**Dataset 2 — University Students**  
Features: Department, Gender, HSC/SSC scores, Income, Hometown, Computer skills, Study Preparation time, Gaming hours, Attendance, Job status, English proficiency, Extra activities, Semester.  
Target: `Overall` GPA. Strongest predictor: `Last` semester GPA (correlation = 0.93).

---

## Methodology

Both notebooks follow the same pipeline:

1. **Exploratory Data Analysis** — countplots, histograms, correlation heatmap
2. **Preprocessing** — Label encoding for categorical features, StandardScaler for numerical features
3. **Baseline Model Comparison** — multiple algorithms trained and compared
4. **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation on the best baseline model
5. **Evaluation** — Accuracy + F1-score (classification), MSE + R² (regression), confusion matrix

### Classification Models (DATA1)
Logistic Regression · KNN · SVM · Decision Tree · Random Forest · Gradient Boosting · AdaBoost · Gaussian Naive Bayes · XGBoost · CatBoost

### Regression Models (DATA2)
Linear Regression · Random Forest Regressor · Gradient Boosting Regressor · AdaBoost Regressor

---

## Repository Structure

```
├── Dataset/
│   ├── Dataset1.csv          # High school student performance data
│   └── Dataset2.csv          # University student performance data
├── Source Code/
│   ├── DATA1.ipynb           # Classification notebook
│   └── DATA2.ipynb           # Regression notebook
├── Final_Report(Literature Review)/
│   └── Final_student_performace.pdf
└── README.md
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
xgboost
catboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost catboost
```

---

## Results Summary

**Classification (Dataset 1)**
- Best model: **SVM** — Accuracy: 75%
- After GridSearchCV tuning, optimal hyperparameters selected automatically

**Regression (Dataset 2)**
- Best model: **Random Forest Regressor** — R²: 0.89, MSE: 0.11
- After GridSearchCV tuning: R² = 0.89, MSE = 0.114

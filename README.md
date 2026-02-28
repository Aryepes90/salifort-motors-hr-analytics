# Employee Attrition Prediction — Salifort Motors
### Google Advanced Data Analytics Certificate | Capstone Project

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project develops a machine learning solution to predict employee attrition at Salifort Motors, a fictional large consulting firm. The HR department collected behavioral and performance data on approximately 15,000 employees and needed a data-driven way to identify which employees are likely to leave — before they do.

Reducing voluntary turnover is a strategic priority: recruiting, interviewing, and onboarding new employees is both expensive and time-consuming. An accurate predictive model allows HR to intervene proactively with at-risk employees.

---

## Business Problem

**Can we predict whether an employee will leave the company based on their work history, workload, and performance data?**

If so, what are the primary drivers of attrition — and what can the company do about them?

---

## Dataset

- **Source:** [Kaggle — HR Analytics and Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv)
- **Size:** 14,999 rows × 10 features (after removing 3,008 duplicates from the original 15,000)
- **Target variable:** `left` — binary indicator of whether the employee departed

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee-reported job satisfaction (0–1) |
| `last_evaluation` | Score from most recent performance review (0–1) |
| `number_project` | Number of projects the employee contributed to |
| `average_monthly_hours` | Average hours worked per month |
| `time_spent_at_company` | Tenure in years |
| `work_accident` | Whether the employee experienced a workplace accident |
| `promotion_last_5_years` | Whether the employee was promoted in the last 5 years |
| `department` | Employee's department |
| `salary` | Salary level (low / medium / high) |

**Class distribution:** 83% stayed, 17% left — a moderately imbalanced dataset.

---

## Methodology

This project follows the **PACE framework** (Plan → Analyze → Construct → Execute), a structured approach to data analytics used in professional environments.

### 1. Data Cleaning
- Renamed columns to consistent `snake_case`
- Removed 3,008 duplicate rows
- Identified 824 outliers in `time_spent_at_company` via IQR method; removed for Logistic Regression modeling (tree-based models are robust to outliers)

### 2. Exploratory Data Analysis
Key findings from EDA:
- Employees who left had **lower average satisfaction levels** (20–70%) compared to those who stayed (40–80%)
- **Higher project loads** correlated with increased attrition — overloaded employees left more frequently
- Employees with **volatile monthly hours** (peaks vs. steady schedules) were more likely to leave
- **Sales, Technical, and Support** departments had the highest absolute attrition counts
- Surprisingly, **salary level had minimal predictive power** — low and medium earners actually stayed longer

### 3. Feature Engineering
Created a custom `burnout_risk` binary feature, defined as employees simultaneously carrying:
- More than 5 projects (`number_project > 5`)
- More than 250 average monthly hours
- Fewer than 3 years of tenure

This variable captured a conceptual hypothesis: that high early-career workload is a leading indicator of burnout-driven departure.

### 4. Models Built and Compared

Five models were trained and evaluated, optimizing for **recall** — the priority metric when the cost of a false negative (missing an employee about to leave) exceeds the cost of a false positive.

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression (baseline) | 45.3% | 27.4% | 34.1% | 82.2% |
| Logistic Regression + `burnout_risk` | 63.0% | 28.0% | 39.0% | 85.0% |
| Random Forest (CV) | ~99% | 90.9% | — | — |
| **Random Forest (test) ✅ Champion** | **99.3%** | **93.0%** | — | — |
| XGBoost | Comparable to RF | — | — | — |

---

## Results

**The Random Forest model without feature engineering was the champion model**, achieving:
- **Recall: 92.99%** — correctly identified 93 out of every 100 employees who would actually leave
- **Precision: 99.31%** — nearly all flagged employees were genuine attrition risks
- Scores *improved* on the test set relative to cross-validation, suggesting a well-generalized model

### Feature Importance (Random Forest)

`satisfaction_level` was the dominant predictor across all models — employees with low satisfaction were far more likely to leave. This was followed by `time_spent_at_company` and `number_project`.

Notably:
- The engineered `burnout_risk` feature **improved Logistic Regression precision by ~20%** but had negligible impact on the Random Forest, which already captured the underlying relationships through its constituent features.
- `salary` had low importance in both model families, challenging the intuitive assumption that compensation is the primary retention driver.

---

## Key Insights and Recommendations

1. **Target satisfaction proactively.** Since `satisfaction_level` is the strongest predictor of departure, the company should implement regular pulse surveys and manager check-ins — not just annual reviews.

2. **Monitor project load.** Employees assigned more than 5 concurrent projects showed disproportionately high attrition. Consider workload caps or redistribution policies.

3. **Watch the 3–5 year window.** Most departures occurred between years 3 and 5 of tenure — a critical retention window that should trigger proactive engagement, promotion reviews, and career path discussions.

4. **Rethink the salary assumption.** Compensation alone does not explain attrition in this dataset. High-salary employees left at comparable rates, suggesting non-monetary factors — autonomy, recognition, workload — may matter more.

5. **Future modeling direction:** Make `satisfaction_level` the *target variable* in a follow-on study to understand what drives satisfaction itself. This would allow even earlier upstream intervention.

---

## Ethical Considerations

- **Model transparency:** The Random Forest model's predictions should be reviewed by HR professionals, not acted upon automatically. High recall means fewer missed cases, but each prediction represents a real employee.
- **Fairness:** Department and salary were included as features. Care should be taken to ensure the model does not encode or amplify existing organizational biases in performance evaluation or promotion.
- **Data use:** Employee behavioral data carries privacy implications. Predictions should be used to support employees — not to disadvantage or surveil them.

---

## Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — Logistic Regression, Random Forest, GridSearchCV, evaluation metrics
- `xgboost` — gradient boosted classifier
- `pickle` — model serialization

---

## Project Structure

```
salifort-motors-hr-analytics/
│
├── Activity__Course_7_Salifort_Motors_project_lab.ipynb   # Main analysis notebook
├── HR_capstone_dataset.csv                                 # Source dataset
└── README.md                                               # This file
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Aryepes90/salifort-motors-hr-analytics.git
cd salifort-motors-hr-analytics

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Launch Jupyter
jupyter notebook Activity__Course_7_Salifort_Motors_project_lab.ipynb
```

---

## Certificate Context

This project is the capstone of the **Google Advanced Data Analytics Professional Certificate** (Course 7), completed as part of an ongoing transition into quantitative research and data science roles in finance. It demonstrates end-to-end ML project execution: problem framing, EDA, multi-model comparison, feature engineering, and stakeholder-oriented interpretation.

---

## Author

**Andres Rosero Yepes** | [GitHub](https://github.com/Aryepes90) | [LinkedIn](https://linkedin.com/in/andresrosero)

*Currently pursuing: M.S. Data Science — Illinois Institute of Technology | Certificate in Python for Finance (CPF) — The Python Quants*

# Stroke Risk Prediction — Shiny Web App

A supervised binary classification model predicting stroke risk using logistic regression, decision tree, and random forest in R. Includes full data wrangling, EDA, probability & statistical analysis, SMOTE-based class imbalance correction, threshold tuning, and an interactive Shiny dashboard deployed for three stakeholder groups.

**Live App:** https://bruceonyango.link/stroke/

---

## Problem

Stroke is the second leading cause of death globally and a leading cause of long-term disability. Most prediction happens too late — after symptoms appear. This model screens patients using routine clinical measurements to flag high-risk individuals before symptoms occur, enabling early intervention.

---

## Dataset

**Source:** [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
5,110 patients | 12 features | 4.87% stroke prevalence

| Feature | Type | Description |
|---|---|---|
| age | Continuous | Patient age in years |
| avg_glucose_level | Continuous | Average blood glucose (mg/dL) |
| bmi | Continuous | Body mass index (kg/m²) |
| hypertension | Binary | 1 = hypertensive |
| heart_disease | Binary | 1 = heart disease present |
| ever_married | Binary | 1 = married |
| gender | Binary | 1 = male |
| Residence_type | Binary | 1 = urban |
| work_type | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| smoking_status | Categorical | never smoked, formerly smoked, smokes, Unknown |
| stroke | Binary | 1 = had stroke (outcome variable) |

## Project Structure

```
stroke_prediction/
├── app.R                        # Shiny web application
├── stroke_analysis.R            # Full analysis script (Parts 1-6)
├── stroke_model.rds             # Saved logistic regression model
├── roc_data.rds                 # ROC data for all three models
├── stroke_data_cleaned.csv      # Cleaned dataset with engineered features
└── README.md
```

---

---

## Methods

### Data Wrangling
- BMI missing values (201/5110) — median imputation after converting to numeric
- Gender "Other" (1 record) — removed as non-informative category
- Glucose outliers — flagged using IQR method (Q3 + 1.5×IQR), not removed — clinically meaningful signal
- Binary encoding: gender, ever_married, Residence_type
- One-hot encoding: work_type, smoking_status

### Feature Engineering
Four interaction features created based on clinical logic:

| Feature | Formula | Rationale |
|---|---|---|
| glucose_bmi | glucose × bmi | Metabolic syndrome proxy |
| age_glucose | age × glucose | Compounding vascular damage |
| age_bmi | age × bmi | BMI risk amplified by age |
| hypert_age | hypertension × age | Hypertension danger compounds with age |

### Class Imbalance
SMOTE (Synthetic Minority Oversampling Technique) applied to training set only — synthetic stroke cases created by k-NN interpolation in feature space. Test set left untouched for honest evaluation.

### Models Trained
- Logistic Regression
- Decision Tree (rpart)
- Random Forest (100 trees)

All trained on 70/30 train/test split with 5-fold cross validation.

---

## Results

### Model Comparison

| Model | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **0.827** | **0.932** | 0.588 | 0.604 |
| Random Forest | 0.806 | 0.905* | 0.587* | 0.603* |
| Decision Tree | 0.753 | 0.878 | 0.627 | 0.639 |

*Random Forest at threshold 0.10 to achieve comparable sensitivity

### Winner: Logistic Regression at threshold 0.30

- Sensitivity **0.932** — catches 93 in every 100 real stroke patients
- Negative Predictive Value **99.4%** — cleared patients are almost certainly safe
- Only 5 stroke patients missed out of 74 in test set
- Interpretable coefficients — clinically explainable

### Key Predictors (Odds Ratios)

| Predictor | Odds Ratio | Interpretation |
|---|---|---|
| Hypertension | 10.37x | Strongest single risk factor |
| Heart Disease | 1.78x | 78% higher risk |
| Active Smoking | 1.46x | 46% higher risk |
| Age | 1.11x per year | 11% increase per additional year |

### Feature Importance (Random Forest)

`age` → `age_glucose` → `age_bmi` → `smoking_status` → `heart_disease`

Engineered feature `age_glucose` ranked second overall, outperforming all raw clinical variables including raw glucose and raw age individually.

---

## Hypothesis Testing

| Test | H₀ | Result |
|---|---|---|
| One-tailed t-test | No difference in mean age between stroke/no-stroke | REJECT (t=29.68, p≈0) |
| One-tailed t-test | No difference in mean age_bmi between groups | REJECT (t=23.94, p≈0) |
| Chi-square | Hypertension independent of stroke | REJECT (χ²=81.57, p≈0) |
| Chi-square | Smoking status independent of stroke | REJECT (χ²=29.23, p=0.000002) |

---

## Key Findings

- P(Stroke | Age ≥ 60) = 13.15% vs 1.82% under 60 — **7.2x higher risk**
- P(Stroke | Hypertension) = 13.25% vs 3.97% without — **3.3x higher risk**
- P(Stroke | High Glucose) = 13.4% vs 3.68% normal — **3.6x higher risk**
- P(Hypertension | Stroke) via Bayes theorem = 26.51% — confirmed directly

---

## Shiny App

Three stakeholder tabs:

| Tab | Audience | Content |
|---|---|---|
| Executive Dashboard | Hospital executives | Impact metrics, evidence charts, strategic action plan |
| Clinical Decision Support | Clinicians | ROC curves, model comparison, feature importance, threshold tuning |
| My Risk Assessment | Patients | Individual risk calculator, plain language results, personal risk factors |

---

## Strategic Recommendations

**High Risk** (Age ≥ 60 AND hypertension — ~13% stroke rate)  
→ Immediate specialist referral at every consultation

**Medium Risk** (Age ≥ 60 OR hypertension OR heart disease — ~8-10%)  
→ Quarterly monitoring, lifestyle intervention programme

**Low Risk** (Age < 60, no hypertension, no heart disease — ~1.8%)  
→ Annual standard screening, smoking history recorded

---

## Author

**Bruce Onyango**  

## Acknowledgements

Built with guidance from [Claude](https://claude.ai) (Anthropic) — used as a technical collaborator throughout the analytical pipeline including data wrangling decisions, SMOTE implementation, model evaluation strategy, threshold tuning, and Shiny app architecture.

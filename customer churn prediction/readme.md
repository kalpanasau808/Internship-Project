# Telco Customer Churn Prediction & Analytics

This repository contains a comprehensive Machine Learning workflow for predicting customer churn in the telecommunications industry using a dataset of 100,000 synthetic customer records. The project addresses data preprocessing, class imbalance handling using **SMOTE**, model training with **Logistic Regression** and **Decision Trees**, and evaluation using key classification metrics.

---

## 📌 Project Overview

Customer retention is a vital business objective for telecom companies. This project demonstrates an end-to-end Python machine learning pipeline that:
1. Loads and inspects synthetic telco customer churn data.
2. Preprocesses categorical and numerical features.
3. Addresses class imbalance (~2.02:1 ratio) using **SMOTE** (Synthetic Minority Over-sampling Technique).
4. Trains and evaluates **Logistic Regression** and **Decision Tree** classifiers on the imbalanced test set.
5. Visualizes data distributions and model performance metrics.

---

## 📊 Dataset Summary

* **Source File:** `/kaggle/input/datasets/dhrubangtalukdar/telco-customer-churn-data/synthetic_customer_churn_100k.csv`
* **Total Records:** 100,000 rows × 9 columns
* **Target Variable:** `Churn` (`Yes` / `No`)
* **Class Imbalance Ratio:** ~2.02:1 (Non-churn vs. Churn)
* **Missing Values:** 0
* **Duplicate Rows:** 0

### Features List
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | `int64` | Unique customer identifier (dropped during preprocessing) |
| `Age` | `int64` | Customer age |
| `Gender` | `object` | Customer gender (`Female`, `Male`, `Other`) |
| `Tenure` | `int64` | Length of subscription (months) |
| `MonthlyCharges` | `float64` | Monthly subscription cost |
| `Contract` | `object` | Contract type (`Month-to-month`, `One year`, `Two year`) |
| `PaymentMethod` | `object` | Payment option (`Bank transfer`, `Mailed check`, `Electronic check`, `Credit card`) |
| `TotalCharges` | `float64` | Cumulative charges |
| `Churn` | `object` | Target label (`Yes` / `No`) |

---

## 🛠️ Data Preprocessing & Pipeline Steps

1. **Identifier Removal:** Dropped non-predictive identifier columns (`CustomerID`).
2. **Target Encoding:** Converted categorical target values (`Yes` $\rightarrow$ 1, `No` $\rightarrow$ 0).
3. **Feature Encoding:** Applied One-Hot Encoding (`pd.get_dummies(drop_first=True)`) to categorical features (`Gender`, `Contract`, `PaymentMethod`).
4. **Stratified Train-Test Split:** Split data 80:20 preserving original class proportions:
   * **Training Set:** 80,000 samples (53,485 No Churn / 26,515 Churn)
   * **Test Set:** 20,000 samples
5. **Feature Scaling:** Standardized numeric columns using `StandardScaler` fitted on the training data.
6. **Class Imbalance Handling:** Applied **SMOTE** strictly on the training set to balance classes (53,485 samples each for class 0 and 1) without leaking information into the test set.

---

## 📈 Model Performance & Evaluation

Models were trained on the SMOTE-resampled training set and evaluated on the scaled, imbalanced test set (20,000 samples).

### Model Evaluation Results

| Model Architecture | Recall | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** (max_depth=10) | **0.5945** | **0.6385** | **0.6157** | **0.7991** |
| **Logistic Regression** | Evaluated | Evaluated | Evaluated | Evaluated |

---

## 💻 Dependencies

Ensure you have the following Python libraries installed before running the notebook or script:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
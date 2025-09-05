# Telco Customer Churn Prediction 📊

This project implements a machine learning pipeline to predict customer churn for a telecommunications company using the **Telco Customer Churn dataset**. The pipeline includes data preprocessing, model training with Logistic Regression and Random Forest classifiers, hyperparameter tuning using GridSearchCV, and evaluation with AUC-ROC scores. 🎯

## Table of Contents
- [Overview](#overview) ℹ️
- [Dataset](#dataset) 📋
- [Preprocessing](#preprocessing) 🛠️
- [Model Training](#model-training) 🚀
- [Evaluation](#evaluation) 📈
- [Results](#results) 🏆
- [Usage](#usage) 🖱️
- [Requirements](#requirements) 📦
- [How to Run](#how-to-run) ▶️
- [License](#license) 📜

## Overview
The project builds a predictive model to identify customers likely to churn, using a combination of numerical and categorical features. It employs a **Pipeline** with a **ColumnTransformer** for preprocessing and compares **Logistic Regression** and **Random Forest** models to determine the best performer based on AUC-ROC scores. 🌟

## Dataset
The **Telco Customer Churn dataset** contains customer data with features such as tenure, monthly charges, contract type, and more. The target variable is 'Churn', indicating whether a customer has left the service ('Yes' or 'No'). 📊

- **Source**: Provided via a CSV file.
- **Features**: Includes numerical (e.g., tenure, charges) and categorical (e.g., contract type, payment method) columns.
- **Target**: Binary classification (1 for 'Yes', 0 for 'No').

## Preprocessing
- **Numerical Features**: Scaled using **StandardScaler**.
- **Categorical Features**: Encoded using **OneHotEncoder** with `handle_unknown='ignore'`.
- **Data Split**: 80% training, 20% testing with `train_test_split` and `random_state=42`.
- **Dropped Columns**: 'customerID' is removed as it’s not a predictive feature. 🛠️

## Model Training
- **Pipeline**: Combines preprocessing and classification steps.
- **Classifiers**:
  - **Logistic Regression**: Tuned with `C` (0.001, 0.01, 0.1, 1, 10, 100) and `penalty='l2'`.
  - **Random Forest**: Tuned with `n_estimators` (50, 100, 200), `max_depth` (None, 10, 20, 30), and `min_samples_split` (2, 5, 10).
- **Hyperparameter Tuning**: Performed using **GridSearchCV** with 5-fold cross-validation and `scoring='roc_auc'`. 🚀

## Evaluation
- **Metric**: AUC-ROC score.
- **Confusion Matrices**: Generated for both models to visualize true positives, false positives, etc.
- **Best Model**: Selected based on the highest AUC-ROC score. 📈

## Results
- **Best Logistic Regression Parameters**: `{'classifier__C': 1, 'classifier__penalty': 'l2'}`.
- **Best Random Forest Parameters**: `{'classifier__max_depth': None, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 200}`.
- **AUC-ROC Scores**:
  - Logistic Regression: 0.8608 ✅
  - Random Forest: 0.8521 ✅
- **Best Model**: Logistic Regression performed better and was exported. 🏆

## Usage
1. **Install Dependencies**: Use the provided requirements.
2. **Run the Script**: Execute the Python script to train models and export the best pipeline.
3. **Load Model**: Use `joblib.load('best_churn_prediction_pipeline.joblib')` for predictions. 🖱️

## Requirements
Install the required packages using:
```bash
pip install pandas scikit-learn joblib matplotlib
```

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Script**:
   ```bash
   python churn_prediction.py
   ```
4. **Visualize Results**: Confusion matrices will be displayed automatically. ▶️

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 📜

# Shinkansen Passenger Satisfaction Prediction

A comprehensive machine learning pipeline to predict passenger satisfaction on Japan's Shinkansen Bullet Train.

## Overview

This project implements a complete ML pipeline that:
- Merges travel and survey data
- Performs data cleaning and preprocessing
- Encodes categorical features using Target Encoding
- Trains multiple models with 5-fold cross-validation
- Performs hyperparameter tuning using Optuna
- Analyzes feature importance with SHAP
- Generates final predictions for submission

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Files Required

Ensure the following CSV files are in the same directory as the script:
- `Traveldata_train.csv` - Training travel data
- `Surveydata_train.csv` - Training survey data (includes target variable)
- `Traveldata_test.csv` - Test travel data
- `Surveydata_test.csv` - Test survey data

## Usage

Run the complete pipeline:

```bash
python shinkansen_satisfaction_prediction.py
```

## Output

The script will generate:
1. **Console output**: Detailed progress, model performance metrics, and feature importance rankings
2. **`feature_importance_analysis.png`**: Comprehensive feature importance plots including tree-based importance and SHAP analysis
3. **`submission.csv`**: Final predictions in the required format (ID, Overall_Experience)

## Pipeline Steps

1. **Data Loading & Merging**: Combines travel and survey data on passenger ID
2. **Data Exploration**: Analyzes data structure, missing values, and distributions
3. **Data Cleaning**: Handles missing values and outliers
4. **Feature Encoding**: Uses Target Encoding for categorical variables
5. **Model Training**: Trains 5 different models with stratified K-fold CV
6. **Model Selection**: Selects best performing model based on accuracy
7. **Hyperparameter Tuning**: Optimizes model parameters using Optuna
8. **Feature Analysis**: Generates comprehensive feature importance analysis
9. **Final Predictions**: Creates submission file with test predictions

## Models Included

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## Key Features

- **Reproducible**: All random seeds set to 42
- **Robust**: Handles missing values and outliers
- **Comprehensive**: Includes both tree-based and SHAP feature importance
- **Optimized**: Hyperparameter tuning for best performance
- **Production-ready**: Clean, well-documented code with error handling

## Expected Performance

The pipeline typically achieves 90%+ accuracy on the validation set, with the best models usually being tree-based ensemble methods (XGBoost, LightGBM, or Random Forest). 
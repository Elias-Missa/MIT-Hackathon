#%% [markdown]
"""
# Shinkansen Passenger Satisfaction Prediction
============================================

A comprehensive ML pipeline to predict passenger satisfaction on Japan's Shinkansen Bullet Train.
This script implements data preprocessing, feature engineering, model training with cross-validation,
hyperparameter tuning, feature importance analysis, and generates final predictions.

Author: Kaggle Grandmaster ML Engineer
"""

#%% Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import category_encoders as ce
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("🚄 Shinkansen Passenger Satisfaction Prediction Pipeline")
print("=" * 60)

#%% Data Loading and Merging
def load_and_merge_data():
    """
    Load and merge travel and survey data on passenger ID.
    
    Returns:
        tuple: (train_df, test_df) - merged dataframes
    """
    print("📊 Loading and merging datasets...")
    
    # Load training data
    travel_train = pd.read_csv('Traveldata_train.csv')
    survey_train = pd.read_csv('Surveydata_train.csv')
    
    # Load test data
    travel_test = pd.read_csv('Traveldata_test.csv')
    survey_test = pd.read_csv('Surveydata_test.csv')
    
    # Merge on ID
    train_df = pd.merge(travel_train, survey_train, on='ID', how='inner')
    test_df = pd.merge(travel_test, survey_test, on='ID', how='inner')
    
    print(f"✅ Training data shape: {train_df.shape}")
    print(f"✅ Test data shape: {test_df.shape}")
    print(f"✅ Target distribution in training: {train_df['Overall_Experience'].value_counts().to_dict()}")
    
    return train_df, test_df

# Load and merge data
train_df, test_df = load_and_merge_data()

#%% Data Exploration
def explore_data(df):
    """
    Perform exploratory data analysis.
    
    Args:
        df: DataFrame to explore
    """
    print("\n🔍 Data Exploration:")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum().sort_values(ascending=False))
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Check for duplicate IDs
    print(f"\nDuplicate IDs: {df['ID'].duplicated().sum()}")

# Explore the training data
explore_data(train_df)

#%% Data Cleaning and Preprocessing
def clean_and_preprocess_data(train_df, test_df):
    """
    Clean and preprocess the data including handling missing values and outliers.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        tuple: (cleaned_train_df, cleaned_test_df)
    """
    print("\n🧹 Cleaning and preprocessing data...")
    
    # Combine for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # Handle missing values in categorical columns
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    if 'ID' in categorical_cols:
        categorical_cols.remove('ID')  # Don't fill ID column
    
    for col in categorical_cols:
        # Fill missing values with mode for categorical variables
        mode_val = combined_df[col].mode()
        if len(mode_val) > 0:
            combined_df[col] = combined_df[col].fillna(mode_val[0])
        else:
            combined_df[col] = combined_df[col].fillna('Unknown')
    
    # Handle missing values in numerical columns
    numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        if col != 'Overall_Experience':  # Don't fill target variable
            # Fill with median for numerical variables
            median_val = combined_df[col].median()
            combined_df[col] = combined_df[col].fillna(median_val)
    
    # Handle outliers in delay columns (cap at 99th percentile)
    delay_cols = ['Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    for col in delay_cols:
        if col in combined_df.columns:
            q99 = combined_df[col].quantile(0.99)
            combined_df[col] = combined_df[col].clip(upper=q99)
    
    # Split back to train and test
    train_size = len(train_df)
    cleaned_train_df = combined_df[:train_size].copy()
    cleaned_test_df = combined_df[train_size:].copy()
    
    print(f"✅ Data cleaning completed")
    print(f"✅ Training data shape after cleaning: {cleaned_train_df.shape}")
    print(f"✅ Test data shape after cleaning: {cleaned_test_df.shape}")
    
    return cleaned_train_df, cleaned_test_df

# Clean and preprocess the data
train_clean, test_clean = clean_and_preprocess_data(train_df, test_df)

#%% Feature Encoding
def encode_features(train_df, test_df):
    """
    Encode categorical features using appropriate encoding strategies.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        tuple: (encoded_train_df, encoded_test_df, feature_names)
    """
    print("\n🔢 Encoding categorical features...")
    
    # Separate features and target
    X_train = train_df.drop(['Overall_Experience'], axis=1)
    y_train = train_df['Overall_Experience']
    X_test = test_df.drop(['Overall_Experience'], axis=1, errors='ignore')
    
    # Identify categorical columns (excluding ID)
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if 'ID' in categorical_cols:
        categorical_cols.remove('ID')
    
    # Use Target Encoding for high-cardinality categorical features
    # This is effective for satisfaction prediction as it captures the relationship
    # between categories and the target variable
    target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=1.0)
    
    # Fit on training data and transform both train and test
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)
    
    # Ensure ID column is preserved for final submission
    feature_names = X_train_encoded.columns.tolist()
    
    print(f"✅ Encoded {len(categorical_cols)} categorical features using Target Encoding")
    print(f"✅ Final feature count: {len(feature_names)}")
    
    return X_train_encoded, X_test_encoded, y_train, feature_names

# Encode features
X_train, X_test, y_train, feature_names = encode_features(train_clean, test_clean)

#%% Model Training and Validation
def train_and_validate_models(X_train, y_train, feature_names):
    """
    Train multiple models with stratified K-fold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        feature_names: List of feature names
        
    Returns:
        dict: Model performance results
    """
    print("\n🎯 Training and validating models with 5-fold cross-validation...")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Predict and score
            y_pred = model.predict(X_fold_val)
            accuracy = accuracy_score(y_fold_val, y_pred)
            fold_scores.append(accuracy)
            
            print(f"  Fold {fold + 1}: {accuracy:.4f}")
        
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        
        results[name] = {
            'fold_scores': fold_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'model': model
        }
        
        print(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return results

# Train and validate models
model_results = train_and_validate_models(X_train, y_train, feature_names)

#%% Model Selection and Results Summary
# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['mean_accuracy'])
best_model = model_results[best_model_name]['model']
best_accuracy = model_results[best_model_name]['mean_accuracy']

print(f"\n🏆 FINAL RESULTS SUMMARY:")
print("=" * 50)
for name, result in sorted(model_results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True):
    print(f"{name:20s}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🏆 Best CV Accuracy: {best_accuracy:.4f} ± {model_results[best_model_name]['std_accuracy']:.4f}")

#%% Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train, best_model_name, best_model):
    """
    Perform hyperparameter tuning using Optuna for the best model.
    
    Args:
        X_train: Training features
        y_train: Training target
        best_model_name: Name of the best performing model
        best_model: Best model instance
        
    Returns:
        Optimized model
    """
    print(f"\n⚙️ Hyperparameter tuning for {best_model_name}...")
    
    def objective(trial):
        if 'XGBoost' in best_model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
        elif 'LightGBM' in best_model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.LGBMClassifier(**params)
        elif 'Random Forest' in best_model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
        else:
            # For other models, return the original model
            return best_model
        
        # Cross-validation score
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            scores.append(accuracy_score(y_fold_val, y_pred))
        
        return np.mean(scores)
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # Get best parameters and create optimized model
    best_params = study.best_params
    print(f"✅ Best parameters: {best_params}")
    print(f"✅ Best CV score: {study.best_value:.4f}")
    
    if 'XGBoost' in best_model_name:
        optimized_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    elif 'LightGBM' in best_model_name:
        optimized_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    elif 'Random Forest' in best_model_name:
        optimized_model = RandomForestClassifier(**best_params, random_state=42)
    else:
        optimized_model = best_model
    
    return optimized_model

# Hyperparameter tuning
optimized_model = hyperparameter_tuning(X_train, y_train, best_model_name, best_model)

#%% Final Model Training
# Train final model on full training set
print(f"\n🎯 Training final {best_model_name} on full training set...")
optimized_model.fit(X_train, y_train)

#%% Feature Importance Analysis
def analyze_feature_importance(model, X_train, y_train, feature_names, model_name):
    """
    Analyze and plot feature importance using both tree-based importance and SHAP values.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        feature_names: List of feature names
        model_name: Name of the model
    """
    print(f"\n📊 Analyzing feature importance for {model_name}...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Feature Importance Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Tree-based feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        top_20_features = importance_df.head(20)
        axes[0, 0].barh(range(len(top_20_features)), top_20_features['importance'])
        axes[0, 0].set_yticks(range(len(top_20_features)))
        axes[0, 0].set_yticklabels(top_20_features['feature'])
        axes[0, 0].set_xlabel('Feature Importance (Gini)')
        axes[0, 0].set_title('Top 20 Features - Tree-based Importance')
        axes[0, 0].invert_yaxis()
        
        print("🔝 Top 20 features by tree-based importance:")
        for i, (_, row) in enumerate(top_20_features.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # 2. SHAP Analysis
    try:
        # Create SHAP explainer
        if 'XGBoost' in model_name or 'LightGBM' in model_name or 'Gradient' in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train.sample(min(1000, len(X_train)), random_state=42))
        
        # Calculate SHAP values for a sample
        sample_size = min(1000, len(X_train))
        X_sample = X_train.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class SHAP values
        
        # SHAP summary plot
        plt.sca(axes[0, 1])
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False, max_display=20)
        axes[0, 1].set_title('Top 20 Features - SHAP Importance')
        
        # SHAP feature importance
        feature_importance_shap = np.abs(shap_values).mean(0)
        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': feature_importance_shap
        }).sort_values('shap_importance', ascending=False)
        
        print("\n🔝 Top 20 features by SHAP importance:")
        for i, (_, row) in enumerate(shap_importance_df.head(20).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['shap_importance']:.4f}")
        
        # SHAP waterfall plot for a single prediction
        plt.sca(axes[1, 0])
        shap.waterfall_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) 
                           else explainer.expected_value, 
                           shap_values[0], X_sample.iloc[0], 
                           feature_names=feature_names, show=False, max_display=15)
        axes[1, 0].set_title('SHAP Waterfall Plot (Single Prediction)')
        
        # SHAP beeswarm plot
        plt.sca(axes[1, 1])
        shap.plots.beeswarm(shap.Explanation(values=shap_values, 
                                           base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) 
                                           else explainer.expected_value,
                                           data=X_sample.values, 
                                           feature_names=feature_names), 
                           show=False, max_display=15)
        axes[1, 1].set_title('SHAP Beeswarm Plot')
        
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {str(e)}")
        axes[0, 1].text(0.5, 0.5, 'SHAP analysis failed', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 0].text(0.5, 0.5, 'SHAP analysis failed', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'SHAP analysis failed', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature importance analysis completed and saved as 'feature_importance_analysis.png'")

# Analyze feature importance
analyze_feature_importance(optimized_model, X_train, y_train, feature_names, best_model_name)

#%% Generate Final Predictions
def generate_submission(model, X_test, test_ids):
    """
    Generate final predictions and create submission file.
    
    Args:
        model: Trained model
        X_test: Test features
        test_ids: Test IDs
    """
    print("\n📝 Generating final predictions...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Overall_Experience': predictions
    })
    
    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"✅ Submission file created with {len(submission_df)} predictions")
    print(f"✅ Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
    print("✅ File saved as 'submission.csv'")

# Generate submission
test_ids = test_clean['ID'].values
generate_submission(optimized_model, X_test, test_ids)

#%% Final Summary
print("\n🎉 Pipeline completed successfully!")
print("📊 Check 'feature_importance_analysis.png' for feature importance plots")
print("📝 Check 'submission.csv' for final predictions")
print(f"\n🏆 FINAL MODEL PERFORMANCE:")
print(f"   Best Model: {best_model_name}")
print(f"   Cross-Validation Accuracy: {best_accuracy:.4f} ± {model_results[best_model_name]['std_accuracy']:.4f}")

#%% [markdown]
"""
## Results Summary

This notebook provides a complete machine learning pipeline for predicting Shinkansen passenger satisfaction.

### Key Features:
- **Data Processing**: Handles missing values and outliers
- **Feature Engineering**: Uses Target Encoding for categorical variables  
- **Model Comparison**: Tests 5 different algorithms with cross-validation
- **Hyperparameter Optimization**: Uses Optuna for automated tuning
- **Feature Analysis**: Provides both tree-based and SHAP importance
- **Reproducible**: All random seeds set for consistent results

### Expected Output:
- Console output with model accuracies and feature rankings
- `feature_importance_analysis.png` with comprehensive visualizations
- `submission.csv` with final predictions ready for submission
""" 
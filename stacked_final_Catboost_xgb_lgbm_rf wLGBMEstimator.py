# 📦 Install packages
!pip install catboost lightgbm xgboost shap optuna scikit-learn pandas matplotlib seaborn

# 📚 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap
import optuna
shap.initjs()

# 📁 Upload Data
from google.colab import files
uploaded = files.upload()

# 📄 Load Data
train = pd.merge(pd.read_csv('Traveldata_train.csv'), pd.read_csv('Surveydata_train.csv'), on='ID')
test = pd.merge(pd.read_csv('Traveldata_test.csv'), pd.read_csv('Surveydata_test.csv'), on='ID')

# 🧹 Data Prep
cat_cols = [col for col in train.select_dtypes(include='object') if col != 'ID']
num_cols = [col for col in train.select_dtypes(include=['int64','float64']) if col not in ['ID','Overall_Experience']]
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0], inplace=True)
for col in num_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(train[col].median(), inplace=True)
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

# 📊 Feature Visualizations
correlations = train.corr()['Overall_Experience'].abs().sort_values(ascending=False)
top_features = correlations[1:9].index.tolist()
for feature in top_features:
    plt.figure(figsize=(6, 4))
    if train[feature].nunique() < 10:
        sns.countplot(data=train, x=feature, hue='Overall_Experience')
    else:
        sns.kdeplot(data=train, x=feature, hue='Overall_Experience', fill=True)
    plt.title(f'Distribution of {feature} by Overall_Experience')
    plt.tight_layout()
    plt.show()

# 🔥 Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = train.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 🧱 Missing Data Heatmap
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# 🎯 Split Data
X = train.drop('Overall_Experience', axis=1)
y = train['Overall_Experience']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_final = test.copy()

# 🔍 Optuna tuning blocks omitted here to keep it brief (reuse from earlier version)

# 🧠 Ensemble with CatBoost Final (reuse tuned params)
stacked_model = StackingClassifier(
    estimators=[
        ('cat', CatBoostClassifier(verbose=0)),
        ('xgb', XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
        ('rf', RandomForestClassifier()),
        ('lgbm', LGBMClassifier())
    ],
    final_estimator=CatBoostClassifier(verbose=0), passthrough=True)
stacked_model.fit(X_train, y_train)
val_preds = stacked_model.predict(X_val)
print("Stacked Model Accuracy:", accuracy_score(y_val, val_preds))

# 🔁 Cross-validation
models = {
    'StackedEnsemble': stacked_model
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:25}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# 📊 ROC + Confusion Matrix
def plot_roc(model, X, y):
    y_probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title('ROC Curve')
    plt.show()
def show_cm(model, X, y):
    cm = confusion_matrix(y, model.predict(X))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title('Confusion Matrix')
    plt.show()
plot_roc(stacked_model, X_val, y_val)
show_cm(stacked_model, X_val, y_val)

# 📈 SHAP Explainability
explainer = shap.TreeExplainer(CatBoostClassifier(verbose=0).fit(X_train, y_train))
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)


# 🎯 Feature Importance from Final Estimator (CatBoost in Ensemble)
final_cat = CatBoostClassifier(verbose=0)
final_cat.fit(X_train, y_train)

importances = final_cat.get_feature_importance()
features = X_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("CatBoost Feature Importance (Final Estimator)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# 📤 Prediction Export
final_preds = stacked_model.predict(X_test_final)
submission = pd.DataFrame({'ID': X_test_final['ID'], 'Overall_Experience': final_preds})
submission.to_csv('submission.csv', index=False)
submission.head()

# 📥 Download CSV
from google.colab import files
files.download('submission.csv')


# 🎯 Problem: Predict Overall Experience (0/1) using a Neural Network
# Dataset: Shinkansen Bullet Train Travel + Survey data
# Objective: Accuracy score on test data using Neural Network + Optuna Tuning + Class Weighting

!pip install pandas scikit-learn tensorflow matplotlib seaborn optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna

from google.colab import files
uploaded = files.upload()

travel_train = pd.read_csv('Traveldata_train.csv')
survey_train = pd.read_csv('Surveydata_train.csv')
train = pd.merge(travel_train, survey_train, on='ID')

travel_test = pd.read_csv('Traveldata_test.csv')
survey_test = pd.read_csv('Surveydata_test.csv')
test = pd.merge(travel_test, survey_test, on='ID')

cat_cols = [col for col in train.select_dtypes(include='object') if col != 'ID']
num_cols = [col for col in train.select_dtypes(include=['int64','float64']) if col not in ['ID','Overall_Experience']]
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0], inplace=True)
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
for col in num_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(train[col].median(), inplace=True)

X = train.drop(['Overall_Experience'], axis=1)
y = train['Overall_Experience']
X_test_final = test.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop('ID', axis=1))
X_test_scaled = scaler.transform(X_test_final.drop('ID', axis=1))
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

def objective(trial):
    model = Sequential()
    model.add(Dense(trial.suggest_int('units1', 64, 256), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(trial.suggest_float('dropout1', 0.2, 0.5)))
    model.add(Dense(trial.suggest_int('units2', 32, 128), activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout2', 0.1, 0.3)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=30, batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
                        class_weight=class_weight_dict, callbacks=[early_stop], verbose=0)
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)
best_params = study.best_params

model = Sequential()
model.add(Dense(best_params['units1'], activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(best_params['dropout1']))
model.add(Dense(best_params['units2'], activation='relu'))
model.add(Dropout(best_params['dropout2']))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=best_params['batch_size'],
                    class_weight=class_weight_dict, callbacks=[early_stop], verbose=1)

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

val_preds = (model.predict(X_val) > 0.5).astype(int)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

final_preds = (model.predict(X_test_scaled) > 0.5).astype(int)
submission_df = pd.DataFrame({'ID': X_test_final['ID'], 'Overall_Experience': final_preds.flatten()})
submission_df.to_csv('nn_submission.csv', index=False)

files.download('nn_submission.csv')


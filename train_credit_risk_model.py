"""
Credit Risk Model Training - XGBoost 3.1.0

Trains an optimized XGBoost model for credit risk prediction with high precision and recall.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CREDIT RISK MODEL TRAINING - XGBoost 3.1.0")
print("="*60)
print(f'XGBoost version: {xgb.__version__}\n')

# ============================================================================
# 1. Load and Explore Data
# ============================================================================
print("1. Loading data...")
df = pd.read_csv('usecases/credit_risk_dataset.csv')

print(f'Dataset shape: {df.shape}')
print(f'Target distribution:\n{df["loan_status"].value_counts()}')
print(f'Default rate: {df["loan_status"].mean():.2%}\n')

# ============================================================================
# 2. Data Preprocessing
# ============================================================================
print("2. Preprocessing data...")

# Handle missing values
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
print(f'Missing values after imputation: {df.isnull().sum().sum()}')

# Encode categorical variables with LabelEncoder (model-agnostic approach)
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(f'Encoded {len(categorical_cols)} categorical variables\n')

# Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

print(f'Features shape: {X.shape}')
print(f'Feature names: {list(X.columns)}\n')

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')
print(f'Training set default rate: {y_train.mean():.2%}')
print(f'Test set default rate: {y_test.mean():.2%}\n')

# ============================================================================
# 3. Train XGBoost Model
# ============================================================================
print("3. Training XGBoost model...")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f'Scale pos weight: {scale_pos_weight:.2f}')

# XGBoost parameters optimized for precision and recall
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cpu'
}

# Train model
model = xgb.XGBClassifier(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

print('\nModel training complete!\n')

# ============================================================================
# 4. Model Evaluation
# ============================================================================
print("4. Evaluating model...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print('='*60)
print('MODEL PERFORMANCE')
print('='*60)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'ROC-AUC: {auc:.4f}')
print('='*60)

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 5 Most Important Features:')
print(feature_importance.head())

# ============================================================================
# 5. Cross-Validation
# ============================================================================
print("\n5. Running cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print('Cross-Validation ROC-AUC Scores:')
print(f'  Fold scores: {cv_scores}')
print(f'  Mean: {cv_scores.mean():.4f}')
print(f'  Std: {cv_scores.std():.4f}\n')

# ============================================================================
# 6. Save Model
# ============================================================================
print("6. Saving model...")

model_path = 'usecases/xgboost_credit_risk_new.pkl'
joblib.dump(model, model_path)

print(f'✅ Model saved to: {model_path}')
print(f'\nModel info:')
print(f'  XGBoost version: {xgb.__version__}')
print(f'  Features: {len(X.columns)}')
print(f'  Classes: {model.classes_}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall: {recall:.4f}')
print(f'  F1-Score: {f1:.4f}')
print(f'  ROC-AUC: {auc:.4f}')

# ============================================================================
# 7. Test Model Loading
# ============================================================================
print("\n7. Testing model loading...")

loaded_model = joblib.load(model_path)
test_pred = loaded_model.predict(X_test[:5])
print('Test predictions from loaded model:')
print(test_pred)

print('\n✅ Model successfully saved and loaded!')
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Heart Disease Prediction Model for HEXEval Framework

This project provides a clean, well-organized machine learning model for heart disease prediction that can be tested with the HEXEval framework to evaluate XAI (Explainable AI) methods.

## 📁 Project Structure

```
usecases/
├── heart_disease_prediction_clean.ipynb    # Clean, organized notebook
├── heart_disease_prediction.ipynb          # Original notebook (legacy)
├── heart_disease_model_wrapper.py          # Model wrapper for HEXEval
├── heart_disease_model.pkl                 # Trained model (generated after running notebook)
├── heart_disease_scaler.pkl                # Feature scaler (generated after running notebook)
└── README_heart_disease.md                 # This file
```

## 🎯 Model Overview

**Task**: Binary classification for heart disease prediction

**Model**: Logistic Regression

**Performance Metrics**:
- Accuracy: ~82%
- Precision: ~78%
- Recall: ~89%
- F1-Score: ~83%

**Features** (13 total):
1. `age` - Age in years
2. `sex` - Sex (1 = male, 0 = female)
3. `cp` - Chest pain type (0-3)
4. `trestbps` - Resting blood pressure
5. `chol` - Serum cholesterol in mg/dl
6. `fbs` - Fasting blood sugar > 120 mg/dl
7. `restecg` - Resting electrocardiographic results
8. `thalach` - Maximum heart rate achieved
9. `exang` - Exercise induced angina
10. `oldpeak` - ST depression induced by exercise
11. `slope` - Slope of peak exercise ST segment
12. `ca` - Number of major vessels colored by fluoroscopy
13. `thal` - Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

## 🚀 Quick Start

### 1. Generate the Model

Run the clean notebook to generate the model and scaler files:

```bash
jupyter notebook heart_disease_prediction_clean.ipynb
```

Or run it in Google Colab. The notebook will:
- Download the heart disease dataset
- Perform EDA and visualizations
- Train a Logistic Regression model
- Save `heart_disease_model.pkl` and `heart_disease_scaler.pkl`

### 2. Use the Model Wrapper

The model wrapper now inherits from HEXEval's standard `ModelWrapper` class, making it automatically compatible with the framework.

```python
from heart_disease_model_wrapper import HeartDiseaseModel
import numpy as np

# Initialize the model (automatically handles scaling and metadata)
model = HeartDiseaseModel()

# Make predictions
sample_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
prediction = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

print(f"Prediction: {model.class_names[prediction[0]]}")
print(f"Probabilities: {probabilities}")
```

### 3. Integrate with HEXEval

```python
from heart_disease_model_wrapper import HeartDiseaseModel, prepare_for_hexeval
from sklearn.model_selection import train_test_split
import pandas as pd
from hexeval.core.wrapper import ModelWrapper

# Load data
df = pd.read_csv('path/to/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model wrapper
# Since it inherits from ModelWrapper, it works out-of-the-box
model_wrapper = HeartDiseaseModel()

# HEXEval can now use model_wrapper directly because it conforms to the standard interface!
```

## 🔬 Testing with HEXEval Framework

The model wrapper is now fully integrated with the HEXEval core architecture.

### Benefits of the Generic Wrapper:
- **Automatic Preprocessing**: The scaler is handled internally by the base class
- **Standardized API**: `predict_proba` behaves consistently across all models
- **Metadata Handling**: Feature names and classes are managed automatically

### Example HEXEval Integration:

```python
# The wrapper works directly with XAI tools
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=model_wrapper.feature_names,
    class_names=model_wrapper.class_names,
    mode='classification'
)

# Explain a prediction
# The wrapper's predict_proba handles the scaling internally
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model_wrapper.predict_proba
)
```

## 📊 Dataset Information

**Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Size**: 1,025 samples

**Balance**: Relatively balanced (513 disease cases, 512 no disease)

**Features**: 13 clinical features + 1 target variable

## 🛠️ Requirements

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
kagglehub (for dataset download)
pickle (built-in)
```

## 🤝 Domain Context

This model demonstrates XAI evaluation in the **healthcare domain**, which is different from the credit risk domain. This helps test the domain-agnostic capabilities of the HEXEval framework.

**Clinical Interpretation Needs**:
- Medical professionals need to understand why a prediction was made
- Feature importance should align with clinical knowledge
- False negatives (missing disease) are more costly than false positives

## 📝 Notes for HEXEval Testing

1. **Model Type**: Logistic Regression - inherently interpretable, good baseline for XAI methods
2. **Feature Scaling**: StandardScaler applied - important for XAI methods that are scale-sensitive
3. **Binary Classification**: Similar to credit risk, but different domain allows testing model-agnostic properties
4. **High Recall**: Model prioritizes identifying positive cases - important for medical applications

## 🔜 Suggested Next Steps

1. **Compare XAI Methods**: Test LIME, SHAP, and other methods using HEXEval
2. **Evaluate Fidelity**: Measure how well explanations match model behavior
3. **Cross-Domain Analysis**: Compare explanation quality metrics between heart disease and credit risk models
4. **Feature Engineering**: Try polynomial features or interaction terms for improved performance

## 📄 License

This project uses the Heart Disease dataset from Kaggle, which is publicly available for research purposes.

---

**Created**: 2024
**Purpose**: Demonstrate model-agnostic and domain-agnostic capabilities of HEXEval framework
**Status**: Ready for testing ✅

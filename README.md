# Hospitalization Risk Prediction for Senior Living Populations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system for predicting 12-month hospitalization risk in senior living facilities using Medicare claims data. 
##  Project Overview

This project develops a Gradient Boosting model that:
- Achieves **0.7558 ROC-AUC** on held-out test data
- Identifies **67.1% of hospitalizations** at recommended threshold
- Provides **risk stratification** across four tiers (Low, Moderate, High, Very High)

##  Performance Metrics

| Metric | Training CV | Validation | Test |
|--------|-------------|------------|------|
| ROC-AUC | 0.7544 ± 0.0059 | 0.7504 | 0.7558 |
| PR-AUC | - | 0.3434 | 0.3440 |
| Recall @ 0.20 | - | 0.6660 | 0.6710 |
| Precision @ 0.20 | - | 0.2840 | 0.2943 |
| F1 Score @ 0.20 | - | 0.3982 | 0.4091 |

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hospitalization-risk-prediction.git
cd hospitalization-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.predictor import HospitalizationRiskPredictor

# Load the trained model
predictor = HospitalizationRiskPredictor(
    'models/hospitalization_risk_model_gradient_boosting_balanced.pkl',
    threshold=0.20
)

# Predict for a single patient
patient_data = {
    'age': 78,
    'num_chronic_conditions': 4,
    'baseline_hospital_admits': 1.0,
    'baseline_er_visits': 0.5,
    # ... other 32 features
}

result = predictor.predict(patient_data)
print(f"Risk: {result['probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
print(f"High Risk Flag: {result['high_risk_flag']}")
```

### Batch Predictions

```python
import pandas as pd

# Load patient data
patients_df = pd.read_csv('data/patients.csv')

# Generate predictions for all patients
predictions_df = predictor.batch_predict(patients_df)

# Filter high-risk patients
high_risk = predictions_df[predictions_df['high_risk_flag'] == 1]
print(f"High-risk patients: {len(high_risk)} ({len(high_risk)/len(patients_df)*100:.1f}%)")
```

##  Model Development Process

### 1. Data Splitting

```
Total: 116,352 patients
├── Training: 81,445 (70%) - model training & hyperparameter tuning
├── Validation: 11,636 (10%) - threshold optimization ONLY
└── Test: 23,271 (20%) - final evaluation (touched once)
```

### 2. Model Training

- **Algorithm**: Gradient Boosting Classifier
- **Hyperparameters**: 100 estimators, learning rate 0.05, max depth 4
- **Class Balancing**: `class_weight='balanced'` to address 16.2% positive class
- **Cross-Validation**: 5-fold stratified CV on training set only

### 3. Threshold Optimization

Tested 9 thresholds (0.10 to 0.50) on validation set:

| Threshold | Recall | Precision | F1 | Population Flagged |
|-----------|--------|-----------|----|--------------------|
| 0.15 | 83.6% | 25.5% | 0.390 | 52.0% |
| **0.20** | **66.6%** | **28.4%** | **0.398** | **37.9%** |
| 0.25 | 48.5% | 32.6% | 0.390 | 24.1% |

Threshold 0.20 selected for optimal F1 score.

### 4. Final Evaluation

Single evaluation on held-out test set confirmed performance.

## 🎛️ Threshold Selection Guide

Choose threshold based on operational capacity:

### High Recall (0.15)
- **Use Case**: Facilities with robust nursing ratios
- **Flags**: 52% of population
- **Catches**: 84% of hospitalizations
- **Precision**: 26%

### Balanced (0.20) - Recommended
- **Use Case**: Standard care coordination programs
- **Flags**: 37% of population
- **Catches**: 67% of hospitalizations
- **Precision**: 29%

### High Precision (0.25)
- **Use Case**: Limited resources or phased rollout
- **Flags**: 24% of population
- **Catches**: 49% of hospitalizations
- **Precision**: 33%

```python
# Adjust threshold dynamically
predictor.set_threshold(0.15)  # Higher recall
predictor.set_threshold(0.25)  # Higher precision
```

##  Key Features

The model uses 36 features across three categories:

**Top 5 Most Important:**
1. Number of chronic conditions (60% importance)
2. Total baseline visits (8%)
3. Emergency room utilization score (6%)
4. High-risk conditions indicator (5%)
5. Baseline outpatient cost (4%)

**Categories:**
- **Demographics**: Age, gender
- **Clinical**: Chronic conditions, disease indicators, comorbidity scores
- **Utilization**: Hospital admits, ER visits, outpatient visits, costs

##  Risk Stratification

Model provides four risk tiers:

| Risk Level | Predicted Probability | Actual Rate | Patient Count |
|------------|----------------------|-------------|---------------|
| Low Risk | 5.7% | 5.4% | 55,866 |
| Moderate Risk | 19.8% | 19.7% | 33,018 |
| High Risk | 29.2% | 29.1% | 18,740 |
| Very High Risk | 42.2% | 44.3% | 8,728 |

## 📝 Notebooks

### 01_data_exploration.ipynb
- Population demographics analysis
- Feature correlation analysis
- Class distribution assessment
- Baseline utilization patterns

### 02_model_training.ipynb
- Three-way data split implementation
- Cross-validation and hyperparameter tuning
- Model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Threshold optimization on validation set
- Final test set evaluation

### 03_deployment.ipynb
- Model loading and API demonstration
- Single patient predictions
- Batch processing examples
- Threshold adjustment demonstrations
- Operational deployment guidance

## 🛠️ API Reference

### HospitalizationRiskPredictor

```python
class HospitalizationRiskPredictor:
    def __init__(self, model_path: str, threshold: float = 0.20)
    def predict(self, patient_data: Union[Dict, pd.DataFrame]) -> Dict
    def batch_predict(self, dataframe: pd.DataFrame) -> pd.DataFrame
    def set_threshold(self, new_threshold: float) -> None
    def get_threshold_options(self) -> pd.DataFrame
    def get_feature_requirements(self) -> List[str]
    def get_model_info(self) -> Dict
```

### Example: Get Threshold Options

```python
options = predictor.get_threshold_options()
print(options[['threshold', 'recall', 'precision', 'flagged_pct']])
```

## 📦 Data Requirements

Model requires 36 features per prediction:

**Demographics (2 features)**
- Age
- Gender (encoded)

**Clinical Characteristics (14 features)**
- Number of chronic conditions
- Individual condition flags (CHF, diabetes, COPD, etc.)
- Composite risk scores

**Healthcare Utilization (20 features)**
- Baseline hospital admissions
- ER visits
- Outpatient visits
- Costs and utilization scores

See `predictor.get_feature_requirements()` for complete list.

## ⚠️ Limitations

1. **Synthetic Data**: Trained on CMS DE-SynPUF synthetic dataset. Real-world validation required before clinical deployment.

2. **Class Imbalance**: Despite balancing techniques, expect ~70% of flagged patients to not be hospitalized (precision ~30%).

3. **Temporal Decay**: Model requires quarterly retraining to maintain performance as patterns shift.

4. **Feature Availability**: Requires access to complete EHR data including diagnosis history and utilization metrics.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.


---

**Note**: This model is for research and educational purposes. Clinical deployment requires validation on actual facility data and appropriate regulatory approvals.

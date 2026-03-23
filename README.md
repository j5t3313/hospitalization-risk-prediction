# Hospitalization Risk Prediction for Senior Living Populations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project predicting 12-month hospitalization risk in senior living facilities using Medicare claims data.

## Project Overview

This project develops a Gradient Boosting model that:
- Achieves **0.7558 ROC-AUC** on held-out test data
- Identifies **67.1% of hospitalizations** at recommended threshold
- Provides **risk stratification** across four calibrated tiers (Low, Moderate, High, Very High)

## Performance Metrics

| Metric | Training CV | Validation | Test |
|--------|-------------|------------|------|
| ROC-AUC | 0.7544 ± 0.0059 | 0.7504 | 0.7558 |
| PR-AUC | - | 0.3434 | 0.3440 |
| Recall @ 0.20 | - | 0.6660 | 0.6710 |
| Precision @ 0.20 | - | 0.2840 | 0.2943 |
| F1 Score @ 0.20 | - | 0.3982 | 0.4091 |

## Project Structure

```
hospitalization-risk-prediction/
├── data/
│   └── hospitalization_risk_data.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_model_training.ipynb
├── figures/
│   ├── eda_hospitalization_risk.png
│   ├── correlation_matrix.png
│   ├── roc_pr_curves.png
│   ├── threshold_analysis.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
├── models/
│   └── hospitalization_risk_model.pkl
├── Hospitalization_Risk_Model.ipynb      # Project writeup
├── requirements.txt
├── README.md
└── LICENSE
```

### Setup

```bash
git clone https://github.com/yourusername/hospitalization-risk-prediction.git
cd hospitalization-risk-prediction

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Data

The analysis uses the CMS DE-SynPUF (Synthetic Public Use Files) Sample 1 dataset, which mimics real Medicare claims while protecting patient privacy.

**Temporal Structure:**
```
[2008: Feature Calculation Period] → [Prediction Point: Jan 1, 2009] → [2009: Outcome Period]
```

**Dataset:** 116,352 patients | 36 features | 16.2% positive class (18,816 hospitalizations)

## Model Development Process

### 1. Data Splitting

```
Total: 116,352 patients
├── Training: 81,445 (70%) — model training & hyperparameter tuning
├── Validation: 11,636 (10%) — threshold optimization only
└── Test: 23,271 (20%) — final evaluation (touched once)
```

### 2. Model Training

- **Algorithm**: Gradient Boosting Classifier
- **Hyperparameters**: 100 estimators, learning rate 0.05, max depth 4
- **Class Balancing**: `class_weight='balanced'` to address 16.2% positive class
- **Cross-Validation**: 5-fold stratified CV on training set only

### 3. Threshold Optimization

Tested 9 thresholds (0.10–0.50) on validation set:

| Threshold | Recall | Precision | F1 | Population Flagged |
|-----------|--------|-----------|----|--------------------|
| 0.15 | 83.6% | 25.5% | 0.390 | 52.0% |
| **0.20** | **66.6%** | **28.4%** | **0.398** | **37.9%** |
| 0.25 | 48.5% | 32.6% | 0.390 | 24.1% |

Threshold 0.20 selected for optimal F1 score.

### 4. Final Evaluation

Single evaluation on held-out test set confirmed generalization (ROC-AUC within 0.006 of CV estimate).

## Key Features

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

## Risk Stratification

The model provides four calibrated risk tiers:

| Risk Level | Predicted Probability | Actual Rate | Patient Count |
|------------|----------------------|-------------|---------------|
| Low Risk | 5.7% | 5.4% | 55,866 |
| Moderate Risk | 19.8% | 19.7% | 33,018 |
| High Risk | 29.2% | 29.1% | 18,740 |
| Very High Risk | 42.2% | 44.3% | 8,728 |

## Notebooks

### 01_data_preparation.ipynb
- CMS SynPUF data loading and processing
- Demographic and chronic condition feature engineering
- Baseline utilization feature extraction
- Composite feature creation
- Exploratory data analysis and correlation analysis

### 02_model_training.ipynb
- Three-way stratified data split
- Cross-validation and hyperparameter tuning
- Model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Class imbalance treatment and impact analysis
- Threshold optimization on validation set
- Final test set evaluation and visualizations

### Hospitalization_Risk_Model.ipynb
- Full project writeup with methodology, results, and analysis

## Limitations

1. **Synthetic Data**: Trained on CMS DE-SynPUF synthetic dataset. Real-world validation required before any clinical use.
2. **Class Imbalance**: Despite balancing techniques, expect ~70% of flagged patients to not be hospitalized (precision ~30%).
3. **Temporal Decay**: Model would require regular retraining to maintain performance as healthcare patterns shift.
4. **Feature Availability**: Requires access to complete EHR data including diagnosis history and utilization metrics.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

---

**Note**: This model was developed for research and educational purposes using synthetic data. Any clinical application would require validation on actual facility data.

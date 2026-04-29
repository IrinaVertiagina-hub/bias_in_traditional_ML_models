# Bias in Traditional ML Models

An end-to-end project exploring, measuring, and mitigating bias in traditional machine learning models across three real-world datasets.

## Overview

This project investigates how bias emerges in ML models trained on historically biased data, and how modern fairness tools can reduce it. It covers the full pipeline: exploratory data analysis → model training → fairness evaluation → bias mitigation → interactive web demo.

## Datasets

| Dataset | Task | Protected Attribute | Problem |
|---|---|---|---|
| **UCI Adult** | Income prediction (>$50K) | Sex | Women predicted lower income due to 1994 wage gap |
| **COMPAS** | Recidivism prediction | Race | Black defendants flagged high-risk 2x more than white |
| **German Credit** | Credit risk | Age | Younger applicants rejected more often regardless of finances |

## Models

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)

## Fairness Metrics

- **Demographic Parity Difference** — gap in positive prediction rates between groups
- **Demographic Parity Ratio** — ratio of positive rates (US EEOC 80% rule)
- **Equalized Odds Difference** — difference in error rates between groups
- **Equal Opportunity Difference** — difference in true positive rates
- **Predictive Parity Difference** — difference in precision between groups

## Mitigation Methods

- **AIF360 Reweighing** (pre-processing) — reweights training samples to reduce bias before training
- **Fairlearn ThresholdOptimizer** (post-processing) — adjusts decision thresholds after training

## Project Structure

```
bias_in_traditional_ML_models/
├── adult/
│   └── eda.py               # Exploratory data analysis for UCI Adult
├── compas/
│   └── eda.py               # Exploratory data analysis for COMPAS
├── german/
│   └── eda.py               # Exploratory data analysis for German Credit
├── results/                 # Precomputed JSON results (baseline + mitigation)
├── templates/
│   └── index.html           # Web UI
├── utils.py                 # Dataset loading and preprocessing
├── models.py                # Train models + log to MLflow
├── mitigation.py            # AIF360 + Fairlearn mitigation
├── run_all.bat              # Run all baseline experiments
├── run_all_mitigation.bat   # Run all mitigation experiments
├── app.py                   # Flask web server
└── requirements.txt
```

## Installation

```bash
pip install scikit-learn pandas numpy mlflow flask aif360 fairlearn
pip install protobuf==3.20.3  # required for AIF360
```

## Usage

### 1. Run EDA
```bash
cd adult && py eda.py
cd ../compas && py eda.py
cd ../german && py eda.py
```

### 2. Train models
```bash
# Single run
py models.py --dataset adult --model lr --C 1.0
py models.py --dataset adult --model dt --max_depth 5
py models.py --dataset adult --model svm --C 1.0 --kernel rbf
py models.py --dataset adult --model knn --n_neighbors 5

# All combinations
.\run_all.bat
```

### 3. Run mitigation
```bash
# Single run
py mitigation.py --dataset adult --model lr --C 1.0

# All combinations
.\run_all_mitigation.bat
```

### 4. View MLflow results
```bash
py -m mlflow ui
```
Open `http://localhost:5000`

### 5. Launch web demo
```bash
py app.py
```
Open `http://localhost:5050`

## Web Demo

The interactive demo allows you to:
- Select dataset, model, and hyperparameters
- View baseline fairness metrics with plain-English explanations
- Compare baseline vs AIF360 vs Fairlearn mitigation results
- Visualize bias before and after mitigation

## Key Findings

Training traditional ML models on historically biased data reproduces and sometimes amplifies real-world discrimination. All four models showed significant fairness violations across all three datasets before mitigation. Post-mitigation results show meaningful improvement in demographic parity at a small cost to overall accuracy.
```

# Tabular Regression Project

## Overview
This project implements and compares three state-of-the-art machine learning models for tabular regression tasks:
- **XGBoost**: Gradient boosting framework
- **TabNet**: Deep learning architecture specifically designed for tabular data
- **FT-Transformer**: Feature Tokenizer + Transformer architecture for tabular data

## Datasets
1. **House Prices**: Regression on housing price prediction
2. **Energy Efficiency**: Prediction of energy usage in buildings

## Project Structure
```
tabular_regression_project/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/                   # Dataset storage
│   ├── house_prices/       # House prices dataset
│   └── energy_efficiency/  # Energy efficiency dataset
├── src/                    # Source code modules
├── experiments/            # Experiment scripts and results
├── artifacts/              # Saved models and outputs
├── report/                 # Project report and presentation
└── demo/                   # Interactive demo application
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run all experiments**:
   ```bash
   python experiments/run_all_models.py
   ```

3. **View results**:
   - Metrics: `artifacts/metrics/`
   - Plots: `artifacts/plots/`
   - Models: `artifacts/models/`

## Features
- **Data Preprocessing**: Automated cleaning, encoding, and feature engineering
- **Model Training**: Implementation of XGBoost, TabNet, and FT-Transformer
- **Evaluation**: Comprehensive metrics (RMSE, MAE, R²) and comparison tables
- **Interpretability**: SHAP analysis and feature importance visualization
- **Hyperparameter Tuning**: Optuna-based optimization
- **Interactive Demo**: Streamlit application for model testing

## Results
Results are automatically saved to `experiments/results/` with detailed performance metrics and model comparisons.

## Authors
CP322 Project Team

## License
Academic use only
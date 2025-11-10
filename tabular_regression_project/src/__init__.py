"""
Tabular Regression Project

This package contains modules for training and evaluating machine learning models
on tabular regression tasks using XGBoost, TabNet, and FT-Transformer.
"""

__version__ = "1.0.0"
__author__ = "CP322 Project Team"

from . import preprocess
from . import evaluate
from . import shap_analysis

__all__ = [
    "preprocess",
    "evaluate", 
    "shap_analysis"
]
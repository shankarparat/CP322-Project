"""
XGBoost Training Module

Trains and evaluates XGBoost models for tabular regression.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime


class XGBoostRegressor:
    """
    XGBoost model wrapper for regression tasks
    """
    
    def __init__(self, **kwargs):
        """Initialize XGBoost model with parameters"""
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = xgb.XGBRegressor(**default_params)
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            early_stopping_rounds: Early stopping patience
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True


def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train XGBoost model with given parameters
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: Model parameters
        
    Returns:
        Trained XGBoost model
    """
    if params is None:
        params = {}
        
    model = XGBoostRegressor(**params)
    model.train(X_train, y_train, X_val, y_val)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }
    
    return metrics, predictions


def run_xgboost_experiment(X_train, y_train, X_test, y_test, 
                          dataset_name="dataset", save_results=True):
    """
    Run complete XGBoost experiment
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        dataset_name: Name of the dataset
        save_results: Whether to save model and results
        
    Returns:
        Results dictionary
    """
    print(f"Training XGBoost on {dataset_name}...")
    
    # Train model
    model = train_xgboost(X_train, y_train)
    
    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"XGBoost Results on {dataset_name}:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Prepare results
    results = {
        'model_name': 'XGBoost',
        'dataset': dataset_name,
        'metrics': metrics,
        'predictions': predictions,
        'feature_importance': model.get_feature_importance(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results if requested
    if save_results:
        # Save model
        model_path = f"../artifacts/models/xgboost_{dataset_name}.joblib"
        model.save_model(model_path)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df['model'] = 'XGBoost'
        metrics_df['dataset'] = dataset_name
        metrics_path = f"../artifacts/metrics/xgboost_{dataset_name}_metrics.csv"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"Model saved to: {model_path}")
        print(f"Metrics saved to: {metrics_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing XGBoost module...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Run experiment
    results = run_xgboost_experiment(
        X_train, y_train, X_test, y_test, 
        dataset_name="sample", save_results=False
    )
    
    print("XGBoost module test completed successfully!")
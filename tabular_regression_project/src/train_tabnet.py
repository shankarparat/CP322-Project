"""
TabNet Training Module

Trains and evaluates TabNet models for tabular regression.
"""

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime


class TabNetRegressorWrapper:
    """
    TabNet model wrapper for regression tasks
    """
    
    def __init__(self, **kwargs):
        """Initialize TabNet model with parameters"""
        default_params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'lambda_sparse': 1e-3,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': 2e-2},
            'mask_type': 'sparsemax',
            'scheduler_params': {'step_size': 10, 'gamma': 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 0
        }
        default_params.update(kwargs)
        self.model = TabNetRegressor(**default_params)
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              max_epochs=100, patience=10, batch_size=1024):
        """
        Train the TabNet model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            batch_size: Training batch size
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        self.model.fit(
            X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
            eval_set=eval_set,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_input = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_input).flatten()
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        
    def load_model(self, filepath):
        """Load trained model"""
        self.model.load_model(filepath)
        self.is_trained = True


def train_tabnet(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train TabNet model with given parameters
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: Model parameters
        
    Returns:
        Trained TabNet model
    """
    if params is None:
        params = {}
        
    model = TabNetRegressorWrapper(**params)
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


def run_tabnet_experiment(X_train, y_train, X_test, y_test, 
                         dataset_name="dataset", save_results=True):
    """
    Run complete TabNet experiment
    
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
    print(f"Training TabNet on {dataset_name}...")
    
    # Train model
    model = train_tabnet(X_train, y_train)
    
    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"TabNet Results on {dataset_name}:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Prepare results
    results = {
        'model_name': 'TabNet',
        'dataset': dataset_name,
        'metrics': metrics,
        'predictions': predictions,
        'feature_importance': model.get_feature_importance(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results if requested
    if save_results:
        # Save model
        model_path = f"../artifacts/models/tabnet_{dataset_name}"
        model.save_model(model_path)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df['model'] = 'TabNet'
        metrics_df['dataset'] = dataset_name
        metrics_path = f"../artifacts/metrics/tabnet_{dataset_name}_metrics.csv"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"Model saved to: {model_path}")
        print(f"Metrics saved to: {metrics_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing TabNet module...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, n_samples)
    
    # Convert to DataFrame for consistency
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(y)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Run experiment
    results = run_tabnet_experiment(
        X_train, y_train, X_test, y_test, 
        dataset_name="sample", save_results=False
    )
    
    print("TabNet module test completed successfully!")
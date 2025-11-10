"""
Data Preprocessing Module

Handles data cleaning, feature engineering, encoding, and train/test splitting
for tabular regression datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')


class TabularPreprocessor:
    """
    Comprehensive preprocessor for tabular data
    """
    
    def __init__(self, target_col=None, test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.target_encoder = TargetEncoder()
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
    def detect_column_types(self, df):
        """Automatically detect numeric and categorical columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists
        if self.target_col and self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        if self.target_col and self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
            
        return numeric_cols, categorical_cols
    
    def clean_data(self, df):
        """Basic data cleaning operations"""
        # Remove completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def handle_missing_values(self, X_train, X_test=None):
        """Handle missing values in features"""
        numeric_cols, categorical_cols = self.detect_column_types(X_train)
        
        # Handle numeric missing values
        if numeric_cols:
            X_train[numeric_cols] = self.numeric_imputer.fit_transform(X_train[numeric_cols])
            if X_test is not None:
                X_test[numeric_cols] = self.numeric_imputer.transform(X_test[numeric_cols])
        
        # Handle categorical missing values
        if categorical_cols:
            X_train[categorical_cols] = self.categorical_imputer.fit_transform(X_train[categorical_cols])
            if X_test is not None:
                X_test[categorical_cols] = self.categorical_imputer.transform(X_test[categorical_cols])
                
        return X_train, X_test
    
    def encode_categorical_features(self, X_train, y_train, X_test=None):
        """Encode categorical features using target encoding for high cardinality"""
        numeric_cols, categorical_cols = self.detect_column_types(X_train)
        
        for col in categorical_cols:
            if X_train[col].nunique() > 10:  # High cardinality - use target encoding
                self.target_encoder.fit(X_train[[col]], y_train)
                X_train[col] = self.target_encoder.transform(X_train[[col]])[col]
                if X_test is not None:
                    X_test[col] = self.target_encoder.transform(X_test[[col]])[col]
            else:  # Low cardinality - use label encoding
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                self.label_encoders[col] = le
                if X_test is not None:
                    # Handle unseen categories
                    X_test[col] = X_test[col].astype(str)
                    mask = X_test[col].isin(le.classes_)
                    X_test.loc[mask, col] = le.transform(X_test.loc[mask, col])
                    X_test.loc[~mask, col] = -1  # Assign -1 to unseen categories
                    
        return X_train, X_test
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        numeric_cols, _ = self.detect_column_types(X_train)
        
        if numeric_cols:
            X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            if X_test is not None:
                X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
                
        return X_train, X_test
    
    def preprocess_data(self, data, target_col=None, test_data=None):
        """
        Complete preprocessing pipeline
        
        Args:
            data: Training data DataFrame
            target_col: Target column name
            test_data: Optional test data DataFrame
            
        Returns:
            Preprocessed training and test sets
        """
        if target_col:
            self.target_col = target_col
            
        # Clean data
        data = self.clean_data(data)
        if test_data is not None:
            test_data = self.clean_data(test_data)
            
        # Separate features and target
        if self.target_col:
            X = data.drop(columns=[self.target_col])
            y = data[self.target_col]
        else:
            X = data
            y = None
            
        # Split data if no separate test set provided
        if test_data is None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_test = test_data if test_data is not None else None
            y_test = None
            
        # Handle missing values
        X_train, X_test = self.handle_missing_values(X_train, X_test)
        
        # Encode categorical features
        if y_train is not None:
            X_train, X_test = self.encode_categorical_features(X_train, y_train, X_test)
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        if test_data is None:
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train
    
    def get_feature_info(self, df):
        """Get information about features in the dataset"""
        info = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info


def load_house_prices_data(data_path="../data/house_prices/"):
    """Load house prices dataset"""
    train_path = f"{data_path}/train.csv"
    test_path = f"{data_path}/test.csv"
    
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None


def load_energy_efficiency_data(data_path="../data/energy_efficiency/"):
    """Load energy efficiency dataset"""
    file_path = f"{data_path}/energy_efficiency.csv"
    
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing module...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'numeric1': np.random.normal(0, 1, 100),
        'numeric2': np.random.exponential(2, 100),
        'categorical1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical2': np.random.choice(['X', 'Y'], 100),
        'target': np.random.normal(10, 5, 100)
    })
    
    # Add some missing values
    sample_data.loc[sample_data.sample(10).index, 'numeric1'] = np.nan
    sample_data.loc[sample_data.sample(5).index, 'categorical1'] = np.nan
    
    # Test preprocessor
    preprocessor = TabularPreprocessor(target_col='target')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(sample_data)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target train shape: {y_train.shape}")
    print(f"Target test shape: {y_test.shape}")
    print("Preprocessing completed successfully!")
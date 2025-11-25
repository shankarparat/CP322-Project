import pandas as pd
import numpy as np

df = pd.read_csv('../data/house_prices/Uncleaned_house_prices.csv')

df = df.drop(columns=['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'], errors='ignore')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_clean = df[numeric_cols].copy()

for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

Q1, Q3 = df_clean['SalePrice'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df_clean[(df_clean['SalePrice'] >= Q1 - 3*IQR) & 
                     (df_clean['SalePrice'] <= Q3 + 3*IQR)]

df_clean['TotalSF'] = df_clean['TotalBsmtSF'] + df_clean['GrLivArea']
df_clean['TotalBath'] = (df_clean['FullBath'] + 0.5*df_clean['HalfBath'] + 
                          df_clean['BsmtFullBath'] + 0.5*df_clean['BsmtHalfBath'])
df_clean['HouseAge'] = 2024 - df_clean['YearBuilt']
df_clean['YearsSinceRemodel'] = 2024 - df_clean['YearRemodAdd']
df_clean.to_csv('../data/house_prices/house_prices_cleaned.csv', index=False)
print(f"Shape: {df_clean.shape}")

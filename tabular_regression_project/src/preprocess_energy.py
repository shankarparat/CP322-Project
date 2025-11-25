import pandas as pd

df = pd.read_csv('../data/appliances_energy/Uncleaned_energy_appliances.csv')

df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df_clean = df.drop(columns=['date', 'rv1', 'rv2'])

cols = ['Appliances'] + [col for col in df_clean.columns if col != 'Appliances']
df_clean = df_clean[cols]

df_clean.to_csv('../data/appliances_energy/appliances_energy_cleaned.csv', index=False)
print(f"Shape: {df_clean.shape}")

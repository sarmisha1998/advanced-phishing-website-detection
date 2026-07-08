import pandas as pd
from feature_engineering import create_features


print("Loading Dataset...")

df = pd.read_csv("data/processed/processed_dataset.csv")

print(df.shape)

df = create_features(df)

print(df.head())

df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("Final Dataset Saved Successfully")
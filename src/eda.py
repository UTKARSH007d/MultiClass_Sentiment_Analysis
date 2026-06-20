import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/twitter_training.csv",
    header=None
)

# Column names
df.columns = ["ID", "Entity", "Sentiment", "Tweet"]

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())
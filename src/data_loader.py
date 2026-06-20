import pandas as pd

df = pd.read_csv(
    "data/twitter_training.csv",
    header=None
)

df.columns = ["ID", "Entity", "Sentiment", "Tweet"]

print(df.head())
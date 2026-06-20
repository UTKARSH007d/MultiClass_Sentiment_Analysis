import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/cleaned_twitter_data.csv")

# Remove missing clean tweets
df = df.dropna(subset=["Clean_Tweet"])

# Remove empty clean tweets
df = df[df["Clean_Tweet"].str.strip() != ""]

X = df["Clean_Tweet"]

vectorizer = TfidfVectorizer(max_features=5000)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF Shape:")
print(X_tfidf.shape)
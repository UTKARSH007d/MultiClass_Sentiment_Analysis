import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Stopwords
stop_words = set(stopwords.words('english'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert sentence to words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    # Convert words back to sentence
    text = " ".join(words)

    return text
df = pd.read_csv(
    "data/twitter_training.csv",
    header=None
)

df.columns = ["ID", "Entity", "Sentiment", "Tweet"]

# Remove missing tweets
df = df.dropna()

# Remove irrelevant class
df = df[df["Sentiment"] != "Irrelevant"]

# Clean all tweets
df["Clean_Tweet"] = df["Tweet"].apply(preprocess_text)

# Save cleaned dataset
df.to_csv(
    "data/cleaned_twitter_data.csv",
    index=False
)

print("Cleaned dataset saved successfully!")

print(df[["Tweet", "Clean_Tweet"]].head())
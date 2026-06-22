import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("data/cleaned_twitter_data.csv")

# Remove missing values
df = df.dropna(subset=["Clean_Tweet"])

# Remove empty tweets
df = df[df["Clean_Tweet"].str.strip() != ""]


X = df["Clean_Tweet"]

y = df["Sentiment"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train_text)

X_test = vectorizer.transform(X_test_text)


print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# -------------------------------
# Logistic Regression
# -------------------------------

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
joblib.dump(
    lr_model,
    "models/logistic_regression_model.pkl"
)

# -------------------------------
# Naive Bayes
# -------------------------------

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)


# -------------------------------
# Random Forest
# -------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)
joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# -------------------------------
# Results
# -------------------------------

print("\nModel Comparison")

print("-" * 40)

print(f"Logistic Regression : {lr_accuracy:.4f}")

print(f"Naive Bayes         : {nb_accuracy:.4f}")

print(f"Random Forest       : {rf_accuracy:.4f}")
# 🎯 Multi-Class Sentiment Analysis

A Machine Learning and NLP project that classifies text into three sentiment categories:

- Positive 😊
- Neutral 😐
- Negative 😞

The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and Machine Learning models to analyze sentiment from text reviews and comments.

---

## 📌 Project Overview

This project was developed to perform multi-class sentiment classification on textual data.

The workflow includes:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
4. Feature Engineering using TF-IDF
5. Model Training
6. Model Evaluation
7. Streamlit Web Application

---

## 📂 Dataset

Dataset contains text samples labeled as:

- Positive
- Neutral
- Negative

After preprocessing and cleaning:

- Removed missing values
- Removed irrelevant class
- Applied text normalization

Final dataset size:

- 61,121 samples

---

## 🧹 Text Preprocessing

The following NLP preprocessing steps were performed:

- Lowercasing
- Removal of punctuation
- Removal of special characters
- Stopword removal
- Lemmatization

Example:

Original:

```
I LOVE Borderlands!!! It is one of the BEST games.
```

Processed:

```
love borderland one best game
```

---

## ⚙️ Feature Engineering

TF-IDF (Term Frequency - Inverse Document Frequency) was used to convert text into numerical features.

```python
TfidfVectorizer(max_features=5000)
```

Generated Feature Matrix:

```
(59734, 5000)
```

---

## 🤖 Models Trained

### Logistic Regression

Accuracy:

```
75.96%
```

### Naive Bayes

Accuracy:

```
71.62%
```

### Random Forest

Accuracy:

```
90.73%
```

---

## 📊 Final Model Performance

Random Forest achieved the best results.

Classification Summary:

| Class | Precision | Recall | F1 Score |
|---------|---------|---------|---------|
| Negative | 0.91 | 0.92 | 0.92 |
| Neutral | 0.89 | 0.88 | 0.89 |
| Positive | 0.91 | 0.91 | 0.91 |

Overall Accuracy:

```
90.73%
```

---

## 🖥 Streamlit Application

The trained model was deployed using Streamlit.

Features:

- Text Input
- Sentiment Prediction
- Confidence Scores
- Probability Bars
- Word Count
- Character Count
- Interactive UI

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- NLTK
- Streamlit
- Joblib

---

## 🚀 How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```text
MultiClass_Sentiment_Analysis
│
├── app
├── data
├── models
├── src
├── reports
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 👨‍💻 Author
Multi-Class Sentiment Analysis

Submitted by:
Utkarsh Gupta

B.Tech Artificial Intelligence & Machine Learning
Manipal University Jaipur

Submitted to:
Appsit Technologies

Internship Project Report
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import joblib

from src.text_preprocessor import preprocess_text


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Multi-Class Sentiment Analysis",
    page_icon="🎯",
    layout="wide"
)


# ==========================
# LOAD MODEL
# ==========================

model = joblib.load(
    "models/random_forest_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("---")

st.sidebar.write("### 🤖 Model")
st.sidebar.write("Random Forest")

st.sidebar.write("### 📈 Accuracy")
st.sidebar.write("90.73%")

st.sidebar.write("### 🎯 Classes")

st.sidebar.write("""
• Positive

• Neutral

• Negative
""")

st.sidebar.markdown("---")

st.sidebar.write("### 🛠 Technologies")

st.sidebar.write("""
• Python

• NLP

• TF-IDF

• Scikit-Learn

• Random Forest

• Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.success("Internship Project")


# ==========================
# TITLE
# ==========================

st.title("🎯 Multi-Class Sentiment Analysis")

st.caption(
    "NLP • TF-IDF • Random Forest • Streamlit"
)

st.write(
    "Predict whether a customer review is Positive, Neutral, or Negative."
)

st.markdown("---")


# ==========================
# INPUT
# ==========================

review = st.text_area(
    "Enter Review:",
    height=180,
    placeholder="Type your review here..."
)


# ==========================
# REVIEW STATS
# ==========================



# ==========================
# BUTTON
# ==========================

if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning(
            "Please enter a review."
        )

    else:

        # ==================
        # PREPROCESS
        # ==================

        clean_review = preprocess_text(
            review
        )

        review_vector = vectorizer.transform(
            [clean_review]
        )

        prediction = model.predict(
            review_vector
        )[0]

        probabilities = model.predict_proba(
            review_vector
        )[0]

        st.markdown("---")

        # ==================
        # RESULT
        # ==================

        if prediction == "Positive":

            st.success(
                f"🟢 Prediction: {prediction}"
            )

        elif prediction == "Neutral":

            st.warning(
                f"🟠 Prediction: {prediction}"
            )

        else:

            st.error(
                f"🔴 Prediction: {prediction}"
            )

        # ==================
        # CONFIDENCE SCORES
        # ==================

        st.subheader(
            "📊 Confidence Scores"
        )

        for label, prob in zip(
            model.classes_,
            probabilities
        ):

            st.write(
                f"**{label}** : {prob:.2%}"
            )

            st.progress(
                float(prob)
            )
            word_count = len(review.split())

            char_count = len(review)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Words", word_count)

            with col2:
                st.metric("Characters", char_count)


        # ==================
        # PROCESSED TEXT
        # ==================

        st.markdown("---")

        st.subheader(
            "🧹 Processed Text"
        )

        st.info(clean_review)


# ==========================
# EXAMPLES
# ==========================

st.markdown("---")

st.subheader("🧪 Example Reviews")

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        "Positive:\n\nI absolutely love this game."
    )

with col2:
    st.warning(
        "Neutral:\n\nVersion 2.0 was released today."
    )

with col3:
    st.error(
        "Negative:\n\nThis is the worst update ever."
    )


# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.markdown(
    """
    <center>
   © 2026 Multi-Class Sentiment Analysis
<br>
Built with Python, NLP, TF-IDF, Random Forest & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)
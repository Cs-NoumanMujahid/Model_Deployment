import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

st.write("Limited Usage (2-3 Prediction attempts): The storage allocated by PythonAnywhere was Insufficient for model to deploy. So Streamlit is being used, which allows limited prediction attempts")

# Page config
st.set_page_config(
    page_title="Suicide Detection Predictor",
    page_icon="🎗️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f9;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #5cb85c;
        color: white;
        font-weight: bold;
    }

    /* Text area styling */
    textarea {
        background-color: black !important;
        color: white !important;
    }

    /* Top disclaimer box */
    .top-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #fff3cd;
        color: #856404;
        margin-bottom: 15px;
        font-size: 14px;
    }

    /* Bottom disclaimer */
    .bottom-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #f8d7da;
        color: #721c24;
        margin-top: 25px;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def main():

    st.title("🎗️ Suicide Detection Predictor")

    # Top disclaimers
    st.markdown("""
    <div class="top-box">
      <b>Academic Project:</b> This application is developed for a university assignment and is not a clinical or diagnostic tool.<br>
      <b>Limited Training Data:</b> The model is trained on a small sample (~100 instances). For real-world use, significantly larger datasets are required.
    </div>
    """, unsafe_allow_html=True)

    
    st.write("Enter text below to analyze potential risk patterns:")
    

    user_input = st.text_area(
        "Text to Analyze",
        placeholder="Type your text here...",
        height=150
    )

    if st.button("Analyze Text"):
        if user_input.strip():
            try:
                if os.path.exists('naive_bayes_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):

                    model = joblib.load('naive_bayes_model.pkl')
                    vectorizer = joblib.load('tfidf_vectorizer.pkl')

                    cleaned_text = preprocess_text(user_input)
                    X_tfidf = vectorizer.transform([cleaned_text])

                    prediction = model.predict(X_tfidf)[0]

                    st.subheader("Result:")

                    if str(prediction).lower() == "suicide":
                        st.error(" Potential Risk Detected")
                    else:
                        st.success(" Low Risk")

                else:
                    st.error("Model files not found in directory.")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")

    # Example inputs
    st.markdown("###  Try Example Inputs")

    st.code("""Happy Texas day to all the Texans here (: have a great day
Anyone here draw and wanna work on something w me? You don't have to be really good or anything, but I love comics
We did it! My friends and I have a youtube channel that just hit 100 subs!
hi i want new friends because my friends suck lmao a little about me i'm hella bored""")

    st.markdown("**Test for potential risk:**")

    st.code("""I don't see any point in existing. Life is too much of a chore. I just want to call it a day now, forever.
I have nothing 90% of my family despises me, I have no friends, I do nothing but drink all day""")

    # Bottom disclaimer
    st.markdown("""
    <div class="bottom-box">
     This model detects patterns in text based on training data (100 rows) and may produce incorrect results.  
    It should not be used for mental health assessment or decision-making.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

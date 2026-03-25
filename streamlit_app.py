import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# Set page config for a premium look
st.set_page_config(page_title="Suicide Detection Predictor", page_icon="🎗️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #5cb85c;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Download NLTK data
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
    st.title("Suicide Detection Predictor")
    st.info("Enter text below to predict if the text indicates suicidal tendency or not.")

    user_input = st.text_area("Text to Analyze", placeholder="Type your text here...", height=150)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            try:
                if os.path.exists('naive_bayes_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
                    model = joblib.load('naive_bayes_model.pkl')
                    vectorizer = joblib.load('tfidf_vectorizer.pkl')
                    
                    cleaned_text = preprocess_text(user_input)
                    X_tfidf = vectorizer.transform([cleaned_text])
                    
                    prediction = model.predict(X_tfidf.toarray())[0]
                    
                    st.subheader("Result:")
                    if prediction.lower() == "suicide":
                        st.error(f"⚠️ Prediction: {prediction.upper()}")
                    else:
                        st.success(f"✅ Prediction: {prediction.upper()}")
                else:
                    st.error("Error: Model files (.pkl) not found in the directory.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter some text for analysis.")

if __name__ == "__main__":
    main()

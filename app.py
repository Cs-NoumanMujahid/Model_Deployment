from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Ensure stopwords are downloaded when the app starts
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    user_input = ""
    
    if request.method == 'POST':
        user_input = request.form.get('text', '')
        if user_input.strip():
            # Load models (ensure they exist first)
            try:
                model = joblib.load('naive_bayes_model.pkl')
                vectorizer = joblib.load('tfidf_vectorizer.pkl')
                
                cleaned_text = preprocess_text(user_input)
                X_tfidf = vectorizer.transform([cleaned_text])
                
                # Predict
                pred = model.predict(X_tfidf.toarray())
                prediction = pred[0]
            except Exception as e:
                prediction = f"Error: {str(e)} (Ensure you have run the Notebook to generate the .pkl files first!)"
                
    return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)

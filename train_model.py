import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import nltk
from nltk.corpus import stopwords
import sys

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def main():
    print("Loading data...")
    df = pd.read_csv('cleaned_sample_suicide_detection.csv')
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    X = df['cleaned_text']
    y = df['class']
    
    print("Splitting and Vectorizing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Naive Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Saving Models to .pkl...")
    joblib.dump(model, 'naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Done! You can now start app.py")

if __name__ == "__main__":
    main()

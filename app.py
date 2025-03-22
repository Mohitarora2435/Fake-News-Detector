from sklearn.feature_extraction.text import TfidfVectorizer  # Try importing again
import streamlit as st
import pandas as pd
import re
import string
from sklearn.linear_model import LogisticRegression

# Load the trained model (train your model first if needed)
import joblib

# Load the vectorizer and model
vectorization = joblib.load("vectorizer.pkl")  # Save your vectorizer as vectorizer.pkl
model = joblib.load("model.pkl")  # Save your model as model.pkl

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter a news article to check if it's **Fake** or **Real**")

user_input = st.text_area("Enter News Content Here")

if st.button("Check"):
    if user_input:
        processed_input = wordopt(user_input)
        transformed_input = vectorization.transform([processed_input])
        prediction = model.predict(transformed_input)
        result = "Fake News" if prediction[0] == 0 else "Not Fake News"
        
        st.subheader("Prediction Result:")
        st.write(f"📰 {result}")

    else:
        st.warning("Please enter some news text to analyze.")


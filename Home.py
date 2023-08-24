import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC

st.title("Fake News Detector")
st.caption("Aimed at political artcles")
user_input = st.text_area('Please enter your article')

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_words(new_tokens):
    new_tokens = [t.lower() for t in new_tokens]
    stop_words = set(stopwords.words('english'))
    new_tokens = [t for t in new_tokens if t not in stop_words]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
    return new_tokens


@st.cache_data
def read_clean_data():
    data = pd.read_csv('fake_or_real_news.csv')
    data['fake'] = data.label.map({'FAKE': 1, 'REAL': 0})
    X, y = data['text'], data['fake']
    # X = [' '.join(clean_words(word_tokenize(text))) for text in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = read_clean_data()

X_train_vectorized = vectorizer.fit_transform(
    X_train)  # Fit the vectorizer once
X_test_vectorized = vectorizer.transform(X_test)


@st.cache_resource
def load_model():
    clf = LinearSVC()
    clf.fit(X_train_vectorized, y_train)
    score = clf.score(X_test_vectorized, y_test)
    st.info(f"{score*100}% model test accuracy")
    return clf


# Predict Button
if st.button('Predict'):
    with st.spinner('Predicting....'):

        time.sleep(0.5)  # Simulate prediction
        model = load_model()
        user_input_cleaned = clean_words(word_tokenize(user_input))
        vectorized_text = vectorizer.transform(user_input_cleaned)
        prediction = model.predict(vectorized_text)
        # st.snow()
        st.toast("Model has run successfully")
        if prediction[0] == 0:
            st.success("The article is legitimate 🥳")
        else:
            st.error("The article is fake 🛑")

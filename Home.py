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
from sklearn.linear_model import LogisticRegression
import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession

st.set_page_config(layout="wide")


st.title("News Integrity Analyzer")
st.caption("Aims to reduce misinformation")
user_input = st.text_area('Please enter your article headline')

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_source(url):
    try:
        session = HTMLSession()
        response = session.get(url)
        return response
    except requests.exceptions.RequestException as e:
        print(e)
        return None

# links only


def scrape_google(query):

    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.com/search?q=" + query)

    links = list(response.html.absolute_links)
    google_domains = ('https://www.google.',
                      'https://google.',
                      'https://webcache.googleusercontent.',
                      'http://webcache.googleusercontent.',
                      'https://policies.google.',
                      'https://support.google.',
                      'https://maps.google.')

    for url in links[:]:
        if url.startswith(google_domains):
            links.remove(url)

    return links

# make a streamlit link table


def link_table(links):
    data_df = pd.DataFrame({"Links": links})
    st.data_editor(data_df, column_config={
                   "Links": st.column_config.LinkColumn("Trending Links")})


# links text and title
def get_search_results(query):
    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.com/search?q=" + query)

    return response


def parse_results(response):
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".VwiC3b"

    results = response.html.find(css_identifier_result)

    output = []
    try:
        for result in results:

            item = {
                'title': result.find(css_identifier_title, first=True).text,
                'link': result.find(css_identifier_link, first=True).attrs['href'],
                'text': result.find(css_identifier_text, first=True).text
            }

            output.append(item)

        return output
    except:
        return None


def google_search(query):
    response = get_search_results(query)
    return parse_results(response)


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
    data = pd.read_csv('Liar.csv')
    X, y = data['statement'], data['label']
    X = [' '.join(clean_words(word_tokenize(text))) for text in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = read_clean_data()

X_train_vectorized = vectorizer.fit_transform(
    X_train)  # Fit the vectorizer once
X_test_vectorized = vectorizer.transform(X_test)


@st.cache_resource
def load_model():
    clf = LogisticRegression()
    clf.fit(X_train_vectorized, y_train)
    score = clf.score(X_test_vectorized, y_test)
    return clf


# Predict Button
if st.button('Predict'):
    with st.spinner('Predicting....'):

        # links = scrape_google(user_input)
        # link_table(links)

        time.sleep(1)  # Simulate prediction
        model = load_model()
        user_input_cleaned = ' '.join(clean_words(word_tokenize(user_input)))

        vectorized_text = vectorizer.transform([user_input_cleaned])
        prediction = model.predict(vectorized_text)
        # st.snow()
        st.toast("Model has run successfully")

        if prediction[0] in ['true', 'mostly-true']:
            st.success(f"The article is classified as {prediction[0]}.🥳")
        elif prediction[0] in ['half-true', 'barely-true']:
            st.warning(f"The article is classified as {prediction[0]}.🤨")
        else:
            st.error(f"The article is classified as {prediction[0]}.🛑")

        label_explanations = {
            'true': 'The headline is accurate and true.',
            'mostly-true': 'The headline is mostly true with minor inaccuracies.',
            'half-true': 'The headline is partially true and partially false.',
            'barely-true': 'The headline is slightly true but mostly inaccurate.',
            'false': 'The headline is not true and contains false information.',
            'pants-fire': 'The headline contains blatant lies and false information.'
        }

        st.subheader("Output Scale")
        explanations_df = pd.DataFrame(label_explanations.items(), columns=[
            'label', 'explanation'])
        st.dataframe(explanations_df, hide_index=True,
                     use_container_width=True)

    st.subheader("Related News")

    with st.spinner('Scraping Reviews'):
        googleres = google_search(user_input)
        if googleres is not None:
            st.dataframe(googleres, hide_index=True, use_container_width=True)
        else:
            st.error("No related articles found")

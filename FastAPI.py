import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


def clean_words(new_tokens):
    new_tokens = [t.lower() for t in new_tokens]
    stop_words = set(stopwords.words('english'))
    new_tokens = [t for t in new_tokens if t not in stop_words]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
    return new_tokens


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

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


app = FastAPI()  # FastAPI object app


@app.get('/')
def Hello():
    return {'message': 'Hello FastAPI worked'}


@app.post('/predictHeadline')
def predict_news(data: str):
    user_input_cleaned = ' '.join(clean_words(word_tokenize(data)))
    vectorized_text = vectorizer.transform([user_input_cleaned])
    prediction = model.predict(vectorized_text)
    print(prediction)
    return {
        # Convert prediction to a list for JSON serialization
        'prediction': prediction.tolist()
    }


@app.post('/Falcon')
def chat(data: str):
    model = "tiiuae/falcon-40b"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(data,
                         max_length=200,
                         do_sample=True,
                         top_k=10,
                         num_return_sequences=1,
                         eos_token_id=tokenizer.eos_token_id,
                         )

    generated_text = sequences[0]['generated_text']
    return {
        # Return output
        'output': generated_text
    }


# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn FastAPI:app --reload

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import re
import string
import gensim.downloader as api

# Preprocess
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # remove HTML tags and URLs
    text = re.sub(r'http\S+', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    # remove non-alphabetical characters and digits
    text = re.sub('[^a-zA-Z\s]', '', text)
    # convert to lowercase
    text = text.lower()
    # tokenize the text
    tokens = word_tokenize(text)
    # remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # join the tokens back into a string
    text = ' '.join(lemmatized_tokens)
    return text

tfidf = pickle.load(open('tf-idf-2.pkl','rb'))
model = pickle.load(open('knn-tf-idf2.pkl','rb'))

st.title("Financial Sentiment Analysis ")

input_sms = st.text_area("Enter the message")

if st.button('Predict 👈'):

    # 1. preprocess
    transformed_sms = preprocess(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header('Negative Review 😞')
    elif result == 1:
        st.header('Neutral Review 😐')
    elif result == 2:
        st.header('Positive Review 😃')

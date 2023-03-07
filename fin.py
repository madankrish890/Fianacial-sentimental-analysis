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

import gensim.downloader as api

# Download the Google News word2vec model
model = api.load('word2vec-google-news-300')
svc_clf = pickle.load(open('svc.pkl', 'rb'))

st.title('Financial Sentiment analysis')

input_review=st.text_input('Enter the review')

if st.button('Predict'):
    transform_review = preprocess(input_review)
    text_vec = np.mean(model[transform_review], axis=0)
    text_vec = text_vec.reshape(1, -1)
    # Make sure text_vec has the expected dimensions
    # predict
    result = svc_clf.predict(text_vec)
    # Display
    if result == 'positive':
        st.header('Positive Review')
    elif result == 'negative':
        st.header('Negative Review')
    elif result == 'neutral':
        st.header('neutral')

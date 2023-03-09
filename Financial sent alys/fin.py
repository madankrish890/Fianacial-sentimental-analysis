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


# Download the Google News word2vec model
#model = api.load('word2v-google-news-300')
model = pickle.load(open('w2vec_model.pkl','rb'))
knn_clf = pickle.load(open('knn_clf.pkl', 'rb'))

st.title('Financial Sentiment analysis')

input_review=st.text_area("Enter the message")

if st.button('Predict'):
    transform_review = preprocess(input_review)
    sent_vec = np.zeros(300)
    count = 0
    for word in transform_review.split():
        if word in model.wv.key_to_index:
            vec = model.wv[word]
            sent_vec += vec
            count += 1
    if count != 0:
        sent_vec /= count
    text_vec = sent_vec.reshape(1, -1)  # reshape to match the expected input dimensions
    # predict
    result = knn_clf.predict(text_vec)
    # Display
    if result == -1:
        st.header('Negative Review')
    elif result == 0:
        st.header('Neutral Review')
    elif result == 1:
        st.header('Positive Review')


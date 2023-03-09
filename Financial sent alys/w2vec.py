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
import gensim
import joblib
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

model = gensim.models.KeyedVectors.load_word2vec_format("E:\word2vec\GoogleNews-vectors-negative300.bin", binary=True)

svc_clf = pickle.load(open('svc_pretrained.pkl', 'rb'))

st.title('Financial Sentiment analysis')

input_review=st.text_area("Enter the message")

if st.button('Predict'):
    transform_review = preprocess(input_review)
    words = [word for word in transform_review if word in model.key_to_index]
    text_vec = np.mean(model[words], axis=0)
    text_vec = text_vec.reshape(1, -1)
    # reshape to match the expected input dimensions
    # predict
    result = svc_clf.predict(text_vec)
    # Display
    if result == 'negative':
        st.header('Negative Review')
    elif result == 'neutral':
        st.header('Neutral Review')
    elif result == 'positive':
        st.header('Positive Review')


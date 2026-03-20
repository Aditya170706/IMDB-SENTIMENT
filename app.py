import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import pickle
import re
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.preprocessing.sequence import pad_sequences   # ✅ keras not tensorflow.keras

# ── Load model & tokenizer ──────────────────────────────
@st.cache_resource
def load_everything():
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_everything()
max_len = 300

# ── Clean text ──────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(sentence):
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence)
    sentence = sentence.lower()
    words = sentence.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ── Predict ─────────────────────────────────────────────
def predict_sentiment(review):
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    return prob

# ── UI ───────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Analyser")
st.write("Enter a movie review and find out if it's Positive or Negative!")

review = st.text_area("Your Review", placeholder="Type your movie review here...", height=150)

if st.button("Analyse Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analysing..."):
            prob = predict_sentiment(review)
            confidence = prob if prob > 0.5 else 1 - prob
            if prob > 0.5:
                st.success(f"😊 POSITIVE  —  Confidence: {confidence*100:.1f}%")
            else:
                st.error(f"😞 NEGATIVE  —  Confidence: {confidence*100:.1f}%")
            st.progress(float(prob))
            st.caption(f"Raw score: {prob:.4f}  |  0 = Negative, 1 = Positive")

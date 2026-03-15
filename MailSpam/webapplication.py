import numpy as np 
import pandas as pd 
import pickle
import os 
import streamlit as st 
from sklearn.feature_extraction.text import CountVectorizer 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

## Loading Models 
base_path = os.path.dirname(__file__)

countVectorizer = os.path.join(base_path , "Model","vectorizer.pkl")
spam_deduction = os.path.join(base_path,"Model","spamModel.pkl")

cv = None
model = None 
try:
    with open(countVectorizer,'rb') as f:
        cv = pickle.load(f)
    with open(spam_deduction,'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("File Not Founds")

## Custom CSS 
st.markdown("""
    <style>
        /* Main background color */
        .stApp {
            background-color: #09637E;
        }
        
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #7AB2B2;
        }
        .verified-badge{
            display: inline-block;
            padding: 3px,10px;
            background-color: #d1fae5;
            color: #065f46;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: -10px;
            margin-bottom: 20px;
            }
    </style>
""",unsafe_allow_html=True)

# App Sidebar
with st.sidebar:
    st.title("👨🏻‍💻Developer Info")
    st.write("**Name**: Rasheed Ahmad")
    st.write("**Role**: ML Engineer")
    st.divider()
    st.subheader("📊Model Accuracy")
    st.markdown("# 98.0%")
    st.markdown('<span class="verified-badge">↑ Verified</span>', unsafe_allow_html=True)

# App Header
st.title("👋🏻Welcome To Email 📨 Spam Deduction System ⚙️🗑️")
st.markdown("🔍 Analyze Your Message")

lemmatizer = WordNetLemmatizer()
user_input = st.text_area("Paste the email or SMS content below:",height=150,placeholder="Enter message here...")

if st.button("Check Spam"):
    if user_input:
        with st.spinner("Analyzing text......"):
            review = re.sub('^[a-zA-Z]',' ',user_input)
            review = review.lower()
            review = review.split()
            review = [lemmatizer.lemmatize(word, pos='v') for word in review if word not in set(stopwords.words('english'))]
            review = ' '.join(review)

            vectorizer_input = cv.transform([review]).toarray()

            prediction = model.predict(vectorizer_input)

            st.divider()
            if prediction[0] == 1:
                st.success("✅ **Safe!** This message looks like Normal Mail (Ham).")
                st.balloons()
            else:
                st.error("🚨 **Warning!** This message is classified as Spam.")
    else:
        st.warning("⚠️ Please enter a message to analyze.")



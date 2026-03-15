# Machine-Learning-End-To-End-ML-Email-Spam-Detection-System-Project
# 📧 Email Spam Detection System ⚙️🗑️
A machine learning web application built to accurately classify emails and SMS messages as either Spam (malicious/junk) or Ham (safe/normal).

# 🚀 Project Overview
This project leverages Natural Language Processing (NLP) and a Multinomial Naive Bayes classification model to analyze text messages. It features a clean, interactive web interface built with Streamlit, allowing users to paste messages and get real-time spam predictions.

# ✨ Key Features
Advanced Text Preprocessing: Utilizes regular expressions, NLTK stopword removal, and WordNet Lemmatization to thoroughly clean and standardize raw text before analysis.

## Bag of Words Vectorization: 
Converts text into numerical features using Scikit-Learn's CountVectorizer (configured with a 2,500 max feature limit and bigrams).

## High Accuracy ML Model: 
Trained on the SMS Spam Collection dataset, achieving a 98.0% accuracy rate with robust precision and recall.

## Interactive Web UI: 
Deployed via Streamlit with custom CSS for a professional, responsive user experience (complete with visual badges and dynamic animations).

## Pipeline Integrity: 
Carefully structured to prevent data leakage by isolating the fit_transform and transform steps between training and testing datasets.

# 🛠️ Technology Stack
Language: Python

Machine Learning: Scikit-Learn (MultinomialNB, CountVectorizer)

NLP Processing: NLTK (WordNetLemmatizer, stopwords)

Data Manipulation: Pandas, NumPy

Frontend/Deployment: Streamlit

Model Serialization: Pickle

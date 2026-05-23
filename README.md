# 🇮🇳 Indian Social Media Trend Analysis & Content Intelligence System

## 📌 Overview

This project analyzes Indian social media data (Reddit) using NLP
techniques to identify trends, sentiment, keywords, hashtags, and
engagement patterns. It also includes a Streamlit dashboard for
visualization and interaction.

------------------------------------------------------------------------

## 🚀 Features

### 🔹 Data Cleaning

-   Removed noise, URLs, mentions
-   Normalized text

### 🔹 Topic Classification

-   Rule-based classification into:
    -   Politics, Sports, Economy, Technology, Education, Entertainment,
        Social Issues

### 🔹 Sentiment Analysis

-   Positive, Negative, Neutral classification
-   Topic vs Sentiment comparison

### 🔹 Trend Analysis

-   Identified most discussed topics
-   Visualized topic distribution

### 🔹 Keyword Extraction

-   TF-IDF based keyword extraction

### 🔹 Hashtag Generator

-   Topic-wise hashtags using filtered keywords

### 🔹 Engagement Prediction

-   Logistic Regression model
-   Output: High / Medium / Low (approximate)

### 🔹 Streamlit Dashboard

-   KPI metrics
-   Trend charts
-   Sentiment graphs
-   Hashtag generator
-   User input analysis

------------------------------------------------------------------------

## 🧠 Tech Stack

-   Python
-   Pandas, NumPy
-   Matplotlib, Seaborn
-   Scikit-learn
-   Streamlit
-   Regex

------------------------------------------------------------------------

## 📊 Insights

-   Technology dominates discussions
-   Politics is polarized
-   Sports mostly positive
-   Social issues mostly negative

------------------------------------------------------------------------

## 📂 Structure

project/ │ ├── data/ ├── app.py ├── notebook.ipynb ├── report.pdf ├──
README.md

------------------------------------------------------------------------

## ⚙️ Run

pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run app.py

------------------------------------------------------------------------

## ⚠️ Limitations

-   Rule-based topics
-   No real engagement data
-   Prediction is approximate

------------------------------------------------------------------------

## 🔮 Future Work

-   ML-based classification
-   Real-time data
-   Better prediction

------------------------------------------------------------------------

## 👨‍💻 Author

Abhishek Mishra

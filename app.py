import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Trend Analyzer", layout="wide")

st.title("🇮🇳 Indian Social Media Trend Analyzer")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("indian_dataset_clean.csv")

df = load_data()

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 10]

# =========================
# TOPIC CLASSIFICATION
# =========================
topic_dict = {
    "Politics": ["election", "government", "modi", "bjp"],
    "Sports": ["cricket", "match", "team"],
    "Economy": ["jobs", "market", "price"],
    "Technology": ["tech", "software", "data"],
    "Education": ["school", "college", "students"],
    "Entertainment": ["movie", "music", "film"],
    "Social Issues": ["women", "crime", "police"]
}

def assign_topic(text):
    for topic, keywords in topic_dict.items():
        if any(word in text for word in keywords):
            return topic
    return "Other"

df["topic"] = df["clean_text"].apply(assign_topic)

df = df[df["topic"] != "Other"]

# =========================
# 📊 KPI SECTION
# =========================
st.subheader("📌 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Posts", len(df))
col2.metric("Unique Topics", df["topic"].nunique())
col3.metric("Avg Text Length", int(df["clean_text"].str.len().mean()))

# =========================
# 🔥 TRENDING TOPICS
# =========================
st.subheader("🔥 Trending Topics")

topic_counts = df["topic"].value_counts()

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=topic_counts.values, y=topic_counts.index, ax=ax)
    ax.set_title("Top Topics")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5,5))
    topic_counts.plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# =========================
# 😊 SENTIMENT ANALYSIS
# =========================
st.subheader("😊 Sentiment Analysis")

if "label" in df.columns:
    topic_sentiment = pd.crosstab(df["topic"], df["label"])

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        topic_sentiment.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        topic_sentiment_percent = topic_sentiment.div(
            topic_sentiment.sum(axis=1), axis=0
        )

        fig, ax = plt.subplots(figsize=(6,4))
        topic_sentiment_percent.plot(kind="bar", stacked=True, ax=ax)
        st.pyplot(fig)

# =========================
# 📈 TOP INSIGHTS TABLE
# =========================
st.subheader("📊 Top Insights")

top_topics = df["topic"].value_counts().reset_index()
top_topics.columns = ["Topic", "Posts"]

st.dataframe(top_topics)

# =========================
# 🏷️ HASHTAG GENERATOR
# =========================
st.subheader("🏷️ Hashtag Generator")

def get_keywords(text_series):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(text_series)

    words = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    sorted_idx = scores.argsort()[::-1]

    result = []
    for i in sorted_idx:
        w = words[i]
        if len(w) > 3:
            result.append(w)
        if len(result) == 5:
            break
    return result

topic_keywords = {}
for topic in df["topic"].unique():
    topic_keywords[topic] = get_keywords(
        df[df["topic"] == topic]["clean_text"]
    )

# display in columns
cols = st.columns(3)

i = 0
for topic, words in topic_keywords.items():
    with cols[i % 3]:
        st.markdown(f"### {topic}")
        st.write(", ".join(["#" + w.capitalize() for w in words]))
    i += 1

# =========================
# 📥 USER INPUT (ADVANCED FEATURE)
# =========================
st.subheader("🧠 Try Your Own Text")

user_input = st.text_area("Enter a post")

if st.button("Analyze"):
    cleaned = clean_text(user_input)

    topic = assign_topic(cleaned)

    st.write("📌 Predicted Topic:", topic)

    words = get_keywords(pd.Series([cleaned]))
    hashtags = ["#" + w.capitalize() for w in words]

    st.write("🏷️ Suggested Hashtags:", ", ".join(hashtags))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("🚀 Built with Streamlit | Project by Abhishek Mishra")

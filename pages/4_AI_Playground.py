import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Playground", layout="wide")
st.header("🤖 AI Playground - Hugging Face")

@st.cache_resource
def load_sentiment():
    # If torch not installed, use framework="tf"
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_summarizer():
    # Smaller model for less memory
    return pipeline("summarization", model="t5-small")

tab1, tab2 = st.tabs(["😊 Sentiment", "📝 Summarize"])

with tab1:
    text = st.text_area("Enter text:", height=120)
    if st.button("Analyze"):
        if text.strip():
            pipe = load_sentiment()
            res = pipe(text)[0]
            st.metric("Sentiment", res['label'], f"{res['score']:.1%}")
        else:
            st.warning("Enter text")

with tab2:
    long_text = st.text_area("Long article:", height=200)
    if st.button("Summarize"):
        if len(long_text) > 50:
            pipe = load_summarizer()
            out = pipe(long_text, max_length=100, min_length=20)[0]
            st.success(out['summary_text'])
        else:
            st.warning("Enter at least 50 characters")
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Playground", layout="wide")
st.header("🤖 AI Playground - Hugging Face")

# Cache models to avoid reloading
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_summarizer():
    # t5-small is lightweight (~300MB)
    return pipeline("summarization", model="t5-small")

tab1, tab2 = st.tabs(["😊 Sentiment Analysis", "📝 Text Summarization"])

with tab1:
    st.subheader("Sentiment Analysis")
    text = st.text_area("Enter your text:", height=120,
                       placeholder="I love this product! It's amazing.")
    if st.button("Analyze Sentiment"):
        if text.strip():
            with st.spinner("Analyzing..."):
                pipe = load_sentiment()
                result = pipe(text)[0]
                label = result['label']
                score = result['score']
                emoji = "😊" if label == "POSITIVE" else "😞"
                st.metric(f"{emoji} Sentiment", label, f"Confidence: {score:.1%}")
        else:
            st.warning("Please enter some text.")

with tab2:
    st.subheader("Text Summarization")
    long_text = st.text_area("Paste long article:", height=200,
                            placeholder="Enter at least 100 characters...")
    if st.button("Summarize"):
        if len(long_text) > 50:
            with st.spinner("Generating summary..."):
                pipe = load_summarizer()
                summary = pipe(long_text, max_length=100, min_length=20)[0]
                st.success("Summary:")
                st.write(summary['summary_text'])
        else:
            st.warning("Please enter at least 50 characters.")

st.sidebar.markdown("""
**ℹ️ About**
- Sentiment: `distilbert-base-uncased`
- Summarization: `t5-small` (lightweight)
- Models are cached using `@st.cache_resource`
- Using CPU-only PyTorch for memory efficiency
""")
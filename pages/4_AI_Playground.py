import streamlit as st
from textblob import TextBlob

st.set_page_config(page_title="AI Playground", layout="wide")
st.header("🤖 AI Playground - Sentiment Analysis")

st.info("Lightweight sentiment analysis using TextBlob. No deep learning models required, fast and memory efficient.")

st.markdown("---")

# Text input
text = st.text_area("Enter your text here:", height=150, 
                    placeholder="Example: I love this product! It's absolutely amazing.")

if st.button("🔍 Analyze Sentiment", type="primary"):
    if text.strip():
        # Perform sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Range: -1 (negative) to +1 (positive)
        subjectivity = blob.sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment = "Positive 😊"
            color = "green"
        elif polarity < -0.1:
            sentiment = "Negative 😞"
            color = "red"
        else:
            sentiment = "Neutral 😐"
            color = "gray"
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment)
            st.metric("Polarity Score", f"{polarity:.3f}", 
                      help="Range: -1 (very negative) to +1 (very positive)")
        with col2:
            st.metric("Subjectivity", f"{subjectivity:.3f}",
                      help="Range: 0 (factual/objective) to 1 (personal opinion)")
        
        # Visualize polarity as a progress bar
        st.subheader("Polarity Meter")
        polarity_normalized = (polarity + 1) / 2  # convert -1..1 to 0..1
        st.progress(polarity_normalized)
        st.caption("← Negative | Neutral | Positive →")
        
        # Example feedback
        if polarity > 0.5:
            st.success("Very positive feedback! 🎉")
        elif polarity < -0.5:
            st.error("Very negative feedback. 😔")
        elif polarity > 0:
            st.info("Slightly positive tone.")
        elif polarity < 0:
            st.warning("Slightly negative tone.")
        else:
            st.info("Neutral tone.")
    else:
        st.warning("Please enter some text to analyze.")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    - **Model:** TextBlob (rule-based)
    - **No external APIs** – runs locally
    - **Polarity:** -1 (negative) to +1 (positive)
    - **Subjectivity:** 0 (fact) to 1 (opinion)
    """)
    st.caption("Lightweight alternative to Hugging Face models")
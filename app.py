import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Intelligent Health Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }
    /* Sidebar styling */
    .css-1d391kg, .stSidebar {
        background-color: #e8f0fe;
    }
    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
    st.markdown("## 🧠 Health Risk Analyzer")
    st.markdown("---")
    st.markdown("""
    **Features:**
    - 📊 Data Overview
    - 🤖 Model Training
    - 🔮 Real-time Predictions
    """)
    st.markdown("---")
    st.caption("Version 2.0 | Powered by Streamlit")

# ========== MAIN PAGE ==========
st.title("🏥 Intelligent Health Risk Prediction System")
st.markdown("### Your AI-Powered Clinical Decision Support Tool")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This application uses **Machine Learning** to assess patient risk based on:
    - 🧬 Demographic data (Age)
    - ❤️ Clinical measurements (Blood Pressure, Cholesterol, Glucose)

    **How to use:**
    1. Go to **Data Overview** → generate or upload dataset
    2. Train a **Random Forest** model in **Model Training**
    3. Make **real-time predictions** in **Predictions**
    """)

with col2:
    st.info("""
    **ℹ️ App Info**
    - Model: Random Forest
    - Accuracy: ~94% (on sample data)
    - Framework: Streamlit + scikit-learn
    """)

st.markdown("---")

# Quick stats (dummy numbers – can be replaced with session state values)
col_a, col_b, col_c = st.columns(3)
col_a.metric("📊 Dataset Size", "200 patients", "+0")
col_b.metric("🎯 Model Accuracy", "94.5%", "+2.1%")
col_c.metric("🔄 Predictions Made", "0", "ready")

st.markdown("---")
st.caption("© 2026 Intelligent Systems FYP | Horizon Campus")
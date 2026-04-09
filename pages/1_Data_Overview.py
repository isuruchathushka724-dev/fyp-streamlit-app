import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Cache dataset generation - runs only once per session or when cache cleared
@st.cache_data(ttl=3600, show_spinner=False)
def generate_dataset(n_samples=200, random_seed=42):
    """Generate synthetic medical dataset. Cached for 1 hour."""
    np.random.seed(random_seed)
    df = pd.DataFrame({
        'Patient_ID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 85, n_samples),
        'Blood_Pressure': np.random.randint(90, 190, n_samples),
        'Cholesterol': np.random.randint(150, 320, n_samples),
        'Glucose': np.random.randint(70, 200, n_samples),
        'Outcome': np.random.choice(['High Risk', 'Low Risk'], n_samples, p=[0.4, 0.6])
    })
    return df

st.header("📊 Data Overview")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = generate_dataset()

# Button to clear cache and generate new data
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 New Dataset"):
        # Clear the cache for this function
        generate_dataset.clear()
        st.session_state.dataset = generate_dataset()
        st.rerun()

df = st.session_state.dataset

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", df.shape[0])
col2.metric("Features", df.shape[1] - 1)
col3.metric("High Risk %", f"{(df['Outcome'] == 'High Risk').mean():.1%}")

st.subheader("Sample Data (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Summary Statistics")
st.dataframe(df.describe(), use_container_width=True)

# Charts
tab1, tab2 = st.tabs(["📈 Age Distribution", "📊 Risk by Feature"])
with tab1:
    fig = px.histogram(df, x='Age', color='Outcome', nbins=30, title='Age Distribution by Risk')
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig2 = px.box(df, x='Outcome', y='Cholesterol', title='Cholesterol Levels by Risk')
    st.plotly_chart(fig2, use_container_width=True)

st.caption("💡 Dataset is cached. Click 'New Dataset' to generate fresh data.")
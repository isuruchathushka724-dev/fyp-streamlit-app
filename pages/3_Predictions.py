import streamlit as st
import numpy as np

st.header("🔮 Live Risk Predictions")

# Check if model exists in session state (from training page)
if 'model' not in st.session_state or st.session_state.model is None:
    st.warning("⚠️ No trained model found. Please go to **Model Training** page and train a model first.")
    st.stop()

model = st.session_state.model
feature_cols = st.session_state.get('feature_cols', ['Age', 'Blood_Pressure', 'Cholesterol', 'Glucose'])

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45, step=1)
    bp = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120, step=1)
with col2:
    chol = st.number_input("Cholesterol (mg/dL)", 150, 400, 200, step=5)
    glucose = st.number_input("Glucose (mg/dL)", 70, 250, 110, step=5)

if st.button("🧠 Predict Risk", type="primary"):
    input_features = np.array([[age, bp, chol, glucose]])
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ **High Risk Patient**\n\nProbability: {proba:.1%}")
    else:
        st.success(f"✅ **Low Risk Patient**\n\nProbability: {1-proba:.1%}")
    
    st.metric("Risk Score", f"{proba:.1%}")

with st.sidebar:
    st.subheader("ℹ️ Model Info")
    if 'model_accuracy' in st.session_state:
        st.write(f"Accuracy: {st.session_state.model_accuracy:.2%}")
    st.write("Model loaded from session state (shared across pages).")
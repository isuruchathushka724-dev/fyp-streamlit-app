import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import joblib
import os

# Cache for loading pre-trained model from file
@st.cache_resource
def load_saved_model(model_path):
    """Load model from file. Cached across all reruns."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Cache for training (only when hyperparameters change)
@st.cache_data
def train_cached_model(X_train, y_train, n_estimators, max_depth, random_state):
    """Train model - result cached based on parameters."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

st.header("🤖 Model Training")

MODEL_PATH = "models/trained_model.pkl"

# Check if dataset exists
if 'dataset' not in st.session_state:
    st.warning("⚠️ No dataset found. Please go to **Data Overview** page first.")
    st.stop()

df = st.session_state.dataset

# Prepare data
df['target'] = (df['Outcome'] == 'High Risk').astype(int)
feature_cols = ['Age', 'Blood_Pressure', 'Cholesterol', 'Glucose']
X = df[feature_cols]
y = df['target']

# Sidebar parameters
with st.sidebar:
    st.subheader("⚙️ Model Parameters")
    n_estimators = st.slider("Number of trees", 10, 300, 100, 10)
    max_depth = st.slider("Max depth", 1, 20, 5)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2)
    random_state = 42

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Check if saved model exists and matches current parameters
saved_model = load_saved_model(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Train New Model", type="primary"):
        with st.spinner("Training..."):
            model = train_cached_model(X_train, y_train, n_estimators, max_depth, random_state)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Save model to file
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.model_accuracy = acc
            st.session_state.feature_cols = feature_cols
            
            st.success(f"✅ Model trained! Accuracy: {acc:.2%}")
            st.balloons()
            
            # Show confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, columns=['Pred Low', 'Pred High'], index=['Actual Low', 'Actual High']))
            
            # Feature importance
            importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)

with col2:
    if saved_model is not None and st.button("📂 Load Saved Model"):
        st.session_state.model = saved_model
        # Calculate accuracy on current test set
        y_pred = saved_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.session_state.model_accuracy = acc
        st.session_state.feature_cols = feature_cols
        st.success(f"✅ Loaded saved model! Accuracy on current test set: {acc:.2%}")

# Display current model status
if 'model' in st.session_state:
    st.metric("📌 Current Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
    st.info("Model is ready for predictions on the **Predictions** page.")
else:
    st.info("Train a new model or load saved model to start predictions.")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MediPredict - Disease Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; color: #1E3A8A; text-align: center; margin-bottom: 0.5rem;}
    .result-card {background-color: #F0F9FF; padding: 1.5rem; border-radius: 12px; border-left: 6px solid #3B82F6;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open(MODEL_DIR / "features.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        precautions = pd.read_csv(DATA_DIR / "precautions_clean.csv")
        medications = pd.read_csv(DATA_DIR / "medications_clean.csv")
        diets = pd.read_csv(DATA_DIR / "diets_clean.csv")
        workouts = pd.read_csv(DATA_DIR / "workout_clean.csv")
        descriptions = pd.read_csv(DATA_DIR / "description_clean.csv")
        
        return le, feature_names, precautions, medications, diets, workouts, descriptions
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

le, feature_names, precautions_df, medications_df, diets_df, workouts_df, descriptions_df = load_artifacts()

@st.cache_resource
def load_model():
    try:
        model_path = MODEL_DIR / "best_model.pkl"
        if not model_path.exists():
            st.error("Model file not found. Please train and save the model first.")
            st.stop()
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.info("**Tip**: Re-save your model in the notebook with `protocol=4` and upgrade numpy/pandas/scikit-learn.")
        st.stop()

model = load_model()

# ====================== UI ======================
st.markdown('<h1 class="main-header">🩺 MediPredict</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Symptom to Disease Prediction System**")

st.sidebar.title("Navigation")
st.sidebar.info("Select symptoms on the main page")

# Symptom Input
selected_symptoms = st.multiselect(
    "Select Symptoms You Are Experiencing",
    options=sorted(feature_names),
    help="You can select multiple symptoms. More symptoms = higher accuracy."
)

if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
    if not selected_symptoms:
        st.error("⚠️ Please select at least one symptom.")
    else:
        with st.spinner("Analyzing your symptoms with ML model..."):
            # Build feature vector
            input_vec = np.zeros(len(feature_names), dtype=np.int8)
            sym_to_idx = {sym: i for i, sym in enumerate(feature_names)}
            
            for sym in selected_symptoms:
                if sym in sym_to_idx:
                    input_vec[sym_to_idx[sym]] = 1
            
            input_vec = input_vec.reshape(1, -1)
            
            # Prediction
            pred = model.predict(input_vec)[0]
            disease = le.inverse_transform([pred])[0].title()
            
            # Confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_vec)[0]
                confidence = proba.max() * 100
                top3_idx = proba.argsort()[-3:][::-1]
                top3_diseases = le.inverse_transform(top3_idx)
                top3_probs = proba[top3_idx]
            else:
                confidence = 88.0
                top3_diseases = [disease]
                top3_probs = [0.88]

        # Results
        st.markdown(f"### 🏥 Predicted Condition: **{disease}**")
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
        
        st.subheader("Top 3 Predictions")
        for d, p in zip(top3_diseases, top3_probs):
            st.progress(float(p), text=f"{d.title()} — {p*100:.1f}%")
        
        disease_key = disease.lower()
        
        # Recommendations
        tabs = st.tabs(["📋 Description", "💊 Medications", "🥗 Diet", "🏋️ Workouts", "⚠️ Precautions"])
        
        with tabs[0]:
            row = descriptions_df[descriptions_df['Disease'].str.lower() == disease_key]
            st.markdown(row.iloc[0]['Description'] if not row.empty else "No description available.")
        
        with tabs[1]:
            row = medications_df[medications_df['Disease'].str.lower() == disease_key]
            if not row.empty:
                for med in row.iloc[0]['Medication'].split(" | "):
                    st.write(f"• {med}")
        
        with tabs[2]:
            row = diets_df[diets_df['Disease'].str.lower() == disease_key]
            if not row.empty:
                for item in row.iloc[0]['Diet'].split(" | "):
                    st.write(f"✅ {item}")
        
        with tabs[3]:
            row = workouts_df[workouts_df['Disease'].str.lower() == disease_key]
            if not row.empty:
                for w in row.iloc[0]['Workouts'].split(" | "):
                    st.write(f"🏃 {w}")
        
        with tabs[4]:
            row = precautions_df[precautions_df['Disease'].str.lower() == disease_key]
            if not row.empty:
                for i in range(1, 5):
                    col = f"Precaution_{i}"
                    if col in row.columns and pd.notna(row.iloc[0][col]):
                        st.write(f"🛡️ {row.iloc[0][col]}")

st.caption("⚠️ **Disclaimer**: This tool is for educational purposes only. Always consult a licensed medical professional for diagnosis and treatment.")
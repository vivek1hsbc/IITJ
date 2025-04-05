import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Student Score Predictor", layout="centered")

# Title
st.title("🎓 Student Score Predictor")
st.markdown("Enter student details to predict their expected score.")

# --- User Inputs ---
hours_studied = st.number_input("📘 Hours Studied", min_value=0.0, max_value=24.0, step=0.5, value=6.0)
previous_scores = st.number_input("📊 Previous Scores (%)", min_value=0.0, max_value=100.0, step=1.0, value=70.0)
sleep_hours = st.number_input("💤 Sleep Hours", min_value=0.0, max_value=24.0, step=0.5, value=7.0)
sample_papers = st.number_input("📄 Sample Question Papers Practiced", min_value=0, max_value=50, step=1, value=5)
extra_activities = st.radio("🎨 Extracurricular Activities", ["No", "Yes"])

# Convert input to DataFrame
data = {
    "Unnamed: 0": 0,
    "Hours Studied": hours_studied,
    "Previous Scores": previous_scores,
    "Sleep Hours": sleep_hours,
    "Sample Question Papers Practiced": sample_papers,
    "Extracurricular Activities_Yes": 1 if extra_activities == "Yes" else 0
}
input_df = pd.DataFrame([data])

# Predict Button
if st.button("🎯 Predict Score"):
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)

        if isinstance(model, tuple) and model[0] == "dl":
            _, scaler, keras_model = model
            input_scaled = scaler.transform(input_df)
            prediction = keras_model.predict(input_scaled)
        else:
            prediction = model.predict(input_df)

        st.success(f"📈 Predicted Score: {float(prediction[0]):.2f}%")

    except FileNotFoundError:
        st.error("❌ Model file not found. Please make sure 'best_model.pkl' is in the same directory.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

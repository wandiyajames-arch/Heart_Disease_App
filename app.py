import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING (cached so it only loads once per session)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "rf_pipeline.pkl"

FEATURE_ORDER = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
    "fasting_blood_sugar", "ecg", "max_heart_rate", "exercise_induced_chest_pain",
    "st_depression", "st_slope", "stained_blood_vessels", "blood_disorder",
]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this tool")
    st.write(
        "This app uses a **Random Forest** classifier trained on clinical "
        "features to estimate the likelihood of heart disease presence."
    )
    st.markdown("**Pipeline:** `StandardScaler` → `RandomForestClassifier`")

    st.markdown("---")
    st.subheader("⚠️ Medical Disclaimer")
    st.caption(
        "This tool is for educational purposes only and is **not** a "
        "substitute for professional medical advice, diagnosis, or "
        "treatment. Always consult a qualified physician regarding any "
        "health concerns."
    )

    st.markdown("---")
    if model is not None:
        st.success("Model loaded successfully.")
    else:
        st.error("Model file not found — see instructions below.")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.write(
    "Enter the patient's clinical details below. The model will estimate "
    "the probability of heart disease based on patterns learned from "
    "historical patient data."
)
st.divider()

if model is None:
    st.error(
        "**`rf_pipeline.pkl` was not found next to `app.py`.** "
        "Make sure the model file is uploaded to the same folder/repo "
        "before running or deploying this app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# INPUT FORM
# (wrapped in st.form so the app only reruns once, on submit —
#  not on every single widget interaction)
# ─────────────────────────────────────────────────────────────
with st.form("patient_form"):
    st.subheader("Patient Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox(
            "Sex", options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
        )
        chest_pain_type = st.selectbox(
            "Chest Pain Type", options=[0, 1, 2, 3],
            help="0: Typical angina · 1: Atypical angina · 2: Non-anginal pain · 3: Asymptomatic",
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200
        )

    with col2:
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        ecg = st.selectbox(
            "Resting ECG Results", options=[0, 1, 2],
            help="0: Normal · 1: ST-T wave abnormality · 2: Left ventricular hypertrophy",
        )
        max_heart_rate = st.number_input(
            "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150
        )
        exercise_induced_chest_pain = st.selectbox(
            "Exercise Induced Angina", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col3:
        st_depression = st.number_input(
            "ST Depression (exercise vs. rest)", min_value=0.0, max_value=10.0,
            value=1.0, step=0.1,
        )
        st_slope = st.selectbox(
            "ST Slope", options=[0, 1, 2],
            help="0: Upsloping · 1: Flat · 2: Downsloping",
        )
        stained_blood_vessels = st.selectbox(
            "Number of Major Vessels Stained (0-4)", options=[0, 1, 2, 3, 4]
        )
        blood_disorder = st.selectbox(
            "Thalassemia / Blood Disorder Code", options=[0, 1, 2, 3]
        )

    st.markdown("")
    submitted = st.form_submit_button("🔍 Predict Diagnosis", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame(
        [[
            age, sex, chest_pain_type, resting_bp, cholesterol,
            fasting_blood_sugar, ecg, max_heart_rate, exercise_induced_chest_pain,
            st_depression, st_slope, stained_blood_vessels, blood_disorder,
        ]],
        columns=FEATURE_ORDER,
    )

    try:
        prediction = model.predict(input_data)[0]
        proba = (
            model.predict_proba(input_data)[0]
            if hasattr(model, "predict_proba")
            else None
        )
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")
        st.stop()

    st.divider()
    st.subheader("Result")

    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error("🚨 **High Risk:** The model predicts the presence of heart disease.")
        else:
            st.success("✅ **Low Risk:** The model predicts no heart disease.")

        if proba is not None:
            risk_probability = proba[1]
            st.metric("Estimated Probability of Heart Disease", f"{risk_probability:.1%}")

    with gauge_col:
        if proba is not None:
            st.write("**Confidence breakdown**")
            st.progress(float(proba[1]), text=f"High risk: {proba[1]:.1%}")
            st.progress(float(proba[0]), text=f"Low risk: {proba[0]:.1%}")

    with st.expander("See the exact input sent to the model"):
        st.dataframe(input_data, use_container_width=True)

    st.caption(
        "Remember: this is a statistical estimate from a machine learning "
        "model, not a medical diagnosis. Please consult a cardiologist for "
        "an accurate clinical evaluation."
    )
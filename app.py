import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- Load The Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Page Configuration ---
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem !important;
            font-weight: bold;
            color: #ff6f00;
            text-align: center;
            animation: fadeInDown 1s ease-in-out;
        }
        @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .prediction-result {
            font-size: 1.5rem;
            color: #1565c0;
            font-weight: bold;
            text-align: center;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Title and Description ---
st.markdown("<div class='main-title'>Boston House Price Predictor 🏠</div>", unsafe_allow_html=True)
st.write(
    """
    This app predicts the **median value of owner-occupied homes** in Boston suburbs.
    Enter the details of a neighborhood, and the model will estimate the house price.
    This is based on a Linear Regression model trained on the classic Boston Housing dataset.
    """
)
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Input Features")

def user_input_features():
    CRIM = st.sidebar.slider('Criminality Rate (CRIM)', 0.0, 90.0, 0.1)
    ZN = st.sidebar.slider('Proportion of Residential Land Zoned (ZN)', 0.0, 100.0, 12.5)
    INDUS = st.sidebar.slider('Proportion of Non-Retail Business Acres (INDUS)', 0.0, 30.0, 11.1)
    CHAS = st.sidebar.selectbox('Borders Charles River? (CHAS)', (0, 1))
    NOX = st.sidebar.slider('Nitric Oxides Concentration (NOX)', 0.3, 0.9, 0.5)
    RM = st.sidebar.slider('Average Number of Rooms (RM)', 3.0, 9.0, 6.2)
    AGE = st.sidebar.slider('Proportion of Old Units (AGE)', 0.0, 100.0, 68.0)
    DIS = st.sidebar.slider('Weighted Distances to Employment Centres (DIS)', 1.0, 13.0, 3.8)
    RAD = st.sidebar.slider('Index of Accessibility to Radial Highways (RAD)', 1.0, 24.0, 9.5)
    TAX = st.sidebar.slider('Property-Tax Rate (TAX)', 180.0, 720.0, 408.0)
    PTRATIO = st.sidebar.slider('Pupil-Teacher Ratio (PTRATIO)', 12.0, 22.0, 18.4)
    B = st.sidebar.slider('Proportion of Black Residents (B)', 0.0, 400.0, 356.0)
    LSTAT = st.sidebar.slider('% Lower Status of the Population (LSTAT)', 1.0, 40.0, 12.6)

    data = {
        'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS,
        'NOX': NOX, 'RM': RM, 'AGE': AGE, 'DIS': DIS,
        'RAD': RAD, 'TAX': TAX, 'PTRATIO': PTRATIO, 'B': B, 'LSTAT': LSTAT
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main Panel ---
st.header("Your Input")
st.dataframe(input_df)

if st.sidebar.button("Predict House Price"):
    with st.spinner("Calculating price prediction..."):
        time.sleep(1.5)  # Simulate processing delay
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        predicted_price = prediction[0] * 1000
    
    st.markdown(f"<div class='prediction-result'>Predicted House Price: ${predicted_price:,.2f}</div>", unsafe_allow_html=True)
    st.balloons()

st.markdown("---")
st.write("Disclaimer: The Boston Housing dataset has known ethical issues. This app is for educational purposes only.")

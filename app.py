import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------
# 1. PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Power Zone 3 Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# 2. CUSTOM CSS STYLING
# -------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    h1 {
        color: #2c3e50;
    }
    h3 {
        color: #34495e;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 5px;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# 3. LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    try:
        with open('electricity_consumption_lr_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please make sure 'electricity_consumption_lr_model.pkl' is in the same directory.")
        return None

model = load_model()

# -------------------------
# 4. SIDEBAR
# -------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933864.png", width=100)
    st.title("Settings & Info")
    st.info(
        """
        This application predicts the **Power Consumption of Zone 3** 
        based on environmental factors and consumption in other zones 
        using a Linear Regression model.
        """
    )
    st.markdown("---")
    st.write("**Model:** Linear Regression")
    st.write("**Data Source:** Electric Power Consumption Dataset")
    st.markdown("---")
    st.write("Developed with ❤️ using Streamlit")

# -------------------------
# 5. MAIN APP UI
# -------------------------
st.title("⚡ Electricity Consumption Forecaster")
st.markdown("Enter the environmental parameters and current zone loads below to predict Zone 3 usage.")

# Container for inputs
with st.container():
    st.markdown("### 1. Environmental Factors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=6.5, step=0.1, help="Current ambient temperature")
    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=73.8, step=0.1)
    with col3:
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=0.08, step=0.01)

    col4, col5 = st.columns(2)
    with col4:
        gen_diff_flows = st.number_input("General Diffuse Flows", min_value=0.0, value=0.05, step=0.01)
    with col5:
        diff_flows = st.number_input("Diffuse Flows", min_value=0.0, value=0.12, step=0.01)

    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    st.markdown("### 2. Other Zones Consumption")
    
    col6, col7 = st.columns(2)
    with col6:
        zone1 = st.number_input("Power Consumption Zone 1 (KW)", min_value=0.0, value=34055.0, step=10.0)
    with col7:
        zone2 = st.number_input("Power Consumption Zone 2 (KW)", min_value=0.0, value=16128.0, step=10.0)

# -------------------------
# 6. PREDICTION LOGIC
# -------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Consumption"):
    if model:
        # Prepare input array matching the training columns order
        # [Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows, Zone1, Zone2]
        features = np.array([[temp, humidity, wind_speed, gen_diff_flows, diff_flows, zone1, zone2]])
        
        try:
            prediction = model.predict(features)[0]
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Display Result nicely
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="margin-bottom:0px; color:#555;">Predicted Consumption (Zone 3)</h2>
                    <h1 style="color:#4CAF50; font-size: 48px; margin-top:10px;">{prediction:,.2f} KW</h1>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please ensure input feature order matches the trained model.")

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# --- 1. Load All Assets ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('newgen.h5')
    
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
        
    with open('LabelEncoder.pkl', 'rb') as label_file:
        location_encoder = pickle.load(label_file)
        
    return model, preprocessor, location_encoder

model, preprocessor, location_encoder = load_assets()

# --- App Header ---
st.title("🌧️ Australian Rain Prediction App")
st.markdown("Enter today's weather details below to predict if it will rain tomorrow.")
st.divider()

# --- 2. User Input UI ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📍 Location & Basic")
    # MAGIC TRICK: This automatically loads all 49 cities from your encoder!
    location = st.selectbox("Location", location_encoder.classes_) 
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=5.0)
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=8.0)

with col2:
    st.header("🌡️ Temperature & Clouds")
    min_temp = st.number_input("Min Temperature (°C)", value=12.0)
    max_temp = st.number_input("Max Temperature (°C)", value=25.0)
    temp_9am = st.number_input("Temp at 9am (°C)", value=15.0)
    temp_3pm = st.number_input("Temp at 3pm (°C)", value=23.0)
    cloud_9am = st.slider("Cloud Cover 9am (oktas)", min_value=0.0, max_value=9.0, value=4.0)
    cloud_3pm = st.slider("Cloud Cover 3pm (oktas)", min_value=0.0, max_value=9.0, value=4.0)

with col3:
    st.header("💨 Wind Data")
    compass = ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE', 'E', 'ENE', 'NW']
    wind_gust_dir = st.selectbox("Wind Gust Direction", compass)
    wind_dir_9am = st.selectbox("Wind Direction 9am", compass)
    wind_dir_3pm = st.selectbox("Wind Direction 3pm", compass)
    wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", min_value=0.0, value=40.0)
    wind_speed_9am = st.number_input("Wind Speed 9am (km/h)", min_value=0.0, value=15.0)
    wind_speed_3pm = st.number_input("Wind Speed 3pm (km/h)", min_value=0.0, value=20.0)

st.divider()

st.header("💧 Humidity & Pressure")
col4, col5 = st.columns(2)

with col4:
    humidity_9am = st.slider("Humidity 9am (%)", min_value=0.0, max_value=100.0, value=60.0)
    humidity_3pm = st.slider("Humidity 3pm (%)", min_value=0.0, max_value=100.0, value=50.0)
with col5:
    pressure_9am = st.number_input("Pressure 9am (hPa)", min_value=900.0, max_value=1100.0, value=1015.0)
    pressure_3pm = st.number_input("Pressure 3pm (hPa)", min_value=900.0, max_value=1100.0, value=1012.0)

st.divider()

# --- 3. Prediction Logic ---
if st.button("🔮 Predict Weather For Tomorrow", type="primary", use_container_width=True):
    
    # Encode the location into a number FIRST
    encoded_loc = location_encoder.transform([location])[0]
    
    # Pack the dictionary (Notice 'Location' now uses the encoded number)
    input_dict = {
        'Location': [encoded_loc],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed],
        'WindDir9am': [wind_dir_9am],
        'WindDir3pm': [wind_dir_3pm],
        'WindSpeed9am': [wind_speed_9am],
        'WindSpeed3pm': [wind_speed_3pm],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Pressure9am': [pressure_9am],
        'Pressure3pm': [pressure_3pm],
        'Cloud9am': [cloud_9am],
        'Cloud3pm': [cloud_3pm],
        'Temp9am': [temp_9am],
        'Temp3pm': [temp_3pm]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    try:
        processed_data = preprocessor.transform(input_df)
        final_input = np.asarray(processed_data).astype('float32')
        
        prediction = model.predict(final_input)
        probability = float(prediction[0][0]) 
        
        if probability > 0.5:
            st.error(f"### ☔ Expect Rain Tomorrow!\n**Confidence / Probability:** {probability:.1%}")
            st.progress(probability)
        else:
            st.success(f"### ☀️ Clear Skies Expected!\n**Probability of Rain:** {probability:.1%}")
            st.progress(probability)
            
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
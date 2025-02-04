import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title
st.title("Laptop Price Predictor")

# User Inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size (inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Convert categorical values to numeric
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Compute Pixels Per Inch (PPI)
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size

    # Ensure column names exactly match training data
    feature_names = ['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 
                     'PPI', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']

    # Make sure the feature names match the training data
    query_df = pd.DataFrame([[company, type, ram, weight, touchscreen, ips, 
                              ppi, cpu, hdd, ssd, gpu, os]], 
                            columns=feature_names)

    # Rename columns to exactly match the trained model
    query_df.rename(columns={'IPS': 'Ips', 'PPI': 'ppi'}, inplace=True)

    # Predict price
    predicted_price = np.exp(pipe.predict(query_df)[0])
    st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")

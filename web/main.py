import streamlit as st

# Set up Streamlit page
st.set_page_config(page_title="SmartSensor", layout="centered")

# Load and apply custom CSS
import os
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Show banner title
st.markdown("""
<div class='banner'>
  <span class='banner-title'>SmartSensor BioAI</span>
</div>
""", unsafe_allow_html=True)

# Welcome message
st.markdown("""
## Welcome to SmartSensor BioAI

This application helps you analyze and characterize chemical compounds using smartphone camera images.

### Features:
1. **Image Processing**
   - Upload and process images
   - Apply various image processing techniques
   - Save processed images

2. **Model Prediction**
   - Run predictions on processed images
   - View detailed model results
   - Download prediction data

### Getting Started:
1. Navigate to the Image Processing page to upload and process your images
2. Go to the Model Prediction page to run predictions on your processed images
""")

# Show footer
st.markdown("<div class='footer'>Created by @Team SmartSensor.</div>", unsafe_allow_html=True)

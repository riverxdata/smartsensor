import streamlit as st

st.set_page_config(page_title="SmartSensor", layout="centered")

# Đọc và nhúng CSS từ file style.css
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Banner duy nhất
st.markdown("""
<div class='banner'>
  <span class='banner-title'>SmartSensor BioAI</span>
</div>
""", unsafe_allow_html=True)

# Input fields
sepal_length = st.text_input("Sepal Length", value="1")
sepal_width = st.text_input("Sepal Width", value="2")
petal_length = st.text_input("Petal Length", value="3")
petal_width = st.text_input("Petal Width", value="4")

# Predict button with custom style
predict_btn = st.button("Predict")

result = None
if predict_btn:
    # Dummy model: just output a static value or a list
    result = [2]

if result is not None:
    st.markdown(f"""
    <div style='background-color:#d4f7d4; padding: 20px; border-radius: 5px; margin-top: 20px;'>
        <span style='color:#217346; font-size: 22px; font-weight: 500;'>
            The output is {result}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for Predict button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: white;
        color: #e75480;
        border: 3px solid #e75480;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        padding: 8px 32px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ffe6ef;
        color: #c2185b;
        border: 3px solid #c2185b;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Created by @Team SmartSensor.</div>", unsafe_allow_html=True)

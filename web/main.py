import streamlit as st

# Set up Streamlit page
st.set_page_config(page_title="SmartSensor", layout="centered")

# Load and apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Show banner title
st.markdown("""
<div class='banner'>
  <span class='banner-title'>SmartSensor BioAI</span>
</div>
""", unsafe_allow_html=True)

# --- Image upload and selection ---
import os
from PIL import Image

img_dir = "img"
os.makedirs(img_dir, exist_ok=True)

# Upload new image
uploaded_file = st.file_uploader("Upload an image (jpg, png, jpeg, bmp)", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.thumbnail((300, 300))  # Resize for display
    save_path = os.path.join(img_dir, uploaded_file.name)
    image.save(save_path)
    st.success(f"Image saved as {uploaded_file.name} (resized)")

# List and select images
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

if img_files:
    selected_img = st.selectbox("Select an image to predict concentration", img_files)
    if selected_img:
        img_path = os.path.join(img_dir, selected_img)
        st.image(img_path, caption=f"Selected image: {selected_img}", width=300)

# --- Prediction button and output ---
predict_btn = st.button("Predict")

result = None
if predict_btn and selected_img:
    # Call prediction model here
    # result = predict_concentration(img_path)
    result = [2]  # Dummy output

# Show prediction result
if result is not None:
    st.markdown(f"""
    <div style='background-color:#d4f7d4; padding: 20px; border-radius: 5px; margin-top: 20px;'>
        <span style='color:#217346; font-size: 22px; font-weight: 500;'>
            The output is {result}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Show footer
st.markdown("<div class='footer'>Created by @Team SmartSensor.</div>", unsafe_allow_html=True)

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

# Display all model results when user clicks Predict
RESULT_DIR = "data/ampicilline/ip_1_10_delta"
RESULT_FILES = [
    "full_model_infor.txt",
    "full_RGB_model.sav",
    "metrics.csv",
    "sensor.log",
    "testsize_0.2_model_infor.txt",
    "testsize_0.2_RGB_model.sav",
    "test_testsize_0.2_.csv",
    "train_testsize_0.2.csv"
]

if predict_btn:
    st.markdown("## Prediction/Model Results")
    for fname in RESULT_FILES:
        fpath = os.path.join(RESULT_DIR, fname)
        st.markdown(f"### {fname}")
        if os.path.exists(fpath):
            if fname.endswith((".txt", ".csv", ".log")):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                st.code(content, language="text")
            else:
                with open(fpath, "rb") as f:
                    btn_label = f"Download {fname}"
                    st.download_button(btn_label, f, file_name=fname)
        else:
            st.warning(f"File not found: {fname}")

# Show footer
st.markdown("<div class='footer'>Created by @Team SmartSensor.</div>", unsafe_allow_html=True)

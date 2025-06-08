import streamlit as st
import os
from PIL import Image

# Set page title
st.set_page_config(page_title="Model Prediction - SmartSensor", layout="centered")

# Show banner title
st.markdown("""
<div class='banner'>
  <span class='banner-title'>Model Prediction</span>
</div>
""", unsafe_allow_html=True)

# --- Model prediction section ---
st.markdown("## Run Model Prediction")

# List and select images
img_dir = "img"
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

if img_files:
    selected_img = st.selectbox("Select an image to predict concentration", img_files)
    if selected_img:
        img_path = os.path.join(img_dir, selected_img)
        st.image(img_path, caption=f"Selected image: {selected_img}", width=300)

        # Model prediction options
        st.markdown("### Prediction Options")
        
        # Model parameters
        st.markdown("#### Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["RGB Model", "Full Model"]
            )
            
        with col2:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.1
            )

        # Prediction button
        if st.button("Run Prediction", type="primary"):
            st.info("Running prediction...")
            
            # Display results
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
            
            st.markdown("### Prediction Results")
            for fname in RESULT_FILES:
                fpath = os.path.join(RESULT_DIR, fname)
                st.markdown(f"#### {fname}")
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
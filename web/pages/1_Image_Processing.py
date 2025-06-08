import streamlit as st
import os
from PIL import Image
from smartsensor.process_image import process_image

# Set page title
st.set_page_config(page_title="Image Processing - SmartSensor", layout="centered")

# Show banner title
st.markdown(
    """
<div class='banner'>
  <span class='banner-title'>Image Processing</span>
</div>
""",
    unsafe_allow_html=True,
)

# --- Image upload and processing ---
st.markdown("## Upload and Process Images")

# Create image directory if not exists
img_dir = "img"
os.makedirs(img_dir, exist_ok=True)

# Upload new image
uploaded_file = st.file_uploader(
    "Upload an image (jpg, png, jpeg, bmp)", type=["jpg", "jpeg", "png", "bmp"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.thumbnail((300, 300))  # Resize for display
    save_path = os.path.join(img_dir, uploaded_file.name)
    image.save(save_path)
    st.success(f"Image saved as {uploaded_file.name} (resized)")

# List and select images
img_files = [
    f
    for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

if img_files:
    selected_img = st.selectbox("Select an image to process", img_files)
    if selected_img:
        img_path = os.path.join(img_dir, selected_img)
        st.image(img_path, caption=f"Selected image: {selected_img}", width=300)

        # Image processing options
        st.markdown("### Processing Options")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Process Image"):
                st.info("Image processing in progress...")
                # Add your image processing logic here
                # TODO: add real logic to here
                # process_image(data=data, outdir=outdir, kit=kit)
                st.success("Image processing completed!")

        with col2:
            if st.button("Save Processed Image"):
                st.info("Saving processed image...")
                # Add your save logic here
                st.success("Processed image saved!")

# Show footer
st.markdown(
    "<div class='footer'>Created by @Team SmartSensor.</div>", unsafe_allow_html=True
)

import streamlit as st
import os
import time
import random
from PIL import Image
from io import BytesIO
from smartsensor.process_image import process_image
from smartsensor.e2e import end2end_pipeline
import shutil
import pandas as pd

# Constants
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
RESULT_DIR = "result"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

IMAGE_EXT = (".jpg", ".jpeg", ".png")
FAKE_CLASSES = ["class A", "class B", "class C"]

st.set_page_config(page_title="SmartSensor Prototype", layout="wide")
st.title("🧠 SmartSensor ML UI Prototype")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📤 Upload Images", "🧪 Process Images", "🎓 Train Model", "🔮 Predict"]
)


def load_and_resize(path: str, max_size=(256, 256)) -> Image.Image:
    img = Image.open(path)
    img.thumbnail(max_size)
    return img


# --- Tab 1: Upload ---
with tab1:
    st.header("📤 Upload Raw Images")

    uploaded = st.file_uploader(
        "Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded:
        for img_file in uploaded:
            save_path = os.path.join(UPLOAD_DIR, img_file.name)
            with open(save_path, "wb") as f:
                f.write(img_file.getbuffer())
        st.success(f"Uploaded {len(uploaded)} images.")

    # Check for clear request in session state
    if st.session_state.get("clear_uploads"):
        for file in os.listdir(UPLOAD_DIR):
            if file.lower().endswith(IMAGE_EXT):
                os.remove(os.path.join(UPLOAD_DIR, file))
        st.session_state["clear_uploads"] = False
        st.rerun()

    # List uploaded images
    images = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(IMAGE_EXT)]

    if images:
        st.subheader("📂 Uploaded Images")

        # Button to trigger deletion
        clear_col, _ = st.columns([1, 9])
        with clear_col:
            if st.button("🗑️ Clear All Uploaded Files"):
                st.session_state["clear_uploads"] = True
                st.rerun()

        # Display images
        cols = st.columns(10)
        for i, name in enumerate(images):
            img = load_and_resize(os.path.join(UPLOAD_DIR, name))
            with cols[i % 10]:
                st.image(img, caption=name, use_container_width=True)

squared_frame_dir = os.path.join(PROCESSED_DIR, "squared_frame")

# --- Tab 2: Process ---
with tab2:
    st.header("🧪 Segment / Process Uploaded Images")
    kit_version = st.selectbox("🧰 Select Kit Version", ["1.0.0", "1.1.0"], index=1)
    if st.button("🚀 Process Images"):
        st.info(f"Processing with kit version {kit_version}...")
        try:
            process_image(data=UPLOAD_DIR, outdir=PROCESSED_DIR, kit=kit_version)
            st.success("✅ Image processing completed!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to process images: {e}")

    # Only list images if directory exists
    if os.path.isdir(squared_frame_dir):
        processed_imgs = [
            f for f in os.listdir(squared_frame_dir) if f.lower().endswith(IMAGE_EXT)
        ]
    else:
        processed_imgs = []

    if processed_imgs:
        st.subheader("🖼️ Processed Images")

        # Clear button
        clear_col, _ = st.columns([1, 9])
        with clear_col:
            if st.button("🗑️ Clear Processed Images"):
                # Remove only files in squared_frame_dir
                for f in processed_imgs:
                    os.remove(os.path.join(squared_frame_dir, f))
                st.success("✅ All processed images deleted.")
                st.rerun()

        # Display
        cols = st.columns(10)
        for i, name in enumerate(processed_imgs):
            img = load_and_resize(os.path.join(squared_frame_dir, name))
            with cols[i % 10]:
                st.image(img, caption=name, use_container_width=True)

    else:
        st.info("No processed images found.")

# --- Tab 3: Train ---
with tab3:
    st.header("🎓 Train Model (Prototype)")

    normalize_method = st.selectbox(
        "🧪 Normalization Method",
        ["raw", "ratio", "delta"],
        index=1,
        key="train_norm_method",
    )

    kit_version = st.selectbox(
        "🧰 Select Kit Version", ["1.0.0", "1.1.0"], index=1, key="train_kit_version"
    )

    degree = st.selectbox(
        "📐 Polynomial Degree", ["1", "2"], index=1, key="train_degree"
    )

    cv = st.slider("🔁 Cross Validation Folds", 2, 10, 5, key="train_cv")

    test_size = st.slider(
        "🧪 Test Size (fraction)", 0.1, 0.5, 0.2, 0.05, key="train_test_size"
    )

    skip_fs = st.checkbox("🚫 Skip Feature Selection", value=True, key="train_skip_fs")

    if st.button("🎯 Start Training"):
        with st.spinner("Training..."):
            data_path = os.path.join(
                PROCESSED_DIR,
                f"features_rgb_{normalize_method}_normalized_roi.csv",
            )
            outdir = os.path.join(RESULT_DIR, normalize_method)
            os.makedirs(outdir, exist_ok=True)

            end2end_pipeline(
                data=data_path,
                features="meanR,meanG,meanB,modeR,modeG,modeB",
                degree=int(degree),
                skip_feature_selection=skip_fs,
                cv=cv,
                outdir=outdir,
                prefix=normalize_method,
                test_size=test_size,
            )

        st.success("✅ Training completed!")

        # Show metrics
        metrics_file = os.path.join(outdir, "metrics.csv")
        if os.path.isfile(metrics_file):
            st.subheader("📊 Evaluation Metrics")

            df = pd.read_csv(metrics_file)

            # Display full table
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Metrics file not found.")

# --- Tab 4: Predict ---
with tab4:
    st.header("🔮 Predict Using Fake Model")

    predict_files = st.file_uploader(
        "Upload image(s) to predict",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if predict_files:
        st.subheader("Prediction Results")
        cols = st.columns(4)
        for i, file in enumerate(predict_files):
            img = Image.open(file)
            pred_class = random.choice(FAKE_CLASSES)
            conf = round(random.uniform(0.6, 0.99), 2)
            with cols[i % 4]:
                st.image(
                    img,
                    caption=f"{file.name} → {pred_class} ({conf*100:.1f}%)",
                    use_container_width=True,
                )

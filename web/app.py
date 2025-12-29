import streamlit as st
import os
from PIL import Image
from smartsensor.const import KITS
from smartsensor.process.any2jpg import heic2jpg
from smartsensor.process_image import process_image
from smartsensor.e2e import end2end_pipeline
from smartsensor.model.train import fit
from smartsensor.predict import predict_new_data
import shutil
import pandas as pd
import zipfile
import io

# Constants
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
RESULT_DIR = "result"
PREDICTED_DIR = "predict"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

IMAGE_EXT = (".jpg", ".jpeg", ".png")
FAKE_CLASSES = ["class A", "class B", "class C"]

st.set_page_config(page_title="SmartSensor Prototype", layout="wide")
st.title("🧠 SmartSensor ML UI Prototype")

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Images", "🧪 Process Images", "🎓 Train Model", "🔮 Predict"])


def load_and_resize(path: str, max_size=(256, 256)) -> Image.Image:
    img = Image.open(path)
    img.thumbnail(max_size)
    return img


def zip_folder_to_buffer(folder_path: str) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname=arcname)
    zip_buffer.seek(0)
    return zip_buffer


# --- Tab 1: Upload ---
with tab1:
    st.header("📤 Upload Raw Images")

    uploaded = st.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png", "heic"],
        accept_multiple_files=True,
    )

    if uploaded:
        for img_file in uploaded:
            save_path = os.path.join(UPLOAD_DIR, img_file.name)
            with open(save_path, "wb") as f:
                f.write(img_file.getbuffer())

        # Convert HEIC to JPG if necessary
        if img_file.name.lower().endswith(".heic"):
            heic_path = os.path.join(UPLOAD_DIR, img_file.name)
            with open(heic_path, "wb") as f:
                f.write(img_file.getbuffer())
            heic2jpg(UPLOAD_DIR)
            st.success(f"Converted {img_file.name} to JPG.")

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
    kit_version = st.selectbox("🧰 Select Kit Version", KITS.keys(), index=0)
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
        processed_imgs = [f for f in os.listdir(squared_frame_dir) if f.lower().endswith(IMAGE_EXT)]
    else:
        processed_imgs = []

    if processed_imgs:
        st.subheader("🖼️ Processed Images")

        # Clear button
        clear_col, _ = st.columns([1, 9])
        with clear_col:
            if st.button("🗑️ Clear Processed Images"):
                shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
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
    kit_version = st.selectbox("🧰 Select Kit Version", KITS.keys(), index=0, key="kit_version2")
    normalize_method = st.selectbox(
        "🧪 Normalization Method",
        ["raw", "ratio", "delta"],
        index=1,
        key="train_norm_method",
    )

    degree = st.selectbox("📐 Polynomial Degree", ["1", "2"], index=0, key="train_degree")

    cv = st.slider("🔁 Cross Validation Folds", 2, 10, 5, key="train_cv")

    test_size = st.slider("🧪 Test Size (fraction)", 0.1, 0.5, 0.2, 0.1, key="train_test_size")
    skip_fs = st.checkbox("🚫 Skip Feature Selection", value=True, key="train_skip_fs")

    replication = st.slider("Replication ", 100, 1000, 100, 100, key="replication")

    if st.button("🎯 Start Training"):
        with st.spinner("Training..."):
            outdir = os.path.join(RESULT_DIR, f"current_trainning_{normalize_method}")
            os.makedirs(outdir, exist_ok=True)

            # Train test with replication
            full_data, config, metrics = end2end_pipeline(
                data=PROCESSED_DIR,
                kit=kit_version,
                norm=normalize_method,
                features="meanR,meanG,meanB,modeR,modeG,modeB",
                degree=int(degree),
                skip_feature_selection=True,
                cv=cv,
                outdir=outdir,
                prefix=normalize_method,
                test_size=test_size,
                replication=replication,
            )

            # Save model for single image prediction later
            feature_items = "meanR,meanG,meanB,modeR,modeG,modeB".split(",")
            fit(
                train=full_data,
                features=feature_items,
                degree=int(degree),
                skip_feature_selection=skip_fs,
                cv=cv,
                outdir=outdir,
                prefix="final_model",
            )

        st.success("✅ Training completed!")

        # Zip to download
        zip_buffer = zip_folder_to_buffer(outdir)
        # Show metrics
        metrics_file = os.path.join(outdir, "metrics.csv")
        if os.path.isfile(metrics_file):
            st.subheader("📊 Evaluation Metrics")

            df = pd.read_csv(metrics_file)

            # Display full table
            st.dataframe(df, use_container_width=True)

            # Add download button
            st.download_button(
                label="📁 Download ZIP Archive",
                data=zip_buffer,
                file_name=f"smartsensor_degree_{degree}_norm_{normalize_method}.zip",
                mime="application/zip",
            )

        else:
            st.warning("Metrics file not found.")

# --- Tab 4: Predict ---
with tab4:
    st.header("🔮 Predict")
    kit_version = st.selectbox("🧰 Select Kit Version", KITS.keys(), index=0, key="kit_version4")
    models = [f for f in os.listdir(RESULT_DIR) if f != ".gitignore"]
    models = st.selectbox("🤖 Select Trained Model", models, index=0, key="models")
    predict_files = st.file_uploader(
        "Upload image(s) to predict",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button("🧹 Clear prediction folder"):
        shutil.rmtree(PREDICTED_DIR, ignore_errors=True)
        os.makedirs(PREDICTED_DIR, exist_ok=True)
        st.success("🧽 Prediction folder cleaned.")

    # Save newly uploaded files
    if predict_files:
        for i, img_file in enumerate(predict_files):
            save_path = os.path.join(PREDICTED_DIR, img_file.name)
            with open(save_path, "wb") as f:
                f.write(img_file.getbuffer())
        st.success(f"✅ Uploaded {len(predict_files)} image(s).")

    # Display existing uploaded images
    existing_images = [
        f
        for f in sorted(os.listdir(PREDICTED_DIR))
        if os.path.isfile(os.path.join(PREDICTED_DIR, f)) and f.lower().endswith(("jpg", "jpeg", "png"))
    ]

    if existing_images:
        st.subheader("🖼️ New images")
        cols = st.columns(10)
        for i, img_name in enumerate(existing_images):
            img_path = os.path.join(PREDICTED_DIR, img_name)
            img = load_and_resize(img_path)
            with cols[i % 10]:
                st.image(img, caption=img_name, use_container_width=True)

    # Predict button
    if st.button("🎯 Start process image and predict"):
        outdir = os.path.join(RESULT_DIR, normalize_method)
        os.makedirs(outdir, exist_ok=True)
        if not existing_images:
            st.warning("⚠️ Please upload at least one image before starting prediction.")
        else:
            with st.spinner("🧠 Processing and predicting..."):
                try:
                    # Process images first
                    process_image(data=PREDICTED_DIR, outdir=PREDICTED_DIR, kit=kit_version)

                    # Then predict
                    df = predict_new_data(
                        model_dir=os.path.join(RESULT_DIR, models),
                        processed_dir=PREDICTED_DIR,
                        outdir=PREDICTED_DIR,
                    )
                    # Show squared_frame output images
                    squared_frame_path = os.path.join(PREDICTED_DIR, "squared_frame")
                    if os.path.isdir(squared_frame_path):
                        squared_images = [
                            f
                            for f in sorted(os.listdir(squared_frame_path))
                            if f.lower().endswith(("jpg", "jpeg", "png"))
                        ]
                        if squared_images:
                            st.subheader("🔲 Squared Frame Output")
                            cols = st.columns(10)
                            for i, name in enumerate(squared_images):
                                img_path = os.path.join(squared_frame_path, name)
                                img = load_and_resize(img_path)
                                with cols[i % 10]:
                                    st.image(img, caption=name, use_container_width=True)

                    st.success("✅ Prediction completed!")
                    st.subheader("📊 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

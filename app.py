import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


MODEL_PATH = "artifacts/model_trainer/model.keras"
LABELS_PATH = "artifacts/model_trainer/labels.json"
TRAIN_DIR = Path("artifacts/data_ingestion/preprocessed/train")
IMAGE_SIZE = 32


@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    return model, labels


def get_class_sample_image(class_folder: str) -> Image.Image | None:
    folder = TRAIN_DIR / class_folder
    images = list(folder.glob("*.png"))
    if not images:
        return None
    return Image.open(images[0]).convert("L")


def preprocess_canvas(image_data: np.ndarray):
    gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return gray, binary


def segment_characters(gray: np.ndarray, binary: np.ndarray, padding: int = 4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours]
    bboxes = sorted(bboxes, key=lambda b: b[0])
    bboxes = [(x, y, w, h) for x, y, w, h in bboxes if w * h > 100]

    char_images = []
    for x, y, w, h in bboxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray.shape[1], x + w + padding)
        y2 = min(gray.shape[0], y + h + padding)
        char_images.append(gray[y1:y2, x1:x2])

    return char_images, bboxes


def predict_character(char_img: np.ndarray, model, labels: dict) -> tuple[str, float]:
    img = Image.fromarray(char_img).convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    return labels[str(idx)], float(preds[0][idx])


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bangla OCR", layout="centered")
st.title("Bangla Handwritten Word Recognition")
st.write("Draw a Bangla word on the canvas, then click **Predict**.")

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=8,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=200,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Canvas is empty. Please draw something first.")
    else:
        image_data = canvas_result.image_data.astype(np.uint8)
        is_blank = (image_data[:, :, :3] == 255).all()

        if is_blank:
            st.warning("Canvas is empty. Please draw something first.")
        else:
            try:
                model, labels = load_model_and_labels()
            except Exception:
                st.error("Model not found. Run `python train.py` first to train the model.")
                st.stop()

            gray, binary = preprocess_canvas(image_data)
            char_images, bboxes = segment_characters(gray, binary)

            if not char_images:
                st.warning("No characters detected. Try drawing more clearly with spaces between characters.")
            else:
                predictions = [predict_character(img, model, labels) for img in char_images]

                st.subheader("Segmented Characters")
                cols = st.columns(len(char_images))
                for col, char_img, (label, conf) in zip(cols, char_images, predictions):
                    thumb = Image.fromarray(char_img).resize((64, 64), Image.LANCZOS)
                    col.image(thumb, caption=f"Class {label} · {conf:.1%}")

                st.subheader("Recognized Word")
                cols = st.columns(len(predictions))
                for col, (label, conf) in zip(cols, predictions):
                    sample = get_class_sample_image(label)
                    if sample:
                        col.image(sample.resize((64, 64), Image.LANCZOS))
                    else:
                        col.markdown(f"**{label}**")

                st.subheader("Confidence Scores")
                for i, (label, conf) in enumerate(predictions):
                    st.progress(conf, text=f"Character {i + 1}  —  Class {label}:  {conf:.1%}")

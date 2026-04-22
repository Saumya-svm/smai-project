"""
T3.3 — Tamil Handwritten Character Recognition
Streamlit app: draw a Tamil character → live Unicode prediction
"""

import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "tamil_cnn.pth"
LABEL_PATH  = "label_map.json"
IMG_SIZE    = 32
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CANVAS_SIZE = 400
TOP_K       = 5

# ── Model definition (must match train.py) ───────────────────────────────────
class TamilCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Load model & labels ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)   # {"0": "அ", "1": "ஆ", ...}
    num_classes = len(label_map)
    model = TamilCNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, label_map


def preprocess(canvas_data: np.ndarray) -> torch.Tensor:
    """Convert RGBA canvas array to model-ready tensor."""
    # Canvas gives RGBA; use alpha channel as mask
    gray = canvas_data[:, :, 3].astype(np.float32)

    # Invert: drawn strokes should be white on black background
    gray = cv2.bitwise_not(gray.astype(np.uint8))

    # Resize to model input
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize
    tensor = torch.tensor(gray, dtype=torch.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1, 1, H, W)


def predict(model, label_map, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, TOP_K)
    results = [
        (label_map[str(idx.item())], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]
    return results


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tamil Script Recognizer", page_icon="த", layout="centered")

st.title("Tamil Handwritten Character Recognizer")
st.markdown(
    "Draw a **Tamil character** on the canvas below. "
    "The model predicts its Unicode character in real time."
)

try:
    model, label_map = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run `python train.py` first "
        "to train and save the model."
    )
    st.stop()

# ── Tabs: Draw | Upload | Practice ───────────────────────────────────────────
tab_draw, tab_upload, tab_practice = st.tabs(["Draw", "Upload Image", "Practice Mode"])

# ─── Draw tab ────────────────────────────────────────────────────────────────
with tab_draw:
    st.markdown("Use your mouse or touchscreen to draw a Tamil character.")

    col1, col2 = st.columns([2, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color   = "rgba(0,0,0,0)",
            stroke_width = 18,
            stroke_color = "#FFFFFF",
            background_color = "#000000",
            height       = CANVAS_SIZE,
            width        = CANVAS_SIZE,
            drawing_mode = "freedraw",
            key          = "canvas_draw",
            update_streamlit = True,
        )

    with col2:
        st.markdown("### Prediction")
        if canvas_result.image_data is not None:
            arr = canvas_result.image_data
            # Only predict if something is drawn (non-zero alpha pixels)
            if arr[:, :, 3].sum() > 0:
                tensor  = preprocess(arr)
                results = predict(model, label_map, tensor)

                top_char, top_conf = results[0]
                st.markdown(
                    f"<div style='font-size:80px; text-align:center'>{top_char}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** {top_conf*100:.1f}%")

                st.markdown("#### Top {TOP_K} candidates")
                for char, prob in results:
                    st.progress(prob, text=f"{char}  —  {prob*100:.1f}%")
            else:
                st.info("Start drawing to see predictions.")
        else:
            st.info("Start drawing to see predictions.")

# ─── Upload tab ──────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a grayscale image of a Tamil character",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Uploaded image", width=200)

        arr   = np.array(img)
        arr_u = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.tensor(arr_u, dtype=torch.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

        results = predict(model, label_map, tensor)
        top_char, top_conf = results[0]

        st.markdown(
            f"<div style='font-size:80px; text-align:center'>{top_char}</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Confidence:** {top_conf*100:.1f}%")
        for char, prob in results:
            st.progress(prob, text=f"{char}  —  {prob*100:.1f}%")

# ─── Practice tab ────────────────────────────────────────────────────────────
with tab_practice:
    st.markdown(
        "A **target character** is shown. Draw it and see if the model agrees!"
    )

    if "target_idx" not in st.session_state:
        st.session_state.target_idx = 0

    if st.button("New character"):
        st.session_state.target_idx = np.random.randint(0, len(label_map))

    target_char = label_map[str(st.session_state.target_idx)]
    st.markdown(
        f"<div style='font-size:100px; text-align:center; color:#00aaff'>{target_char}</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"Draw the character above: **{target_char}**")

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        canvas_practice = st_canvas(
            fill_color   = "rgba(0,0,0,0)",
            stroke_width = 18,
            stroke_color = "#FFFFFF",
            background_color = "#000000",
            height       = CANVAS_SIZE,
            width        = CANVAS_SIZE,
            drawing_mode = "freedraw",
            key          = "canvas_practice",
            update_streamlit = True,
        )

    with col_p2:
        st.markdown("### Feedback")
        if canvas_practice.image_data is not None:
            arr = canvas_practice.image_data
            if arr[:, :, 3].sum() > 0:
                tensor  = preprocess(arr)
                results = predict(model, label_map, tensor)
                top_char, top_conf = results[0]

                if top_char == target_char:
                    st.success(f"Correct! Model predicted: {top_char}")
                else:
                    st.warning(f"Model predicted: {top_char} ({top_conf*100:.1f}%)")

                for char, prob in results:
                    marker = " ✓" if char == target_char else ""
                    st.progress(prob, text=f"{char}{marker}  —  {prob*100:.1f}%")
            else:
                st.info("Start drawing to get feedback.")

st.markdown("---")
st.caption("SMAI Assignment 3 · T3.3 Tamil Script Recognition · IIIT Hyderabad 2025-26")

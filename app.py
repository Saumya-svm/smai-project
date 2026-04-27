"""
Tamil Handwritten Character Recognition
Streamlit app: draw a Tamil character → live Unicode prediction
"""

import json
import os
import random
import re
from pathlib import Path

import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "checkpoints" / "20260415_063836" / "ckpt_epoch020_best.pth"
LABEL_PATH   = BASE_DIR / "idx_to_class.json"
IMG_SIZE     = 32
RAW_IMG_SIZE = 64
FALLBACK_MEAN = 204.32005463156813 / 255.0
FALLBACK_STD  = 101.75917259098979 / 255.0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CANVAS_SIZE  = 380
TOP_K        = 5

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tamil Script Recognizer",
    page_icon="த",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@400;600&display=swap');
.tamil-display { font-family: 'Noto Sans Tamil', serif; }
.big-tamil     { font-family: 'Noto Sans Tamil', serif; font-size: 7rem; text-align: center; line-height: 1.2; }
.pred-char     { font-family: 'Noto Sans Tamil', serif; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Compact Tamil character reference (vowels + pure consonants) ──────────────
TAMIL_CHARS = {
    "அ":"a","ஆ":"ā","இ":"i","ஈ":"ī","உ":"u","ஊ":"ū",
    "எ":"e","ஏ":"ē","ஐ":"ai","ஒ":"o","ஓ":"ō","ஔ":"au",
    "க":"ka","ங":"ṅa","ச":"ca","ஞ":"ña","ட":"ṭa","ண":"ṇa",
    "த":"ta","ந":"na","ப":"pa","ம":"ma","ய":"ya","ர":"ra",
    "ல":"la","வ":"va","ழ":"ḻa","ள":"ḷa","ற":"ṟa","ன":"ṉa",
}
CHARS = list(TAMIL_CHARS.keys())


# ── Model architecture ────────────────────────────────────────────────────────
class TamilCNN(nn.Module):
    """3-block CNN — must match the architecture used in train.py"""
    def __init__(self, num_classes: int = 156, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_size = (img_size // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# ── Default label map ─────────────────────────────────────────────────────────
def _make_default_label_map(num_classes: int = 156) -> dict:
    """Build a Tamil character label map without an external file."""
    vowels = list("அஆஇஈஉஊஎஏஐஒஓஔ")
    consonants = list("கஞஜஷஸஹக்ஷ")   # pure consonants (virama form not needed here)
    base_consonants = "கசடதநபமயரலவழளறன"
    vowel_signs = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
    compound = [c + v for c in base_consonants for v in vowel_signs]
    all_chars = vowels + [c + "்" for c in base_consonants] + compound
    return {str(i): (all_chars[i] if i < len(all_chars) else f"_c{i}") for i in range(num_classes)}


# ── Checkpoint / preprocessing helpers ────────────────────────────────────────
def _default_preprocess(num_classes: int = 156) -> dict:
    return {
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "raw_img_size": RAW_IMG_SIZE,
        "mean": FALLBACK_MEAN,
        "std": FALLBACK_STD,
        "stats_source": "fallback",
    }


def _resolve_existing_path(path_str: str | None, ckpt_path: str | None = None) -> Path | None:
    if not path_str:
        return None

    path = Path(path_str).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, BASE_DIR / path, Path.cwd() / path])
        if ckpt_path:
            candidates.append(Path(ckpt_path).expanduser().resolve().parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


@st.cache_data(show_spinner=False)
def _compute_train_stats(data_path: str) -> tuple[float, float, int]:
    with h5py.File(data_path, "r") as f:
        ds = f["Train Data/x_train"]
        total = 0
        pixel_sum = 0.0
        pixel_sq_sum = 0.0

        for start in range(0, len(ds), 1024):
            batch = ds[start:start + 1024].astype(np.float64) / 255.0
            pixel_sum += float(batch.sum())
            pixel_sq_sum += float(np.square(batch).sum())
            total += batch.size

        mean = pixel_sum / total
        var = max(pixel_sq_sum / total - mean * mean, 1e-12)
        return mean, float(np.sqrt(var)), int(ds.shape[-1])


def _load_stats_from_train_log(ckpt_path: str) -> tuple[float, float] | None:
    log_path = Path(ckpt_path).expanduser().resolve().parent / "train.log"
    if not log_path.exists():
        return None

    pattern = re.compile(r"Pixel mean:\s*([0-9.]+)\s+std:\s*([0-9.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def _infer_num_classes(state_dict: dict, label_map: dict) -> int:
    if label_map:
        return len(label_map)

    for key in ("classifier.4.weight", "classifier.3.weight", "classifier.1.weight"):
        weight = state_dict.get(key)
        if weight is not None and hasattr(weight, "shape"):
            return int(weight.shape[0])
    return 156


def _build_preprocess_config(ckpt: dict, ckpt_path: str, num_classes: int) -> dict:
    cfg = _default_preprocess(num_classes)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    cfg["img_size"] = int(args.get("img_size", cfg["img_size"]))

    raw_size = ckpt.get("raw_img_size")
    if raw_size is not None:
        cfg["raw_img_size"] = int(raw_size)

    mean = ckpt.get("data_mean", ckpt.get("mean"))
    std = ckpt.get("data_std", ckpt.get("std"))
    if mean is not None and std is not None:
        cfg["mean"] = float(mean)
        cfg["std"] = float(std)
        cfg["stats_source"] = "checkpoint"
        return cfg

    data_path = _resolve_existing_path(args.get("data"), ckpt_path)
    if data_path is not None:
        mean, std, raw_size = _compute_train_stats(str(data_path))
        cfg["mean"] = float(mean)
        cfg["std"] = float(std)
        cfg["raw_img_size"] = int(raw_size)
        cfg["stats_source"] = str(data_path)
        return cfg

    stats = _load_stats_from_train_log(ckpt_path)
    if stats is not None:
        cfg["mean"], cfg["std"] = stats
        cfg["stats_source"] = "train.log"

    return cfg


# ── Load model & labels ───────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt_path: str, label_path: str):
    """Load checkpoint and label map. Returns (model, label_map, preprocess_cfg, error)."""
    preprocess = _default_preprocess()

    # --- label map ---
    label_map: dict = {}
    if os.path.exists(label_path):
        try:
            with open(label_path, encoding="utf-8") as f:
                raw = json.load(f)
            # support both {str: char} and {int: char}
            label_map = {str(k): v for k, v in raw.items()}
        except Exception as e:
            return None, {}, preprocess, f"Label map load error: {e}"

    # --- checkpoint ---
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    except FileNotFoundError:
        return None, label_map, preprocess, f"Checkpoint not found: {ckpt_path}"
    except Exception as e:
        return None, label_map, preprocess, str(e)

    # Detect checkpoint layout and class count
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt

    num_classes = _infer_num_classes(state_dict, label_map)
    preprocess = _build_preprocess_config(ckpt if isinstance(ckpt, dict) else {}, ckpt_path, num_classes)

    # If label_map still empty, build default
    if not label_map:
        label_map = _make_default_label_map(num_classes)

    model = TamilCNN(num_classes, img_size=preprocess["img_size"]).to(DEVICE)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        return None, label_map, preprocess, f"State dict mismatch: {e}"

    model.eval()
    return model, label_map, preprocess, None


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _ensure_light_background(gray: np.ndarray) -> np.ndarray:
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() < 127:
        return 255 - gray
    return gray


def _center_character(gray: np.ndarray, target_side: int) -> np.ndarray | None:
    ink_mask = ((255 - gray) > 10).astype(np.uint8)
    coords = cv2.findNonZero(ink_mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    pad = 4
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, gray.shape[1]), min(y + h + pad, gray.shape[0])
    crop = gray[y1:y2, x1:x2]

    ch, cw = crop.shape
    side = max(ch, cw)
    square = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off:y_off + ch, x_off:x_off + cw] = crop
    return cv2.resize(square, (target_side, target_side), interpolation=cv2.INTER_LINEAR)


def _preprocess_like_training(gray: np.ndarray, preprocess: dict) -> torch.Tensor:
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(preprocess["img_size"], preprocess["img_size"]),
        mode="bilinear",
        align_corners=False,
    )
    tensor = (tensor - preprocess["mean"]) / preprocess["std"]
    return tensor.to(DEVICE)


def preprocess_canvas(img_array: np.ndarray, preprocess: dict):
    """RGBA numpy array (from st_canvas) → model tensor or None."""
    # st_canvas renders background_color as fully opaque pixels (alpha=255
    # everywhere), so the alpha channel cannot distinguish strokes from
    # background.  Use the R channel instead: background="#000000" → R=0,
    # stroke_color="#FFFFFF" → R=255.
    stroke_intensity = img_array[:, :, 0].astype(np.uint8)
    if stroke_intensity.max() < 10:
        return None

    # Invert so the image has dark ink on a white background, matching the
    # training data polarity.
    gray = 255 - stroke_intensity
    canonical = _center_character(gray, preprocess["raw_img_size"])
    if canonical is None:
        return None
    return _preprocess_like_training(canonical, preprocess)


def preprocess_pil(pil_img: Image.Image, preprocess: dict):
    """PIL image → model tensor."""
    gray = np.array(pil_img.convert("L"))
    gray = _ensure_light_background(gray)
    canonical = _center_character(gray, preprocess["raw_img_size"])
    if canonical is None:
        return None
    return _preprocess_like_training(canonical, preprocess)


# ── Prediction ────────────────────────────────────────────────────────────────
def run_predict(model, label_map: dict, tensor, top_k: int = TOP_K):
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=-1).squeeze(0)
    top_probs, top_idx = probs.topk(top_k)
    return [(label_map.get(str(i.item()), str(i.item())), float(p))
            for p, i in zip(top_probs, top_idx)]


def demo_predict(target: str | None = None):
    """Random predictions for demo mode."""
    pool = CHARS[:]
    if target and target in pool:
        pool.remove(target)
    chars = [target] + random.sample(pool, TOP_K - 1) if target else random.sample(CHARS, TOP_K)
    vals  = sorted(np.random.dirichlet(np.ones(TOP_K)).tolist(), reverse=True)
    if target:
        vals[0] = max(vals[0], 0.35)          # bias toward correct in practice mode
        s = sum(vals); vals = [v / s for v in vals]
    return list(zip(chars, vals))


# ── Prediction display ────────────────────────────────────────────────────────
def show_predictions(preds):
    for i, (label, prob) in enumerate(preds):
        sub = TAMIL_CHARS.get(label, "")
        cp  = f"U+{ord(label):04X}" if len(label) == 1 else ""
        c1, c2, c3 = st.columns([1, 5, 1])
        c1.markdown(f"<div class='pred-char'>{label}</div>", unsafe_allow_html=True)
        c2.markdown(f"**{label}** &nbsp; {sub} &nbsp; `{cp}`")
        c2.progress(float(prob))
        c3.markdown(f"**{prob * 100:.1f}%**")
        if i == 0:
            st.divider()


# ── Session state defaults ────────────────────────────────────────────────────
_defaults = dict(
    canvas_key=0, p_canvas_key=100,
    target=random.choice(CHARS),
    correct=0, total=0, feedback=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    ckpt    = st.text_input("Checkpoint (.pth)", value=str(MODEL_PATH))
    mapping = st.text_input("Label map (.json)", value=str(LABEL_PATH))

    model, label_map, preprocess_cfg, err = load_model(ckpt, mapping)

    if model:
        st.success(f"✅ Model loaded — {preprocess_cfg['num_classes']} classes")
        st.caption(
            "Inference preprocessing matches training: "
            f"{preprocess_cfg['raw_img_size']}→{preprocess_cfg['img_size']} bilinear, "
            f"mean={preprocess_cfg['mean']:.4f}, std={preprocess_cfg['std']:.4f}"
        )
    else:
        st.warning("⚠️ Demo mode — no model loaded")
        if err:
            st.caption(err)

    st.divider()
    stroke_w = st.slider("Stroke width", 5, 20, 10)
    st.caption("Draw large, filling most of the canvas. Thinner strokes match training data better.")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔤 Tamil Handwritten Character Recognizer")
st.caption("Draw a Tamil character and get live Unicode predictions.")

tab_draw, tab_practice, tab_upload = st.tabs(
    ["✍️ Draw & Predict", "🎯 Practice Mode", "📁 Upload Image"]
)


# ══ Tab 1: Draw & Predict ═════════════════════════════════════════════════════
with tab_draw:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Draw a character")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_w,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE, width=CANVAS_SIZE,
            drawing_mode="freedraw",
            display_toolbar=True,
            update_streamlit=True,
            key=f"draw_{st.session_state.canvas_key}",
        )
        b1, b2 = st.columns(2)
        predict_btn = b1.button("🔍 Predict", use_container_width=True)
        if b2.button("🗑️ Clear", use_container_width=True):
            st.session_state.canvas_key += 1
            st.rerun()

    with col2:
        st.subheader("Top predictions")
        if predict_btn:
            tensor = None
            if canvas and canvas.image_data is not None:
                tensor = preprocess_canvas(canvas.image_data, preprocess_cfg)
            if tensor is None:
                st.info("Draw a character first!")
            elif model:
                show_predictions(run_predict(model, label_map, tensor))
            else:
                st.info("Demo mode — showing random predictions.")
                show_predictions(demo_predict())
        else:
            st.caption("Press **Predict** after drawing.")


# ══ Tab 2: Practice Mode ══════════════════════════════════════════════════════
with tab_practice:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Target character")
        target = st.session_state.target
        st.markdown(f"<div class='big-tamil'>{target}</div>", unsafe_allow_html=True)
        sub = TAMIL_CHARS.get(target, "")
        cp  = f"U+{ord(target):04X}" if len(target) == 1 else ""
        st.markdown(
            f"<div style='text-align:center;color:gray'>{sub} &nbsp; <code>{cp}</code></div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Score display
        if st.session_state.total > 0:
            acc = st.session_state.correct / st.session_state.total * 100
            st.metric(
                "Accuracy",
                f"{acc:.0f}%",
                delta=f"{st.session_state.correct}/{st.session_state.total} correct",
            )

        if st.button("🔀 New character", use_container_width=True):
            st.session_state.target   = random.choice(CHARS)
            st.session_state.feedback = None
            st.session_state.p_canvas_key += 1
            st.rerun()

        # Feedback
        if st.session_state.feedback is not None:
            ok, predicted = st.session_state.feedback
            if ok:
                st.success("🎉 Correct!")
            else:
                st.error(f"❌ Model predicted **{predicted}** — expected **{target}**")

    with col2:
        st.subheader("Write it here")
        p_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_w,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE, width=CANVAS_SIZE,
            drawing_mode="freedraw",
            display_toolbar=True,
            update_streamlit=True,
            key=f"practice_{st.session_state.p_canvas_key}",
        )
        b1, b2 = st.columns(2)
        if b1.button("✅ Check", use_container_width=True):
            tensor = None
            if p_canvas and p_canvas.image_data is not None:
                tensor = preprocess_canvas(p_canvas.image_data, preprocess_cfg)
            if tensor is None:
                st.warning("Draw something first!")
            else:
                st.session_state.total += 1
                if model:
                    top_char = run_predict(model, label_map, tensor, top_k=1)[0][0]
                else:
                    # Demo: bias 55% toward correct answer
                    top_char = target if random.random() > 0.45 else random.choice(CHARS)
                ok = (top_char == target)
                if ok:
                    st.session_state.correct += 1
                st.session_state.feedback = (ok, top_char)
                st.rerun()
        if b2.button("🗑️ Clear ", use_container_width=True):
            st.session_state.p_canvas_key += 1
            st.rerun()


# ══ Tab 3: Upload Image ═══════════════════════════════════════════════════════
with tab_upload:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Upload image")
        uploaded = st.file_uploader(
            "PNG / JPEG of a Tamil character",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded:
            pil = Image.open(uploaded)
            st.image(pil, width=260, caption="Uploaded image")

    with col2:
        st.subheader("Predictions")
        if uploaded:
            tensor = preprocess_pil(pil, preprocess_cfg)
            if tensor is None:
                st.warning("Could not detect a character in the uploaded image.")
            elif model:
                show_predictions(run_predict(model, label_map, tensor))
            else:
                st.info("Load a model via the sidebar for real predictions.")
                show_predictions(demo_predict())
        else:
            st.caption("Upload an image to see predictions.")


st.divider()
st.caption("SMAI Assignment 3 · T3.3 Tamil Script Recognition · IIIT Hyderabad 2025-26")

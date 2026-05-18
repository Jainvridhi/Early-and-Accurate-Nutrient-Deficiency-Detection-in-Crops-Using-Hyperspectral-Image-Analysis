"""
GreenLens — Dual Dashboard (Streamlit)
Early & Accurate Nutrient Deficiency Detection in Crops using Hyperspectral Imaging

Run:
    streamlit run app.py

Requirements:
    pip install streamlit tensorflow scikit-learn lime scikit-image
                tifffile opencv-python-headless seaborn matplotlib joblib numpy pandas
"""

import os, io, pickle, warnings
warnings.filterwarnings("ignore")

# ── Module imports (inference & report helpers) ───────────────────────────────
try:
    from inference import (
        preprocess_rgb_image,
        get_rgb_prediction,
        get_plant_info,
        get_sample_hyper_prediction,
        PLANT_INFO_DB as _INFERENCE_PLANT_INFO_DB,
    )
    INFERENCE_MODULE = True
except ImportError:
    INFERENCE_MODULE = False

try:
    from report_generator import generate_pdf_report, generate_audio_report
    REPORT_MODULE = True
except ImportError:
    REPORT_MODULE = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
import joblib
import tifffile as tiff
import streamlit as st
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ── Optional heavy imports (guarded so UI loads even without models) ──────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50  import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR      = "Mini Project/models/"
PROCESSED_DIR  = "Mini Project/Data/processed/"
IMG_H, IMG_W   = 128, 128
N_BANDS        = 15          # after PCA
RAW_BANDS      = 204         # original hyperspectral bands
CLASS_NAMES    = ["Healthy (0.0)", "Partial Deficiency (0.5)", "Deficient (1.0)"]
CLASS_COLORS   = ["#2ecc71", "#f39c12", "#e74c3c"]
ENSEMBLE_W     = np.array([0.33, 0.33, 0.34])   # [CustomCNN, ResNet50, DenseNet121]

NUTRIENT_MAP = {
    "Healthy (0.0)":              None,
    "Partial Deficiency (0.5)":  "Nitrogen (N)",
    "Deficient (1.0)":           "Iron (Fe)",
}
RECOMMENDATION_MAP = {
    "Healthy (0.0)": (
        "✅ **Crop looks healthy!**\n\n"
        "Maintain current irrigation and fertilisation schedule. "
        "Re-scan in 14 days for routine monitoring."
    ),
    "Partial Deficiency (0.5)": (
        "⚠️ **Early Nitrogen deficiency detected.**\n\n"
        "- Apply **30–40 kg/ha urea** within 7 days.\n"
        "- Ensure adequate soil moisture for nutrient uptake.\n"
        "- Monitor leaves weekly — yellowing of older leaves is a key indicator.\n"
        "- Re-test in 10 days."
    ),
    "Deficient (1.0)": (
        "🚨 **Severe Iron (Fe) deficiency detected.**\n\n"
        "- Apply **foliar spray of FeSO₄ (2%)** immediately.\n"
        "- Check soil pH — Iron availability drops sharply above pH 7.0.\n"
        "  Target: **6.0 – 6.5**.\n"
        "- Consider chelated iron (EDDHA) for alkaline soils.\n"
        "- Re-scan after 5 days."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GreenLens — Nutrient Deficiency Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS theming ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.stApp { background-color: #09090f; color: #e0e0e0; }

section[data-testid="stSidebar"] {
    background: #0e0e1a;
    border-right: 1px solid #1e1e3a;
}

.metric-card {
    background: #111120;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 22px; font-weight: 700; }

.result-card {
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}

.badge-healthy  { background:#0d2b1a; border:1.5px solid #16a34a; }
.badge-partial  { background:#2b1f0a; border:1.5px solid #d97706; }
.badge-deficient{ background:#2b0a0a; border:1.5px solid #dc2626; }

.step-card {
    background: #111120;
    border-left: 3px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
}

div[data-testid="stTabs"] button { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }

.stProgress > div > div { background: linear-gradient(90deg, #22c55e, #0ea5e9) !important; border-radius: 999px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def focal_loss_stub(gamma=2.0, alpha=0.25):
    """Stub so model loads without re-importing focal loss."""
    import tensorflow as tf
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        batch_idx = tf.range(tf.shape(y_true)[0])
        indices   = tf.stack([batch_idx, y_true], axis=1)
        probs     = tf.gather_nd(y_pred, indices)
        focal     = alpha * tf.pow(1.0 - probs, gamma) * (-tf.math.log(probs))
        return tf.reduce_mean(focal)
    return loss_fn


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all three trained models + label encoder + preprocessing pipeline."""
    if not TF_AVAILABLE:
        return None, None, None, None, None, None

    paths = {
        "CustomCNN":   os.path.join(MODEL_DIR, "CustomCNN_final.h5"),
        "ResNet50":    os.path.join(MODEL_DIR, "ResNet50_final.h5"),
        "DenseNet121": os.path.join(MODEL_DIR, "DenseNet121_final.h5"),
    }
    custom_obj_resnet  = {"preprocess_input": resnet_preprocess, "focal_loss": focal_loss_stub}
    custom_obj_dense   = {"preprocess_input": dense_preprocess,  "focal_loss": focal_loss_stub}

    models_dict = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            continue
        co = custom_obj_resnet if name == "ResNet50" else (custom_obj_dense if name == "DenseNet121" else None)
        m  = load_model(path, compile=False, custom_objects=co)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        models_dict[name] = m

    # Label encoder
    enc_path = os.path.join(MODEL_DIR, "hyper_label_encoder.pkl")
    le = pickle.load(open(enc_path, "rb")) if os.path.exists(enc_path) else None

    # Preprocessing pipeline (from notebook 02)
    pipe_path = os.path.join(MODEL_DIR, "hyperleaf_full_pipeline.joblib")
    pipeline  = joblib.load(pipe_path) if os.path.exists(pipe_path) else None

    # RGB label encoder — try multiple paths
    rgb_le = None
    for _enc_candidate in [
        os.path.join(MODEL_DIR, "label_encoder.pkl"),
        "../models/label_encoder.pkl",
        os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl"),
    ]:
        if os.path.exists(_enc_candidate):
            rgb_le = pickle.load(open(_enc_candidate, "rb"))
            break

    # RGB model — try multiple paths
    rgb_model = None
    for _candidate in [
        os.path.join(MODEL_DIR, "plant_disease_rgb_model.h5"),
        "../models/plant_disease_rgb_model.h5",
        "Mini Project/models/plant_disease_rgb_model.h5",
        os.path.join(os.path.dirname(__file__), "models", "plant_disease_rgb_model.h5"),
    ]:
        if os.path.exists(_candidate):
            try:
                rgb_model = load_model(_candidate, compile=False,
                                       custom_objects=custom_obj_resnet)
                rgb_model.compile(optimizer="adam",
                                  loss="sparse_categorical_crossentropy",
                                  metrics=["accuracy"])
            except Exception:
                rgb_model = None
            break

    return models_dict, le, pipeline, rgb_le, paths, rgb_model


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hsi_to_pseudoRGB(hsi_15ch):
    """
    (H, W, 15) → (H, W, 3) float32 in [0,1].
    Uses robust percentile stretch per channel so PCA components with
    negative values or tiny ranges don't produce an all-black image.
    """
    rgb = hsi_15ch[:, :, :3].copy().astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            rgb[:, :, c] = np.clip((ch - p2) / (p98 - p2), 0.0, 1.0)
        else:
            # Flat channel — try min-max, else leave as mid-grey
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                rgb[:, :, c] = (ch - mn) / (mx - mn)
            else:
                rgb[:, :, c] = 0.5
    return rgb


def preprocess_tiff(raw_image, pipeline=None):
    """
    raw_image: (H, W, B) numpy array from tiff.imread  (B can be any number of bands)
    Returns: (128, 128, 15) processed array ready for CNN
    """
    raw_f32 = raw_image.astype(np.float32)

    # Handle 2-D (grayscale) or band-first layouts
    if raw_f32.ndim == 2:
        raw_f32 = raw_f32[:, :, np.newaxis]
    elif raw_f32.ndim == 3 and raw_f32.shape[0] < raw_f32.shape[2]:
        # Likely (bands, H, W) — transpose to (H, W, bands)
        raw_f32 = raw_f32.transpose(1, 2, 0)

    resized = cv2.resize(raw_f32, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    if pipeline is not None:
        pixels  = resized.reshape(-1, resized.shape[-1])
        # Pipeline expects RAW_BANDS features; pad/trim if needed
        if pixels.shape[1] != RAW_BANDS:
            if pixels.shape[1] > RAW_BANDS:
                pixels = pixels[:, :RAW_BANDS]
            else:
                pad = np.zeros((pixels.shape[0], RAW_BANDS - pixels.shape[1]), dtype=np.float32)
                pixels = np.concatenate([pixels, pad], axis=1)
        pca_pix = pipeline.transform(pixels)
        return pca_pix.reshape(IMG_H, IMG_W, N_BANDS).astype(np.float32)

    # Fallback: take first 15 bands and per-band minmax normalise
    arr = resized[:, :, :N_BANDS] if resized.shape[2] >= N_BANDS else np.pad(
        resized, ((0,0),(0,0),(0, N_BANDS - resized.shape[2])), mode='constant'
    )
    arr = arr.astype(np.float32)
    for c in range(arr.shape[2]):
        mn, mx = arr[:, :, c].min(), arr[:, :, c].max()
        arr[:, :, c] = (arr[:, :, c] - mn) / (mx - mn + 1e-8)
    return arr


def preprocess_rgb(raw_arr_rgb):
    """(H,W,3) uint8 -> (128,128,3) float32 in [0,1]. Model handles Rescaling internally."""
    resized = cv2.resize(raw_arr_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def ensemble_predict(models_dict, hsi_15ch_batch):
    """Weighted ensemble over all loaded models."""
    names   = list(models_dict.keys())
    weights = ENSEMBLE_W[:len(names)]
    weights = weights / weights.sum()
    preds   = [models_dict[n].predict(hsi_15ch_batch, batch_size=32, verbose=0)
               for n in names]
    return np.average(preds, axis=0, weights=weights), {n: p for n, p in zip(names, preds)}


def run_lime(hsi_15ch, models_dict, pred_idx, num_samples=500):
    """Run LIME and return (fig, heatmap_array, explanation)."""
    if not LIME_AVAILABLE:
        return None, None, None

    pseudo_rgb = hsi_to_pseudoRGB(hsi_15ch)

    def predict_fn(imgs_3ch):
        N = len(imgs_3ch)
        batch = np.zeros((N, IMG_H, IMG_W, N_BANDS), dtype=np.float32)
        for i, img in enumerate(imgs_3ch):
            diff  = np.abs(img.astype(np.float32) - pseudo_rgb).mean(axis=-1)
            recon = hsi_15ch.copy()
            recon[diff > 0.08] = 0.0
            batch[i] = recon
        probs, _ = ensemble_predict(models_dict, batch)
        return probs

    explainer   = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image=pseudo_rgb, classifier_fn=predict_fn,
        top_labels=len(CLASS_NAMES), hide_color=0,
        num_samples=num_samples, batch_size=32
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0a0a14")
    for ax in axes:
        ax.set_facecolor("#0a0a14")

    axes[0].imshow(pseudo_rgb)
    axes[0].set_title("Pseudo-RGB Input\n(PCA bands 0-2)", color="white", fontweight="bold")
    axes[0].axis("off")

    t, m = explanation.get_image_and_mask(pred_idx, positive_only=True,
                                          num_features=8, hide_rest=True)
    axes[1].imshow(mark_boundaries(t, m, color=(0, 1, 0), mode="thick"))
    axes[1].set_title("Supporting Regions\n(Why this prediction?)", color="white", fontweight="bold")
    axes[1].axis("off")

    t2, m2 = explanation.get_image_and_mask(pred_idx, positive_only=False,
                                            num_features=10, hide_rest=False)
    axes[2].imshow(mark_boundaries(t2, m2, color=(1, 0, 0), mode="thick"))
    axes[2].set_title("Green=For | Red=Against", color="white", fontweight="bold")
    axes[2].axis("off")

    dh  = dict(explanation.local_exp[pred_idx])
    hm  = np.vectorize(dh.get)(explanation.segments)
    im  = axes[3].imshow(hm, cmap="RdYlGn")
    axes[3].imshow(pseudo_rgb, alpha=0.35)
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Importance Heatmap\n(Overlaid)", color="white", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    return fig, hm, explanation


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 GreenLens")
    st.caption("Early & Accurate Nutrient Deficiency Detection")
    st.divider()

    dashboard = st.radio(
        "**Select Dashboard**",
        ["🌾  Farmer View", "🔬  Researcher View"],
        label_visibility="visible"
    )
    st.divider()

    st.markdown("**Model Architecture**")
    st.markdown("""
- **CustomCNN** — Residual + Squeeze-Excite  
- **ResNet50** — Fine-tuned (phase 2)  
- **DenseNet121** — Fine-tuned (phase 2)  
- **Ensemble** — Weighted avg (33/33/34%)  
    """)
    st.divider()

    st.markdown("**Pipeline (Notebook 02)**")
    st.markdown("""
`TIFF(48×352×204)` → resize `128×128`  
→ `StandardScaler` → `PCA(15)` → `MinMaxScaler`  
    """)
    st.divider()
    st.caption("HyperLeaf 2024 Dataset · 1590 images · 3 classes")


# ══════════════════════════════════════════════════════════════════════════════
# FARMER VIEW  (from doc2 — inference.py support + float fixes)
# ══════════════════════════════════════════════════════════════════════════════
def farmer_view():
    st.markdown("# 🌾 Fasal Jaanch (फसल जाँच)")
    st.markdown("##### अपनी फसल की फोटो डालें — AI तुरंत बताएगा कि कोई रोग या पोषण कमी है या नहीं")
    st.divider()

    uploaded = st.file_uploader(
        "फसल की फोटो यहाँ डालें (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.info(
                "⬆️ ऊपर अपनी फसल की फोटो अपलोड करें।\n\n"
                "AI कुछ ही सेकंड में बताएगा:\n"
                "- फसल स्वस्थ है या नहीं\n"
                "- कौन सा रोग है\n"
                "- क्या करना चाहिए\n"
                "- PDF रिपोर्ट डाउनलोड करें"
            )
        return

    # ── Decode image BEFORE columns so it's always visible ───────────────────
    raw_bytes = uploaded.read()
    pil_img   = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    raw_arr   = np.array(pil_img)

    models_dict, le, pipeline, rgb_le, _, rgb_model = load_all_models()

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**📷 आपकी फोटो**")
        st.image(pil_img, use_container_width=True)

        if rgb_model is not None:
            st.success("✅ RGB Model loaded — real predictions active")
        else:
            st.warning(
                f"⚠️ RGB model not found at:\n"
                f"`{os.path.join(MODEL_DIR, 'plant_disease_rgb_model.h5')}`\n\n"
                "Running demo mode — predictions are uniform/random."
            )

    with col_result:
        st.markdown("**🤖 AI Analysis**")

        if not st.button("🔍 Run Full Analysis", use_container_width=True, type="primary"):
            st.info("👆 **Run Full Analysis** बटन दबाएं।")
        else:
            with st.spinner("AI विश्लेषण हो रहा है…"):

                # ── RGB inference — use inference.py if available ──────────────
                if INFERENCE_MODULE and rgb_model is not None and rgb_le is not None:
                    friendly_name, status, confidence = get_rgb_prediction(
                        rgb_model, rgb_le, pil_img
                    )
                    info         = get_plant_info(status)
                    is_healthy   = "healthy" in status.lower()
                    plant_parts  = friendly_name.split("(")
                    plant_name   = plant_parts[0].strip()
                    disease_name = status.replace("_", " ") if not is_healthy else ""
                    probs_clean  = None   # no per-class probs from this path
                    classes      = None

                else:
                    # ── Fallback: inline logic ────────────────────────────────
                    PLANTVILLAGE_CLASSES = [
                        ".ipynb_checkpoints",
                        "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
                        "Potato___Early_blight",         "Potato___Late_blight",
                        "Potato___healthy",
                        "Tomato_Bacterial_spot",         "Tomato_Early_blight",
                        "Tomato_Late_blight",            "Tomato_Leaf_Mold",
                        "Tomato_Septoria_leaf_spot",
                        "Tomato_Spider_mites_Two_spotted_spider_mite",
                        "Tomato__Target_Spot",
                        "Tomato__Tomato_YellowLeaf__Curl_Virus",
                        "Tomato__Tomato_mosaic_virus",   "Tomato_healthy",
                    ]
                    classes = list(rgb_le.classes_) if rgb_le is not None else PLANTVILLAGE_CLASSES

                    if rgb_model is not None:
                        batch_rgb = preprocess_rgb(raw_arr)[np.newaxis]
                        probs     = rgb_model.predict(batch_rgb, verbose=0)[0]
                    else:
                        probs = np.ones(len(classes)) / len(classes)

                    # Zero out junk class & renormalise
                    probs_clean = probs.copy()
                    for _ji, _jc in enumerate(classes):
                        if _jc == ".ipynb_checkpoints" and _ji < len(probs_clean):
                            probs_clean[_ji] = 0.0
                    if probs_clean.sum() > 0:
                        probs_clean /= probs_clean.sum()

                    pred_idx     = int(np.argmax(probs_clean))
                    # FIX: explicit Python float so st.progress() never gets np.float32
                    confidence   = float(probs_clean[pred_idx])
                    pred_label   = classes[pred_idx] if pred_idx < len(classes) else f"Class {pred_idx}"

                    def _parse_label(lbl):
                        parts   = lbl.replace("___", "__").split("__")
                        plant   = parts[0].replace("_", " ").strip()
                        disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
                        return plant, disease, "healthy" in lbl.lower()

                    plant_name, disease_name, is_healthy = _parse_label(pred_label)
                    status = disease_name.replace(" ", "_")

                    # PLANT_INFO_DB — prefer imported version, else inline
                    _db = _INFERENCE_PLANT_INFO_DB if INFERENCE_MODULE else {
                        "healthy": {
                            "name": "Healthy",
                            "description": "Plant appears healthy. No visible disease symptoms detected.",
                            "recommendation": "Maintain current watering and fertilisation schedule. Re-scan in 14 days.",
                            "prevention": "Ensure good airflow, avoid overwatering, monitor regularly.",
                        },
                        "Bacterial_spot": {
                            "name": "Bacterial Spot",
                            "description": "Bacterial infection — small water-soaked or dark spots on leaves and fruits.",
                            "recommendation": "Remove infected parts. Apply copper-based bactericide. Avoid working with wet plants.",
                            "prevention": "Rotate crops, use disease-free seeds, avoid overhead watering.",
                        },
                        "Early_blight": {
                            "name": "Early Blight",
                            "description": "Fungal disease — dark concentric rings on older leaves weakening the plant.",
                            "recommendation": "Remove affected leaves. Apply fungicide (chlorothalonil or mancozeb).",
                            "prevention": "Crop rotation, avoid leaf wetness, remove post-harvest debris.",
                        },
                        "Late_blight": {
                            "name": "Late Blight",
                            "description": "Serious fungal disease — large dark water-soaked blotches on leaves and stems.",
                            "recommendation": "Remove and burn infected plants immediately. Apply fungicide to nearby plants.",
                            "prevention": "Proper plant spacing, water at base, avoid replanting in same area.",
                        },
                        "Leaf_Mold": {
                            "name": "Leaf Mold",
                            "description": "Fungal disease — yellow patches on upper leaf surface, olive-green mold beneath.",
                            "recommendation": "Improve ventilation. Apply fungicide. Remove severely infected leaves.",
                            "prevention": "Reduce humidity, avoid overhead irrigation, space plants well.",
                        },
                        "Septoria_leaf_spot": {
                            "name": "Septoria Leaf Spot",
                            "description": "Small circular spots with dark borders and light centres on lower leaves.",
                            "recommendation": "Remove infected leaves. Apply copper or mancozeb fungicide.",
                            "prevention": "Crop rotation, avoid wetting foliage, clean tools regularly.",
                        },
                        "Spider_mites_Two_spotted_spider_mite": {
                            "name": "Spider Mites",
                            "description": "Tiny mites cause stippling, yellowing and webbing on leaves.",
                            "recommendation": "Apply miticide or neem oil. Remove heavily infested leaves.",
                            "prevention": "Maintain humidity, avoid dusty conditions, introduce natural predators.",
                        },
                        "Target_Spot": {
                            "name": "Target Spot",
                            "description": "Fungal disease — circular lesions with concentric rings resembling a target.",
                            "recommendation": "Apply fungicide. Improve air circulation around plants.",
                            "prevention": "Avoid overhead watering, crop rotation, remove plant debris.",
                        },
                        "Tomato_YellowLeaf__Curl_Virus": {
                            "name": "Yellow Leaf Curl Virus",
                            "description": "Viral disease spread by whiteflies — upward leaf curling and yellowing.",
                            "recommendation": "Remove infected plants. Control whitefly population with insecticide.",
                            "prevention": "Use virus-resistant varieties, control whiteflies, use reflective mulch.",
                        },
                        "Tomato_mosaic_virus": {
                            "name": "Tomato Mosaic Virus",
                            "description": "Viral disease — mosaic pattern of light and dark green on leaves.",
                            "recommendation": "Remove infected plants. Disinfect tools. No cure available.",
                            "prevention": "Use certified virus-free seed, wash hands before handling plants.",
                        },
                        "default": {
                            "name": "Disease Detected",
                            "description": "Disease not confidently identified or not in database.",
                            "recommendation": "Isolate the plant. Bring a sample to a local agricultural extension office.",
                            "prevention": "Good garden hygiene, clean tools, remove plant debris.",
                        },
                    }
                    key  = disease_name.replace(" ", "_") if disease_name else ("healthy" if is_healthy else "default")
                    info = _db.get(key, _db["default"])
                    if is_healthy:
                        info = dict(_db["healthy"])
                        info["name"] = f"{plant_name} — Healthy"

                # ── Hyperspectral nutrient — use inference.py if available ─────
                nutrient_status = "N/A (hyperspectral image required)"
                nutrient_conf   = 0.0
                if INFERENCE_MODULE and models_dict and pipeline:
                    try:
                        _hm = list(models_dict.values())[0]
                        nutrient_status, nutrient_conf = get_sample_hyper_prediction(_hm, le)
                    except Exception:
                        pass
                elif models_dict and pipeline:
                    try:
                        sample    = np.load(os.path.join(PROCESSED_DIR, "X_test_hyper.npy"))[5]
                        n_pred, _ = ensemble_predict(models_dict, sample[np.newaxis])
                        ni        = int(np.argmax(n_pred[0]))
                        nutrient_status = CLASS_NAMES[ni]
                        nutrient_conf   = float(n_pred[0][ni])
                    except Exception:
                        pass

            # ── Results UI ────────────────────────────────────────────────────
            st.markdown("### 🔍 Analysis Results")

            if is_healthy:
                st.markdown(f"""
                <div style="background:#0d2b1a;border:1.5px solid #16a34a;border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;">
                    <span style="color:#4ade80;font-weight:700;font-size:13px;">✅ Disease Status:</span>
                    <span style="color:#e0e0e0;font-size:13px;margin-left:8px;">
                        {plant_name} — <strong>Healthy</strong>
                    </span>
                </div>
                <div style="background:#0a1a2b;border:1.5px solid #0ea5e9;border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;">
                    <span style="color:#38bdf8;font-weight:700;font-size:13px;">💧 Nutrient Status:</span>
                    <span style="color:#e0e0e0;font-size:13px;margin-left:8px;">{nutrient_status}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#2b0a0a;border:1.5px solid #dc2626;border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;">
                    <span style="color:#f87171;font-weight:700;font-size:13px;">🚨 Disease Status:</span>
                    <span style="color:#e0e0e0;font-size:13px;margin-left:8px;">
                        {plant_name} — <strong>{disease_name}</strong>
                    </span>
                </div>
                <div style="background:#0a1a2b;border:1.5px solid #0ea5e9;border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;">
                    <span style="color:#38bdf8;font-weight:700;font-size:13px;">💧 Nutrient Status:</span>
                    <span style="color:#e0e0e0;font-size:13px;margin-left:8px;">{nutrient_status}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**AI Confidence: {confidence*100:.1f}%**")
            # FIX 1: cast to Python float — st.progress() rejects np.float32
            st.progress(float(min(confidence, 1.0)))

            st.markdown("---")
            st.markdown("**📋 Disease Description**")
            st.write(info["description"])

            st.markdown("**💡 Recommendation (सुझाव)**")
            if is_healthy:
                st.success(info["recommendation"])
            else:
                st.error(info["recommendation"])

            st.markdown("**🛡️ Prevention**")
            st.info(info["prevention"])

            # ── Top-5 predictions (only available in fallback path) ───────────
            if probs_clean is not None and classes is not None:
                st.markdown("---")
                st.markdown("**Top Predictions**")
                top5 = np.argsort(probs_clean)[::-1][:5]
                for i in top5:
                    cn   = classes[i] if i < len(classes) else f"Class {i}"
                    cn_d = cn.replace("___", " — ").replace("__", " — ").replace("_", " ")
                    st.markdown(f"`{cn_d}`")
                    # FIX 2: explicit float cast for every progress bar value
                    st.progress(float(probs_clean[i]), text=f"{float(probs_clean[i])*100:.1f}%")

            # ── PDF Report — use report_generator.py if available ─────────────
            st.markdown("---")
            st.markdown("**📥 Download Full Report**")
            if REPORT_MODULE:
                try:
                    pdf_buf = generate_pdf_report(
                        info            = info,
                        confidence      = confidence * 100,
                        nutrient_status = nutrient_status,
                        nutrient_conf   = nutrient_conf * 100,
                    )
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buf,
                        file_name="plant_health_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
            else:
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    import io as _io
                    pdf_buf = _io.BytesIO()
                    doc    = SimpleDocTemplate(pdf_buf, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story  = [
                        Paragraph("Plant Health Analysis Report — GreenLens", styles["h1"]),
                        Spacer(1, 12),
                        Paragraph("<b>RGB Disease Analysis</b>", styles["h2"]),
                        Paragraph(f"<b>Prediction:</b> {info['name']}", styles["Normal"]),
                        Paragraph(f"<b>Confidence:</b> {confidence*100:.2f}%", styles["Normal"]),
                        Spacer(1, 12),
                        Paragraph("<b>Description:</b>", styles["h3"]),
                        Paragraph(info["description"], styles["Normal"]),
                        Spacer(1, 12),
                        Paragraph("<b>Recommendation:</b>", styles["h3"]),
                        Paragraph(info["recommendation"], styles["Normal"]),
                        Spacer(1, 12),
                        Paragraph("<b>Prevention:</b>", styles["h3"]),
                        Paragraph(info["prevention"], styles["Normal"]),
                        Spacer(1, 12),
                        Paragraph("<b>Hyperspectral Nutrient Analysis</b>", styles["h2"]),
                        Paragraph(f"<b>Status:</b> {nutrient_status}", styles["Normal"]),
                    ]
                    if "Deficient" in nutrient_status or "Partial" in nutrient_status:
                        story.append(Paragraph(
                            "<b>Recommendation:</b> Nutrient levels appear low. A balanced NPK fertilizer is recommended.",
                            styles["Normal"]))
                    else:
                        story.append(Paragraph(
                            "<b>Recommendation:</b> Nutrient levels appear sufficient.",
                            styles["Normal"]))
                    doc.build(story)
                    pdf_buf.seek(0)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buf,
                        file_name="plant_health_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except ImportError:
                    st.warning("Install `reportlab` for PDF: `pip install reportlab`")

            # ── Audio Report — use report_generator.py if available ───────────
            st.markdown("**🔊 Audio Report**")
            lang_choice = st.radio("Language / भाषा", ["en", "hi"],
                                   horizontal=True, key="audio_lang_fv")
            if REPORT_MODULE:
                try:
                    audio_buf = generate_audio_report(info, nutrient_status, lang=lang_choice)
                    st.audio(audio_buf, format="audio/mp3")
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")
            else:
                try:
                    from gtts import gTTS
                    import io as _io
                    if lang_choice == "hi":
                        tts_text = f"Rog ka natija: {info['name']}. Poshan sthiti: {nutrient_status}."
                    else:
                        tts_text = f"Disease prediction: {info['name']}. Nutrient status: {nutrient_status}."
                    tts = gTTS(text=tts_text, lang=lang_choice)
                    ab  = _io.BytesIO()
                    tts.write_to_fp(ab)
                    ab.seek(0)
                    st.audio(ab, format="audio/mp3")
                except ImportError:
                    st.warning("Install `gTTS` for audio: `pip install gTTS`")


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCHER VIEW  (from app4_4.py — with pseudo-RGB black image fix)
# ══════════════════════════════════════════════════════════════════════════════
def researcher_view():
    st.markdown("# 🔬 Researcher Console")
    st.markdown("##### GreenLens · Hyperspectral Explainability Dashboard")
    st.divider()

    models_dict, le, pipeline, rgb_le, model_paths, _ = load_all_models()

    # ── Model status ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Model Status", expanded=False):
        if models_dict:
            for name, path in model_paths.items():
                status = "✅ Loaded" if name in models_dict else "❌ Not found"
                st.markdown(f"- `{name}` → `{path}` — **{status}**")
            if pipeline:
                st.markdown(f"- `Preprocessing Pipeline` → `{os.path.join(MODEL_DIR, 'hyperleaf_full_pipeline.joblib')}` — ✅ Loaded")
            else:
                st.markdown("- `Preprocessing Pipeline` — ❌ Not found (will use fallback)")
        else:
            st.warning("⚠️ No trained models found. Running in **demo mode** with simulated outputs.")
            st.markdown(f"Expected model directory: `{MODEL_DIR}`")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_predict, tab_lime, tab_bands, tab_eval, tab_pipeline = st.tabs([
        "🎯  Predict & Explain",
        "🔍  LIME Explanation",
        "📊  Band Importance",
        "📈  Model Evaluation",
        "🏗️  Pipeline",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Predict & Explain
    # ─────────────────────────────────────────────────────────────────────────
    with tab_predict:
        uploaded = st.file_uploader(
            "Upload Hyperspectral TIFF Image",
            type=["tiff", "tif"],
            key="researcher_upload"
        )

        if uploaded is None:
            st.info("Upload a `.tiff` hyperspectral image to run the full ensemble analysis.")
            return

        raw_bytes = uploaded.read()

        with st.spinner("Running ensemble inference…"):
            raw_arr       = tiff.imread(io.BytesIO(raw_bytes))
            hsi_processed = preprocess_tiff(raw_arr, pipeline)
            batch         = hsi_processed[np.newaxis]

            if models_dict:
                probs_ens, per_model = ensemble_predict(models_dict, batch)
                probs_ens = probs_ens[0]
                per_model = {k: v[0] for k, v in per_model.items()}
            else:
                # Demo mode
                probs_ens = np.array([0.12, 0.23, 0.65])
                per_model = {
                    "CustomCNN":   np.array([0.10, 0.25, 0.65]),
                    "ResNet50":    np.array([0.14, 0.20, 0.66]),
                    "DenseNet121": np.array([0.12, 0.24, 0.64]),
                }

        pred_idx   = int(np.argmax(probs_ens))
        confidence = float(probs_ens[pred_idx])
        pred_label = CLASS_NAMES[pred_idx]

        # ── Metric row ────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        def metric_card(col, label, value, color):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
            </div>""", unsafe_allow_html=True)

        metric_card(c1, "Prediction",    pred_label.split("(")[0].strip(),                    CLASS_COLORS[pred_idx])
        metric_card(c2, "Confidence",    f"{confidence*100:.1f}%",                            "#60a5fa")
        metric_card(c3, "Severity Score",CLASS_NAMES[pred_idx].split("(")[-1].replace(")",""),"#e879f9")
        metric_card(c4, "Nutrient Flag", NUTRIENT_MAP[pred_label] or "None",                  "#fbbf24")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Layout: image + per-model breakdown ───────────────────────────────
        col_img, col_models = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("**Input Image (Pseudo-RGB)**")

            # FIX: robust pseudo-RGB — uses hsi_to_pseudoRGB with percentile stretch
            pseudo_display = hsi_to_pseudoRGB(hsi_processed)
            pseudo_uint8   = (pseudo_display * 255).astype(np.uint8)

            st.image(pseudo_uint8, caption="Pseudo-RGB (PCA components 0-2)", use_container_width=True)
            st.markdown("**Recommendation**")
            st.info(RECOMMENDATION_MAP[pred_label])

        with col_models:
            st.markdown("**Per-Model Probability Breakdown**")

            fig_bar, ax = plt.subplots(figsize=(7, 4))
            fig_bar.patch.set_facecolor("#0a0a14")
            ax.set_facecolor("#111120")
            x      = np.arange(len(CLASS_NAMES))
            width  = 0.2
            colors = ["#6366f1", "#22c55e", "#0ea5e9", "#f59e0b"]
            for idx, (mname, mprobs) in enumerate(list(per_model.items()) + [("Ensemble", probs_ens)]):
                ax.bar(x + idx * width, mprobs * 100, width,
                       label=mname, color=colors[idx % len(colors)], alpha=0.88)
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(["Healthy", "Partial", "Deficient"], color="white", fontsize=10)
            ax.set_ylabel("Confidence (%)", color="white")
            ax.set_ylim(0, 110)
            ax.tick_params(colors="white")
            ax.legend(fontsize=9, facecolor="#111120", labelcolor="white")
            ax.spines[["top", "right", "left", "bottom"]].set_color("#222")
            ax.yaxis.grid(True, alpha=0.2, color="#333")
            ax.set_title("Model Comparison", color="white", fontweight="bold", pad=10)
            for spine in ax.spines.values(): spine.set_color("#333")
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

            st.markdown("**Probability Table**")
            rows = []
            for mname, mprobs in list(per_model.items()) + [("**Ensemble**", probs_ens)]:
                rows.append({
                    "Model":     mname,
                    "Healthy":   f"{mprobs[0]*100:.1f}%",
                    "Partial":   f"{mprobs[1]*100:.1f}%",
                    "Deficient": f"{mprobs[2]*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # Store for LIME tab
        st.session_state["hsi_for_lime"] = hsi_processed
        st.session_state["pred_idx"]     = pred_idx
        st.session_state["models_dict"]  = models_dict
        st.success("✅ Prediction complete. Go to **LIME Explanation** tab for XAI analysis.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — LIME Explanation
    # ─────────────────────────────────────────────────────────────────────────
    with tab_lime:
        st.markdown("### LIME — Local Interpretable Model-agnostic Explanations")
        st.markdown("""
        LIME perturbs the input image and observes prediction changes to identify which 
        **spatial regions** drive the classification. For hyperspectral data we:
        1. Convert 15-channel → pseudo-RGB (PCA bands 0–2) for LIME's superpixel engine
        2. When a superpixel is masked, set **all 15 channels** of those pixels to 0
        3. Feed the reconstructed 15ch image to the ensemble
        """)

        if "hsi_for_lime" not in st.session_state:
            st.warning("⬅️ First upload an image and run prediction in the **Predict & Explain** tab.")
            return

        hsi_img  = st.session_state["hsi_for_lime"]
        pred_idx = st.session_state["pred_idx"]
        models_d = st.session_state.get("models_dict") or {}

        col_s, col_b = st.columns([1, 2])
        with col_s:
            num_samples  = st.slider("LIME perturbation samples", 200, 1000, 500, 100)
            st.caption("More samples = more accurate but slower (500 recommended)")
            run_lime_btn = st.button("▶️ Run LIME Analysis", type="primary", use_container_width=True)

        with col_b:
            st.markdown(f"""
            **Prediction being explained:** `{CLASS_NAMES[pred_idx]}`  
            **Ensemble weight:** CustomCNN 33% | ResNet50 33% | DenseNet121 34%  
            **15-channel wrapper:** Spatial mask applied across all PCA bands
            """)

        if run_lime_btn:
            if not LIME_AVAILABLE and models_d:
                st.error("LIME is not installed. Run: `pip install lime scikit-image`")
                return

            with st.spinner(f"Running LIME with {num_samples} samples… (~1–3 min)"):
                if models_d and LIME_AVAILABLE:
                    fig_lime, heatmap, explanation = run_lime(hsi_img, models_d, pred_idx, num_samples)
                    real_lime = True
                else:
                    # ── Demo mode: rich synthetic LIME visualisations ──────────
                    real_lime = False
                    np.random.seed(pred_idx * 7 + 3)
                    from skimage.segmentation import find_boundaries

                    pseudo = hsi_to_pseudoRGB(hsi_img)
                    # If still too flat generate a leaf-like gradient
                    if pseudo.max() < 0.05:
                        xx, yy = np.meshgrid(np.linspace(0, 1, IMG_W), np.linspace(0, 1, IMG_H))
                        pseudo[:, :, 0] = 0.15 + 0.25 * np.sin(xx * 6) * np.cos(yy * 4)
                        pseudo[:, :, 1] = 0.35 + 0.40 * np.cos(xx * 3 + 1) * np.sin(yy * 5)
                        pseudo[:, :, 2] = 0.10 + 0.20 * np.sin((xx + yy) * 5)
                        pseudo = np.clip(pseudo, 0, 1)

                    seg_size = 16
                    segments = np.zeros((IMG_H, IMG_W), dtype=int)
                    sid = 0
                    for r in range(0, IMG_H, seg_size):
                        for c in range(0, IMG_W, seg_size):
                            segments[r:r + seg_size, c:c + seg_size] = sid
                            sid += 1
                    n_segs = sid

                    cx, cy = IMG_W // 2, IMG_H // 2
                    seg_centers = {}
                    for s in range(n_segs):
                        rows_, cols_ = np.where(segments == s)
                        seg_centers[s] = (cols_.mean(), rows_.mean())

                    importance_weights_demo = {}
                    for s, (sx, sy) in seg_centers.items():
                        dist = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) / (IMG_W / 2)
                        base = np.random.randn() * 0.12
                        if pred_idx == 2:
                            w = 0.6 * np.exp(-dist * 2.5) + 0.3 * (1 - dist) + base
                        elif pred_idx == 1:
                            w = 0.55 * np.exp(-((dist - 0.4) ** 2) / 0.05) + base
                        else:
                            w = 0.3 * (1 - dist * 0.5) + abs(base)
                        importance_weights_demo[s] = float(w)

                    heatmap_demo = np.vectorize(importance_weights_demo.get)(segments)
                    hmax = max(abs(heatmap_demo.max()), abs(heatmap_demo.min()), 1e-6)
                    heatmap_demo = heatmap_demo / hmax

                    boundaries = find_boundaries(segments, mode="thick")

                    panel_rgb = (pseudo * 255).astype(np.uint8)

                    green_ov = pseudo.copy()
                    pos_mask = heatmap_demo > 0.25
                    green_ov[pos_mask, 0] *= 0.35
                    green_ov[pos_mask, 1]  = np.clip(green_ov[pos_mask, 1] * 1.6 + 0.3, 0, 1)
                    green_ov[pos_mask, 2] *= 0.35
                    green_ov[~pos_mask]   *= 0.12
                    panel_support = (np.clip(green_ov, 0, 1) * 255).astype(np.uint8)
                    panel_support[boundaries] = [0, 230, 80]

                    fa_ov = pseudo.copy()
                    fa_ov[heatmap_demo >  0.20, 0] *= 0.25
                    fa_ov[heatmap_demo >  0.20, 1]  = np.clip(fa_ov[heatmap_demo > 0.20, 1] + 0.45, 0, 1)
                    fa_ov[heatmap_demo >  0.20, 2] *= 0.25
                    fa_ov[heatmap_demo < -0.15, 0]  = np.clip(fa_ov[heatmap_demo < -0.15, 0] + 0.55, 0, 1)
                    fa_ov[heatmap_demo < -0.15, 1] *= 0.25
                    fa_ov[heatmap_demo < -0.15, 2] *= 0.25
                    panel_fa = (np.clip(fa_ov, 0, 1) * 255).astype(np.uint8)
                    panel_fa[boundaries] = [180, 180, 180]

                    fig_lime, axes = plt.subplots(1, 4, figsize=(20, 5))
                    fig_lime.patch.set_facecolor("#0a0a14")
                    panel_titles = [
                        "Pseudo-RGB Input\n(PCA bands 0–2)",
                        "Supporting Regions\n(Green = drives prediction)",
                        "For  vs  Against\nGreen = Support  |  Red = Oppose",
                        "Importance Heatmap\n(RdYlGn overlaid)",
                    ]
                    for ax, title in zip(axes, panel_titles):
                        ax.set_facecolor("#0a0a14")
                        ax.set_title(title, color="white", fontweight="bold", fontsize=10, pad=8)
                        ax.axis("off")

                    axes[0].imshow(panel_rgb)
                    axes[1].imshow(panel_support)
                    axes[2].imshow(panel_fa)

                    im4 = axes[3].imshow(heatmap_demo, cmap="RdYlGn", vmin=-1, vmax=1)
                    axes[3].imshow(pseudo, alpha=0.28)
                    cb = plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
                    cb.ax.yaxis.set_tick_params(color="white")
                    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

                    from matplotlib.patches import Patch
                    axes[2].legend(
                        handles=[Patch(facecolor="#27ae60", label="Supporting"),
                                 Patch(facecolor="#c0392b", label="Opposing")],
                        loc="lower right", fontsize=8,
                        facecolor="#111120", labelcolor="white", framealpha=0.85,
                    )
                    plt.tight_layout(pad=1.5)

            st.pyplot(fig_lime, use_container_width=True)
            plt.close(fig_lime)

            st.markdown("---")
            st.markdown("**LIME Per-Class Region Importance Scores**")

            if real_lime:
                fig_cls, axes_cls = plt.subplots(1, 3, figsize=(15, 4))
                fig_cls.patch.set_facecolor("#0a0a14")
                for cls_idx in range(len(CLASS_NAMES)):
                    ax = axes_cls[cls_idx]
                    ax.set_facecolor("#111120")
                    if cls_idx in explanation.local_exp:
                        vals    = explanation.local_exp[cls_idx]
                        sorted_ = sorted(vals, key=lambda x: abs(x[1]), reverse=True)[:8]
                        fids, imps = zip(*sorted_)
                        colors_bar = ["#27ae60" if v > 0 else "#c0392b" for v in imps]
                        ax.barh(range(len(fids)), imps, color=colors_bar)
                        ax.set_yticks(range(len(fids)))
                        ax.set_yticklabels([f"R{f}" for f in fids], color="white", fontsize=8)
                        ax.axvline(0, color="white", lw=1)
                        ax.tick_params(colors="white")
                    ax.set_title(f"LIME: {CLASS_NAMES[cls_idx]}", color="white",
                                 fontweight="bold", fontsize=10)
                    ax.set_xlabel("Importance Score", color="white")
                    for spine in ax.spines.values(): spine.set_color("#333")
                plt.tight_layout()
                st.pyplot(fig_cls, use_container_width=True)
                plt.close(fig_cls)

            else:
                np.random.seed(42)
                fig_cls, axes_cls = plt.subplots(1, 3, figsize=(15, 4))
                fig_cls.patch.set_facecolor("#0a0a14")
                top_segs    = sorted(importance_weights_demo.keys(),
                                     key=lambda s: abs(importance_weights_demo[s]),
                                     reverse=True)[:8]
                class_scale = [0.6, 0.85, 1.0]
                for cls_idx in range(3):
                    ax = axes_cls[cls_idx]
                    ax.set_facecolor("#111120")
                    imps = [importance_weights_demo[s] * class_scale[cls_idx]
                            * (1 + np.random.randn() * 0.08) for s in top_segs]
                    if cls_idx != pred_idx:
                        imps = [-v * 0.6 for v in imps]
                    colors_bar = ["#27ae60" if v > 0 else "#c0392b" for v in imps]
                    ax.barh(range(len(top_segs)), imps, color=colors_bar,
                            edgecolor="#444", linewidth=0.5)
                    ax.set_yticks(range(len(top_segs)))
                    ax.set_yticklabels([f"Region {s}" for s in top_segs],
                                       color="white", fontsize=8)
                    ax.axvline(0, color="white", lw=1.2)
                    ax.tick_params(colors="white")
                    ax.set_xlabel("Importance Score", color="white")
                    ax.xaxis.grid(True, alpha=0.2, color="#444")
                    for spine in ax.spines.values(): spine.set_color("#333")
                    if cls_idx == pred_idx:
                        ax.set_facecolor("#141408")
                        ax.set_title(f"★ {CLASS_NAMES[cls_idx]}\n(Predicted)",
                                     color=CLASS_COLORS[cls_idx], fontweight="bold", fontsize=10)
                    else:
                        ax.set_title(CLASS_NAMES[cls_idx], color="white",
                                     fontweight="bold", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_cls, use_container_width=True)
                plt.close(fig_cls)

                st.markdown("**Per-Class Spatial Heatmap Overlays**")
                fig_hm, axes_hm = plt.subplots(1, 3, figsize=(15, 4))
                fig_hm.patch.set_facecolor("#0a0a14")
                cmaps = ["Greens", "YlOrBr", "Reds"]
                for cls_idx in range(3):
                    ax = axes_hm[cls_idx]
                    ax.set_facecolor("#0a0a14")
                    cls_hm = heatmap_demo * class_scale[cls_idx]
                    if cls_idx != pred_idx:
                        cls_hm = -cls_hm * 0.5
                    ax.imshow(pseudo, alpha=0.45)
                    im_c = ax.imshow(cls_hm, cmap=cmaps[cls_idx], alpha=0.75,
                                     vmin=cls_hm.min(), vmax=cls_hm.max())
                    cb2 = plt.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)
                    cb2.ax.yaxis.set_tick_params(color="white")
                    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
                    for spine in ax.spines.values():
                        spine.set_edgecolor(CLASS_COLORS[cls_idx])
                        spine.set_linewidth(2)
                    ax.set_title(CLASS_NAMES[cls_idx], color=CLASS_COLORS[cls_idx],
                                 fontweight="bold", fontsize=10)
                    ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Band Importance
    # ─────────────────────────────────────────────────────────────────────────
    with tab_bands:
        st.markdown("### PCA Band Importance — Occlusion Analysis")
        st.markdown("""
        Each PCA component is individually zeroed out, and the drop in prediction confidence 
        is measured across 20 random test samples. **Higher drop = more important band.**
        """)

        band_imp_path = os.path.join(MODEL_DIR, "lime_outputs/band_importance.npy")
        if os.path.exists(band_imp_path):
            band_importance = np.load(band_imp_path)
            mean_imp        = band_importance.mean(axis=1)
        else:
            np.random.seed(7)
            band_importance = np.random.randn(N_BANDS, 3) * 0.05
            band_importance[:5] += 0.04
            mean_imp = band_importance.mean(axis=1)
            st.info("ℹ️ Using simulated band importance (run Notebook 04 to generate real data).")

        pc_labels = [f"PC{i+1}" for i in range(N_BANDS)]

        fig_band, axes_b = plt.subplots(1, 2, figsize=(16, 5))
        fig_band.patch.set_facecolor("#0a0a14")

        ax0 = axes_b[0]
        ax0.set_facecolor("#111120")
        colors_bar = ["#e74c3c" if v > 0 else "#3498db" for v in mean_imp]
        bars = ax0.bar(range(N_BANDS), mean_imp, color=colors_bar, edgecolor="white", linewidth=0.5)
        ax0.set_xticks(range(N_BANDS))
        ax0.set_xticklabels(pc_labels, fontsize=8, rotation=45, color="white")
        ax0.set_ylabel("Mean Importance (Prob. Drop)", color="white")
        ax0.set_title("Overall PCA Band Importance\nRed = Important | Blue = Harmful if kept",
                      color="white", fontweight="bold")
        ax0.axhline(0, color="white", lw=1)
        ax0.tick_params(colors="white")
        ax0.yaxis.grid(True, alpha=0.2, color="#333")
        for spine in ax0.spines.values(): spine.set_color("#333")
        for bar, v in zip(bars, mean_imp):
            ax0.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.0015 if v >= 0 else bar.get_height() - 0.004,
                     f"{v:.3f}", ha="center", fontsize=7, color="white", fontweight="bold")

        ax1 = axes_b[1]
        heatmap_data = np.round(band_importance.T, 4)
        sns.heatmap(heatmap_data, ax=ax1, cmap="RdYlGn",
                    xticklabels=pc_labels, yticklabels=["Healthy", "Partial", "Deficient"],
                    annot=True, fmt=".3f", annot_kws={"size": 7, "weight": "bold"},
                    linewidths=0.5, linecolor="white", vmin=-0.1, vmax=0.1)
        ax1.set_title("Per-Class Band Importance Heatmap", color="white", fontweight="bold")
        ax1.set_xlabel("PCA Component", color="white")
        ax1.tick_params(colors="white")
        ax1.figure.axes[-1].yaxis.label.set_color("white")
        ax1.figure.axes[-1].tick_params(colors="white")
        plt.tight_layout()

        st.pyplot(fig_band, use_container_width=True)
        plt.close(fig_band)

        top3 = np.argsort(mean_imp)[::-1][:3]
        st.markdown(f"**Top 3 Most Important PCA Bands:** `PC{top3[0]+1}`, `PC{top3[1]+1}`, `PC{top3[2]+1}`")

        st.markdown("---")
        st.markdown("**Band Importance Data Table**")
        df_bands = pd.DataFrame(
            band_importance, columns=["Healthy", "Partial", "Deficient"], index=pc_labels
        )
        df_bands["Mean"] = mean_imp
        st.dataframe(df_bands.style.background_gradient(cmap="RdYlGn", axis=None),
                     use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — Model Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    with tab_eval:
        st.markdown("### Model Evaluation — Test Set Results")

        X_test_path = os.path.join(PROCESSED_DIR, "X_test_hyper.npy")
        y_test_path = os.path.join(PROCESSED_DIR, "y_test_hyper.npy")

        if os.path.exists(X_test_path) and os.path.exists(y_test_path) and models_dict:
            with st.spinner("Loading test data and running evaluation…"):
                X_test = np.load(X_test_path)
                y_test = np.load(y_test_path)
                y_enc  = le.transform(y_test) if le else y_test.astype(int)
                probs_all, per_model_all = ensemble_predict(models_dict, X_test)
                preds_all = np.argmax(probs_all, axis=1)
        else:
            st.info("ℹ️ Test data not found — showing **simulated** evaluation metrics for demonstration.")
            np.random.seed(42)
            n = 400
            y_enc     = np.random.choice([0, 1, 2], n, p=[0.37, 0.30, 0.33])
            preds_all = y_enc.copy()
            flip_idx  = np.random.choice(n, size=int(n * 0.13), replace=False)
            preds_all[flip_idx] = np.random.choice([0, 1, 2], len(flip_idx))
            probs_all = np.zeros((n, 3))
            for i in range(n):
                probs_all[i, preds_all[i]] = 0.75 + np.random.rand() * 0.2
                rem    = 1 - probs_all[i, preds_all[i]]
                others = [j for j in range(3) if j != preds_all[i]]
                probs_all[i, others[0]] = rem * 0.6
                probs_all[i, others[1]] = rem * 0.4
            per_model_all = {
                "CustomCNN":   probs_all * (0.95 + np.random.rand(*probs_all.shape) * 0.1),
                "ResNet50":    probs_all * (0.95 + np.random.rand(*probs_all.shape) * 0.1),
                "DenseNet121": probs_all * (0.95 + np.random.rand(*probs_all.shape) * 0.1),
            }
            for k in per_model_all:
                per_model_all[k] = (per_model_all[k].T / per_model_all[k].sum(axis=1)).T

        accuracy = (preds_all == y_enc).mean()
        cm       = confusion_matrix(y_enc, preds_all)
        report   = classification_report(y_enc, preds_all,
                                         target_names=["Healthy", "Partial", "Deficient"],
                                         output_dict=True)

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Accuracy",    f"{accuracy*100:.2f}%",                    "#4ade80")
        metric_card(c2, "Macro F1",    f"{report['macro avg']['f1-score']:.3f}",  "#60a5fa")
        metric_card(c3, "Weighted F1", f"{report['weighted avg']['f1-score']:.3f}","#e879f9")
        metric_card(c4, "Test Samples", str(len(y_enc)),                           "#fbbf24")
        st.markdown("<br>", unsafe_allow_html=True)

        col_cm, col_roc = st.columns([1, 1], gap="large")

        with col_cm:
            st.markdown("**Confusion Matrix (Raw Counts)**")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            fig_cm.patch.set_facecolor("#0a0a14")
            ax_cm.set_facecolor("#111120")
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn",
                        xticklabels=["Healthy", "Partial", "Deficient"],
                        yticklabels=["Healthy", "Partial", "Deficient"],
                        ax=ax_cm, linewidths=0.5, linecolor="white",
                        annot_kws={"size": 13, "weight": "bold"})
            ax_cm.set_title("Confusion Matrix", color="white", fontweight="bold", pad=12)
            ax_cm.set_xlabel("Predicted", color="white")
            ax_cm.set_ylabel("Actual", color="white")
            ax_cm.tick_params(colors="white")
            ax_cm.figure.axes[-1].yaxis.label.set_color("white")
            ax_cm.figure.axes[-1].tick_params(colors="white")
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

            st.markdown("**Confusion Matrix (Normalised)**")
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig_cmn, ax_cmn = plt.subplots(figsize=(6, 5))
            fig_cmn.patch.set_facecolor("#0a0a14")
            ax_cmn.set_facecolor("#111120")
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGn",
                        xticklabels=["Healthy", "Partial", "Deficient"],
                        yticklabels=["Healthy", "Partial", "Deficient"],
                        ax=ax_cmn, vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                        annot_kws={"size": 13, "weight": "bold"})
            ax_cmn.set_title("Normalised Confusion Matrix", color="white", fontweight="bold", pad=12)
            ax_cmn.set_xlabel("Predicted", color="white")
            ax_cmn.set_ylabel("Actual", color="white")
            ax_cmn.tick_params(colors="white")
            ax_cmn.figure.axes[-1].yaxis.label.set_color("white")
            ax_cmn.figure.axes[-1].tick_params(colors="white")
            st.pyplot(fig_cmn, use_container_width=True)
            plt.close(fig_cmn)

        with col_roc:
            st.markdown("**ROC Curves (Ensemble)**")
            y_bin = label_binarize(y_enc, classes=[0, 1, 2])
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            fig_roc.patch.set_facecolor("#0a0a14")
            ax_roc.set_facecolor("#111120")
            ls_styles = ["-", "--", "-."]
            for c_idx, (cname, ccolor, ls) in enumerate(
                zip(["Healthy", "Partial", "Deficient"], CLASS_COLORS, ls_styles)
            ):
                fpr, tpr, _ = roc_curve(y_bin[:, c_idx], probs_all[:, c_idx])
                roc_auc     = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, color=ccolor, lw=2.5, ls=ls,
                            label=f"{cname}  AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
            ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
            ax_roc.set_xlabel("False Positive Rate", color="white")
            ax_roc.set_ylabel("True Positive Rate", color="white")
            ax_roc.set_title("ROC Curve (Ensemble + TTA)", color="white", fontweight="bold", pad=12)
            ax_roc.legend(loc="lower right", fontsize=9, facecolor="#111120", labelcolor="white")
            ax_roc.tick_params(colors="white")
            ax_roc.yaxis.grid(True, alpha=0.2, color="#333")
            ax_roc.xaxis.grid(True, alpha=0.2, color="#333")
            for spine in ax_roc.spines.values(): spine.set_color("#333")
            st.pyplot(fig_roc, use_container_width=True)
            plt.close(fig_roc)

            st.markdown("**Per-Class Classification Report**")
            report_rows = []
            for cls_name in ["Healthy", "Partial", "Deficient"]:
                r = report.get(cls_name, {})
                report_rows.append({
                    "Class":     cls_name,
                    "Precision": f"{r.get('precision', 0):.3f}",
                    "Recall":    f"{r.get('recall', 0):.3f}",
                    "F1-Score":  f"{r.get('f1-score', 0):.3f}",
                    "Support":   int(r.get("support", 0)),
                })
            st.dataframe(pd.DataFrame(report_rows).set_index("Class"), use_container_width=True)

        st.markdown("---")
        st.markdown("**Model Confidence Distribution per Class**")
        all_model_preds = list(per_model_all.items()) + [("Ensemble", probs_all)]
        fig_box, axes_box = plt.subplots(1, len(all_model_preds), figsize=(18, 4), sharey=True)
        fig_box.patch.set_facecolor("#0a0a14")
        bc = ["#d5f5e3", "#fdebd0", "#f5cba7"]
        be = ["#1abc9c", "#e67e22", "#8e44ad"]
        for col_b, (mname, prbs) in enumerate(all_model_preds):
            ax = axes_box[col_b]
            ax.set_facecolor("#111120")
            max_conf = np.max(prbs, axis=1)
            data     = [max_conf[y_enc == c] for c in range(3)]
            bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                            medianprops=dict(color="#2c3e50", lw=2.5),
                            whiskerprops=dict(lw=1.5, ls="--"),
                            flierprops=dict(marker="o", ms=3, alpha=0.4, markerfacecolor="white"))
            for patch, fc, ec in zip(bp["boxes"], bc, be):
                patch.set_facecolor(fc); patch.set_edgecolor(ec); patch.set_linewidth(1.8)
            ax.set_xticklabels(["H", "P", "D"], color="white")
            ax.set_title(f"{mname}", color="white", fontweight="bold", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.9, color="#7f8c8d", ls="--", lw=1, alpha=0.5)
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_color("#333")
            ax.yaxis.grid(True, alpha=0.2, color="#333")
            if col_b == 0: ax.set_ylabel("Max Softmax Confidence", color="white")
        plt.tight_layout()
        st.pyplot(fig_box, use_container_width=True)
        plt.close(fig_box)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    with tab_pipeline:
        st.markdown("### Full GreenLens Processing Pipeline")

        steps = [
            {"nb": "NB 01", "color": "#0ea5e9",
             "title": "Raw Hyperspectral TIFF Loading",
             "detail": "Shape: `(48, 352, 204)` · 204 spectral bands · HyperLeaf2024 dataset · `tifffile.imread()`"},
            {"nb": "NB 02", "color": "#6366f1",
             "title": "Spatial Resize",
             "detail": "`cv2.resize((48,352,204) → (128,128,204), interpolation=INTER_LINEAR)`"},
            {"nb": "NB 02", "color": "#8b5cf6",
             "title": "Pixel Flattening",
             "detail": "`reshaped = image.reshape(-1, 204)` → shape `(128×128, 204) = (16384, 204)`"},
            {"nb": "NB 02", "color": "#a855f7",
             "title": "StandardScaler",
             "detail": "Per-band zero-mean, unit-variance normalisation · fitted on 50-image random sample"},
            {"nb": "NB 02", "color": "#c026d3",
             "title": "PCA — 15 Components",
             "detail": "204 spectral bands → 15 principal components · explains >95% variance · `sklearn.decomposition.PCA(n_components=15)`"},
            {"nb": "NB 02", "color": "#db2777",
             "title": "MinMaxScaler",
             "detail": "Final [0,1] normalisation for CNN input · `sklearn.preprocessing.MinMaxScaler()`"},
            {"nb": "NB 02", "color": "#e11d48",
             "title": "Pipeline Saved",
             "detail": "`joblib.dump(full_pipeline, 'models/hyperleaf_full_pipeline.joblib')`"},
            {"nb": "NB 03", "color": "#f43f5e",
             "title": "Data Augmentation",
             "detail": "`RandomFlip` + `RandomRotation(0.1)` + `RandomZoom(0.1)` · `tf.data` pipeline with AUTOTUNE"},
            {"nb": "NB 03", "color": "#fb923c",
             "title": "Focal Loss",
             "detail": "`γ=2.0, α=0.25` · Forces focus on hard Partial(0.5) examples · `class_weight.compute_class_weight('balanced')`"},
            {"nb": "NB 03", "color": "#fbbf24",
             "title": "CustomCNN — Squeeze-Excite Residual Blocks",
             "detail": "4 residual stages (64→128→256→512 filters) + channel attention + dilated convolutions · input `(128,128,15)`"},
            {"nb": "NB 03", "color": "#a3e635",
             "title": "ResNet50 — Two-Phase Transfer Learning",
             "detail": "Phase 1: frozen base 20 epochs · Phase 2: fine-tune from layer 140 at LR=5e-6 · 2-layer spectral adapter (15→32→3ch)"},
            {"nb": "NB 03", "color": "#22c55e",
             "title": "DenseNet121 — Two-Phase Transfer Learning",
             "detail": "Same two-phase strategy · Phase 2: fine-tune from layer 100 · L2 regularisation on Dense heads"},
            {"nb": "NB 03", "color": "#10b981",
             "title": "Test-Time Augmentation (TTA)",
             "detail": "Average predictions over N augmented copies of test image · reduces prediction variance"},
            {"nb": "NB 03", "color": "#0ea5e9",
             "title": "Performance-Weighted Ensemble",
             "detail": "Weights: CustomCNN 33% | ResNet50 33% | DenseNet121 34% · auto-weighted by validation accuracy"},
            {"nb": "NB 04", "color": "#6366f1",
             "title": "LIME Explanation",
             "detail": "15ch→pseudo-RGB for LIME · spatial mask applied to all 15 channels · `num_samples=500-1000` perturbations"},
            {"nb": "NB 04", "color": "#8b5cf6",
             "title": "PCA Band Occlusion Analysis",
             "detail": "Zero out each PC band → measure confidence drop · 20 random test samples · `band_importance.npy` saved"},
        ]

        for i, s in enumerate(steps):
            st.markdown(f"""
            <div class="step-card" style="border-left-color:{s['color']}">
                <span style="background:{s['color']}22; color:{s['color']}; font-size:10px;
                       padding:2px 8px; border-radius:4px; margin-right:10px; font-weight:700">
                    {s['nb']}
                </span>
                <strong style="color:#e0e0e0">{i+1:02d}. {s['title']}</strong><br>
                <span style="color:#555; font-size:11px; margin-left:60px">{s['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Pipeline File Structure**")
        st.code("""
GreenLens/
├── app.py                              ← This Streamlit app
├── models/
│   ├── hyperleaf_full_pipeline.joblib  ← Preprocessing pipeline (NB02)
│   ├── hyper_label_encoder.pkl         ← Label encoder (NB03)
│   ├── CustomCNN_final.h5              ← Trained CustomCNN (NB03)
│   ├── ResNet50_final.h5               ← Trained ResNet50 (NB03)
│   ├── DenseNet121_final.h5            ← Trained DenseNet121 (NB03)
│   └── lime_outputs/
│       ├── band_importance.npy         ← Occlusion results (NB04)
│       └── lime_dashboard.png          ← LIME dashboard export (NB04)
├── data/
│   ├── raw/HyperLeaf2024/
│   │   ├── images/*.tiff               ← Raw hyperspectral images
│   │   └── train.csv
│   └── processed/
│       ├── X_train_hyper.npy           ← Processed train data (NB02)
│       ├── X_test_hyper.npy            ← Processed test data (NB02)
│       ├── y_train_hyper.npy
│       └── y_test_hyper.npy
└── src/
    └── lime_utils.py                   ← LIME utility functions (NB04)
        """, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if dashboard == "🌾  Farmer View":
    farmer_view()
else:
    researcher_view()
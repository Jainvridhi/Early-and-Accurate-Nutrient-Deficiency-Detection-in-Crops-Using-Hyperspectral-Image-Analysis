import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io

# --- 1. IMPORT YOUR PREPROCESSING FUNCTIONS ---
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_rgb_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_hyper_preprocess

# --- 2. PLANT INFORMATION DATABASE ---
# (This is the same as before, no changes)
PLANT_INFO_DB = {
    "healthy": {
        "name": "Healthy",
        "description": "The plant appears to be in good health. No signs of disease are visible.",
        "recommendation": "Continue with your standard watering, fertilizing, and care schedule.",
        "prevention": "Ensure good airflow around plants, avoid overwatering, and continue to monitor for any changes."
    },
    "Bacterial_spot": {
        "name": "Bacterial Spot",
        "description": "This disease is caused by bacteria and appears as small, water-soaked, or black spots on leaves and fruits.",
        "recommendation": "Remove and destroy infected plant parts. Apply a copper-based bactericide as a spray. Avoid working with plants when they are wet.",
        "prevention": "Rotate crops, buy disease-free seeds, and avoid overhead watering as the bacteria spread via water splash."
    },
    "Late_blight": {
        "name": "Late Blight",
        "description": "A serious fungal disease, often seen on tomatoes and potatoes. It appears as large, dark, water-soaked blotches on leaves and stems.",
        "recommendation": "Immediately remove and burn infected plants. Apply a targeted fungicide (e.g., containing chlorothalonil or mancozeb) to all nearby plants.",
        "prevention": "Ensure proper spacing for airflow, water at the base of the plant, and avoid planting in the same spot where diseased plants were."
    },
    "default": {
        "name": "Unknown",
        "description": "The disease could not be confidently identified or is not in our database.",
        "recommendation": "Isolate the plant to prevent potential spread. Bring a sample to a local agricultural extension office for analysis.",
        "prevention": "Practice good general garden hygiene, including cleaning tools and removing plant debris."
    }
}

# --- 3. LOAD MODELS AND ENCODERS ---
@st.cache_resource
def load_all_models():
    """Loads all models and encoders into memory."""
    print("Loading models and encoders...")
    rgb_model = tf.keras.models.load_model(
        "../models/plant_disease_rgb_model.h5",
        custom_objects={'preprocess_input': resnet_rgb_preprocess},
        compile=False
    )
    hyper_model = tf.keras.models.load_model(
        "../models/nutrient_hyper_model.h5",
        custom_objects={'preprocess_input': resnet_hyper_preprocess},
        compile=False
    )
    with open('../models/label_encoder.pkl', 'rb') as f:
        rgb_le = pickle.load(f)
    with open('../models/hyper_label_encoder.pkl', 'rb') as f:
        hyper_le = pickle.load(f)
    print("All models loaded.")
    return rgb_model, hyper_model, rgb_le, hyper_le

# --- 4. PREDICTION FUNCTIONS ---

def preprocess_rgb_image(image_pil):
    """Converts a PIL Image to the format for the RGB model."""
    image = np.array(image_pil)
    if image.shape[-1] == 4: # Handle RGBA
        image = image[..., :3]
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Rescale to [0, 1]
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image

def get_rgb_prediction(model, le, image_pil):
    """Gets a prediction from the RGB model."""
    processed_image = preprocess_rgb_image(image_pil)
    pred_probs = model.predict(processed_image)
    pred_class_index = np.argmax(pred_probs[0])
    pred_class_name = le.inverse_transform([pred_class_index])[0]
    confidence = np.max(pred_probs[0])
    
    try:
        plant, status = pred_class_name.split('__')
        status = status.replace('_', ' ')
        friendly_name = f"{plant} ({status})"
    except:
        friendly_name = pred_class_name
        status = pred_class_name
        
    return friendly_name, status, confidence

def get_plant_info(status):
    """Gets the detailed info for a given status."""
    key = status.replace(' ', '_')
    return PLANT_INFO_DB.get(key, PLANT_INFO_DB['default'])

def get_hyper_prediction(model, le, hyper_image_batch):
    """Gets a prediction from a pre-processed hyperspectral image."""
    pred_probs = model.predict(hyper_image_batch)
    pred_class_index = np.argmax(pred_probs[0])
    pred_class_name = le.classes_[pred_class_index] # e.g., 0.0, 0.5, 1.0
    confidence = np.max(pred_probs[0])
    return pred_class_name, confidence

# --- THIS IS THE NEW FUNCTION ---
@st.cache_data
def get_sample_hyper_prediction(_model, _le):
    """
    Runs the hyperspectral model on a single sample test image.
    This simulates the hyperspectral analysis.
    """
    print("Running sample hyperspectral analysis...")
    # Load a sample image from our test set
    sample_image = np.load('../data/processed/X_test_hyper.npy')[5] # Image index 5
    sample_image_batch = np.expand_dims(sample_image, axis=0)
    
    # Get prediction
    pred_class_name, confidence = get_hyper_prediction(_model, _le, sample_image_batch)
    
    # Format the name
    if pred_class_name == 0.0:
        nutrient_status = "Deficient (0.0)"
    elif pred_class_name == 0.5:
        nutrient_status = "Partial (0.5)"
    else:
        nutrient_status = "Sufficient (1.0)"
        
    return nutrient_status, confidence

# --- 5. EXPLAINABLE AI (LIME) FUNCTION ---
# (This section is the same as before)
@st.cache_resource
def build_clean_model_for_lime():
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False 
    inputs = tf.keras.layers.Input(shape=(128, 128, 3), name="clean_input")
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.layers.Lambda(resnet_hyper_preprocess)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    clean_model = tf.keras.models.Model(inputs, outputs)
    clean_model.load_weights('../models/nutrient_hyper_model.h5')
    return clean_model

lime_model = build_clean_model_for_lime()

def lime_predict_fn(images):
    return lime_model.predict(images)

def get_lime_explanation(le, image_pil):
    st.warning("LIME is running on a sample image. Real-time PCA processing is not yet implemented.")
    sample_image = np.load('../data/processed/X_test_hyper.npy')[5] 
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        sample_image, lime_predict_fn, top_labels=3, hide_color=0, num_samples=1000
    )
    preds = lime_predict_fn(np.expand_dims(sample_image, axis=0))
    top_pred_index = np.argmax(preds[0])
    temp, mask = explanation.get_image_and_mask(
        top_pred_index, positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = mark_boundaries(temp, mask)
    return lime_img, le.classes_[top_pred_index]
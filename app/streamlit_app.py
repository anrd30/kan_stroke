"""
Streamlit App for KNSD Kannada Numeral Classifier
"""
import streamlit as st
import os
import sys
import cv2
import numpy as np
import joblib
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import preprocess_image, extract_knsd_features, extract_loop_features, extract_endpoint_features, extract_junction_features, IMAGE_SIZE

KANNADA_NUMERALS = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯']

st.set_page_config(page_title="KNSD Classifier", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #11998e; text-align: center; }
    .kannada-digit { font-size: 4rem; }
    .feature-box { background: #f0f0f0; padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'knsd_model.pkl')
    if not os.path.exists(model_path):
        return None, None
    data = joblib.load(model_path)
    return data['model'], data['scaler']


def preprocess_uploaded(img):
    """
    Universal preprocessing - handles ANY color combination.
    Always outputs: white strokes (255) on black background (0)
    """
    # Step 1: Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Step 2: Apply adaptive thresholding (works for any lighting/color)
    # This finds local contrast, so it works regardless of absolute colors
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # BINARY_INV: dark areas become white
        blockSize=21,
        C=10
    )
    
    # Step 3: Determine if we need to invert
    # Count pixels in border region (likely background)
    h, w = binary.shape
    border_pixels = np.concatenate([
        binary[0, :],           # Top row
        binary[-1, :],          # Bottom row
        binary[:, 0],           # Left column
        binary[:, -1]           # Right column
    ])
    border_mean = np.mean(border_pixels)
    
    # If border is mostly white (>127), strokes are inverted - flip it
    if border_mean > 127:
        binary = 255 - binary
    
    # Step 4: Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Step 5: Resize to standard size
    resized = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    
    # Step 6: Final threshold to ensure clean binary
    _, final = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    return final


def predict(image, model, scaler):
    processed = preprocess_uploaded(image)
    features = extract_knsd_features(processed)
    features = np.nan_to_num(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    
    loop_feats = extract_loop_features(processed)
    endpoint_feats = extract_endpoint_features(processed)
    junction_feats = extract_junction_features(processed)
    
    return pred, proba[pred] * 100, processed, {
        'loops': int(loop_feats[0]),
        'endpoints': int(endpoint_feats[0]),
        'junctions': int(junction_feats[0])
    }


def main():
    st.markdown('<p class="main-header">🔬 KNSD Kannada Numeral Classifier</p>', unsafe_allow_html=True)
    
    model, scaler = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Run `python experiments/train_evaluate.py` first.")
        return
    
    st.success("✅ Model loaded (29 KNSD features)")
    
    with st.sidebar:
        st.header("📊 KNSD Features")
        st.markdown("""
        - 🔵 **Loop count** (8 features)
        - 🔴 **Endpoints** (5 features)
        - 🟡 **Junctions** (3 features)
        - 📐 **Curvature** (8 features)
        - 📏 **Stroke stats** (5 features)
        """)
    
    uploaded = st.file_uploader("Upload Kannada numeral image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            pred, conf, processed, feats = predict(image, model, scaler)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Input")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Processed")
                st.image(processed, use_container_width=True)
            
            with col3:
                st.subheader("Prediction")
                st.markdown(f'<p class="kannada-digit" style="text-align:center">{KANNADA_NUMERALS[pred]}</p>', unsafe_allow_html=True)
                st.markdown(f"**Digit: {pred}** | **Confidence: {conf:.1f}%**")
            
            st.subheader("🔬 Extracted Features")
            c1, c2, c3 = st.columns(3)
            c1.metric("🔵 Loops", feats['loops'])
            c2.metric("🔴 Endpoints", feats['endpoints'])
            c3.metric("🟡 Junctions", feats['junctions'])


if __name__ == '__main__':
    main()

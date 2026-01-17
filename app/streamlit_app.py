"""
Streamlit App for KNSD Kannada Numeral Classifier
With Educational Learn Mode
"""
import streamlit as st
import os
import sys
import cv2
import numpy as np
import joblib
import random
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import (preprocess_image, extract_knsd_features, 
                           extract_loop_features, extract_endpoint_features, 
                           extract_junction_features, extract_stroke_statistics,
                           IMAGE_SIZE)
from reference_features import compute_quality_score, get_digit_description

KANNADA_NUMERALS = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯']

st.set_page_config(page_title="KNSD Learn", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #11998e; text-align: center; }
    .kannada-digit { font-size: 5rem; text-align: center; }
    .target-digit { font-size: 6rem; text-align: center; color: #3498db; }
    .score-display { font-size: 3rem; font-weight: bold; text-align: center; }
    .feedback-box { background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .correct { color: #27ae60; }
    .incorrect { color: #e74c3c; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: bold; }
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
    """Universal preprocessing - handles ANY color combination."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=21, C=10
    )
    
    h, w = binary.shape
    border_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border_pixels) > 127:
        binary = 255 - binary
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    resized = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    _, final = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    return final


def preprocess_canvas(canvas_data):
    """Process canvas drawing to binary image."""
    if canvas_data is None:
        return None
    
    # Convert RGBA to grayscale
    if len(canvas_data.shape) == 3 and canvas_data.shape[2] == 4:
        # Use alpha channel as the drawing (white background, black strokes)
        alpha = canvas_data[:, :, 3]
        gray = alpha.astype(np.uint8)
    else:
        gray = cv2.cvtColor(canvas_data, cv2.COLOR_BGR2GRAY)
    
    # Check if there's any drawing
    if np.max(gray) < 10:
        return None
    
    # Threshold
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Resize to standard size
    resized = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    
    return resized


def predict(image, model, scaler):
    """Predict digit and extract features."""
    processed = preprocess_uploaded(image)
    features = extract_knsd_features(processed)
    features = np.nan_to_num(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    
    loop_feats = extract_loop_features(processed)
    endpoint_feats = extract_endpoint_features(processed)
    junction_feats = extract_junction_features(processed)
    stroke_feats = extract_stroke_statistics(processed)
    
    return pred, proba[pred] * 100, processed, {
        'loops': int(loop_feats[0]),
        'endpoints': int(endpoint_feats[0]),
        'junctions': int(junction_feats[0]),
        'aspect_ratio': float(stroke_feats[0]),
        'circularity': float(stroke_feats[4])
    }


def predict_canvas(canvas_img, model, scaler):
    """Predict from canvas drawing."""
    processed = preprocess_canvas(canvas_img)
    if processed is None:
        return None, 0, None, {}
    
    features = extract_knsd_features(processed)
    features = np.nan_to_num(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    
    loop_feats = extract_loop_features(processed)
    endpoint_feats = extract_endpoint_features(processed)
    junction_feats = extract_junction_features(processed)
    stroke_feats = extract_stroke_statistics(processed)
    
    return pred, proba[pred] * 100, processed, {
        'loops': int(loop_feats[0]),
        'endpoints': int(endpoint_feats[0]),
        'junctions': int(junction_feats[0]),
        'aspect_ratio': float(stroke_feats[0]),
        'circularity': float(stroke_feats[4])
    }


def classify_mode(model, scaler):
    """Original classification mode with file upload."""
    st.header("📤 Upload & Classify")
    
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
                st.markdown(f'<p class="kannada-digit">{KANNADA_NUMERALS[pred]}</p>', unsafe_allow_html=True)
                st.markdown(f"**Digit: {pred}** | **Confidence: {conf:.1f}%**")
            
            st.subheader("🔬 Extracted Features")
            c1, c2, c3 = st.columns(3)
            c1.metric("🔵 Loops", feats['loops'])
            c2.metric("🔴 Endpoints", feats['endpoints'])
            c3.metric("🟡 Junctions", feats['junctions'])


def learn_mode(model, scaler):
    """Educational mode with drawing canvas and feedback."""
    st.header("📚 Learn Kannada Numerals")
    st.markdown("Practice writing Kannada numerals and get instant feedback!")
    
    # Initialize session state
    if 'target_digit' not in st.session_state:
        st.session_state.target_digit = random.randint(0, 9)
    if 'correct_count' not in st.session_state:
        st.session_state.correct_count = 0
    if 'total_attempts' not in st.session_state:
        st.session_state.total_attempts = 0
    
    # Sidebar stats
    with st.sidebar:
        st.header("📊 Your Progress")
        st.metric("Correct", st.session_state.correct_count)
        st.metric("Attempts", st.session_state.total_attempts)
        if st.session_state.total_attempts > 0:
            accuracy = (st.session_state.correct_count / st.session_state.total_attempts) * 100
            st.metric("Accuracy", f"{accuracy:.0f}%")
        
        st.divider()
        if st.button("🔄 Reset Progress"):
            st.session_state.correct_count = 0
            st.session_state.total_attempts = 0
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Draw this digit:")
        target = st.session_state.target_digit
        st.markdown(f'<p class="target-digit">{KANNADA_NUMERALS[target]}</p>', unsafe_allow_html=True)
        st.markdown(f"**{get_digit_description(target)}**")
        
        if st.button("🔀 New Digit", use_container_width=True):
            st.session_state.target_digit = random.randint(0, 9)
            st.rerun()
    
    with col2:
        st.subheader("✏️ Your Drawing:")
        
        # Import canvas component
        try:
            from streamlit_drawable_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=15,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        except ImportError:
            st.error("📦 Please install: `pip install streamlit-drawable-canvas`")
            st.code("pip install streamlit-drawable-canvas")
            return
    
    # Check button
    if st.button("✅ Check My Drawing", use_container_width=True, type="primary"):
        if canvas_result.image_data is not None:
            pred, conf, processed, feats = predict_canvas(canvas_result.image_data, model, scaler)
            
            if pred is not None:
                st.session_state.total_attempts += 1
                target = st.session_state.target_digit
                
                st.divider()
                
                # Result columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.subheader("Processed")
                    if processed is not None:
                        st.image(processed, use_container_width=True)
                
                with res_col2:
                    st.subheader("Detected")
                    st.markdown(f'<p class="kannada-digit">{KANNADA_NUMERALS[pred]}</p>', unsafe_allow_html=True)
                    st.markdown(f"**Confidence: {conf:.1f}%**")
                
                with res_col3:
                    st.subheader("Result")
                    if pred == target:
                        st.session_state.correct_count += 1
                        st.markdown('<p class="score-display correct">✓ Correct!</p>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f'<p class="score-display incorrect">✗ Try Again</p>', unsafe_allow_html=True)
                        st.markdown(f"You drew **{pred}**, but target was **{target}**")
                
                # Quality feedback
                st.subheader("📝 Feedback")
                score, feedback = compute_quality_score(target, feats)
                
                # Score meter
                score_color = "#27ae60" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
                st.progress(score / 100)
                st.markdown(f"**Quality Score: {score}/100**")
                
                # Feedback messages
                for msg in feedback:
                    st.markdown(f'<div class="feedback-box">{msg}</div>', unsafe_allow_html=True)
                
                # Feature breakdown
                with st.expander("🔬 Feature Details"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Loops", feats['loops'])
                    c2.metric("Endpoints", feats['endpoints'])
                    c3.metric("Junctions", feats['junctions'])
                    c4.metric("Circularity", f"{feats['circularity']:.2f}")
            else:
                st.warning("Please draw something first!")
        else:
            st.warning("Please draw something first!")


def main():
    st.markdown('<p class="main-header">📚 KNSD Learn: Kannada Numerals</p>', unsafe_allow_html=True)
    
    model, scaler = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Run `python experiments/train_evaluate.py` first.")
        return
    
    # Mode selection via tabs
    tab1, tab2 = st.tabs(["📚 Learn Mode", "📤 Classify Mode"])
    
    with tab1:
        learn_mode(model, scaler)
    
    with tab2:
        classify_mode(model, scaler)


if __name__ == '__main__':
    main()

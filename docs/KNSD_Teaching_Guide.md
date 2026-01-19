# KNSD Complete Tutorial: Teaching Guide

A comprehensive line-by-line guide to understand and teach the KNSD (Kannada Numeral Shape Descriptor) project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Understanding the Problem](#2-understanding-the-problem)
3. [KNSD Feature Extraction (Core Innovation)](#3-knsd-feature-extraction)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Training the Model](#5-training-the-model)
6. [The Streamlit Application](#6-the-streamlit-application)
7. [Educational Mode (Quality Feedback)](#7-educational-mode)
8. [How Everything Connects](#8-how-everything-connects)

---

## 1. Project Overview

### What does this project do?
This project recognizes handwritten Kannada numerals (೦-೯) using a **novel feature extraction method** called KNSD.

### Why is it special?
- **Not a black-box**: Unlike CNN/deep learning, we use 29 interpretable features
- **Loop-based**: Exploits that Kannada digits like ೦,೬,೮,೯ have characteristic loops
- **Educational**: Gives specific feedback, not just "right/wrong"

### Project Structure
```
KNSD_KannadaNumerals/
├── src/
│   └── knsd_features.py      # Core: Feature extraction (THE MAIN INNOVATION)
├── experiments/
│   └── train_evaluate.py     # Training pipeline
├── app/
│   ├── streamlit_app.py      # Web application
│   └── reference_features.py # Quality feedback logic
├── models/
│   └── knsd_model.pkl        # Trained SVM model
├── data/
│   ├── kannada_mnist/        # Kannada-MNIST dataset
│   └── dataset_custom/       # Custom collected images
└── analysis/                 # All graphs and visualizations
```

---

## 2. Understanding the Problem

### The Challenge
Given a handwritten image of a Kannada numeral, classify it as 0-9.

### Kannada Numerals vs English
```
Kannada: ೦  ೧  ೨  ೩  ೪  ೫  ೬  ೭  ೮  ೯
English: 0  1  2  3  4  5  6  7  8  9
```

### Key Observation (Our Innovation)
Kannada numerals have **topological features**:
- ೦ (zero): Has 1 large circular loop
- ೬ (six): Has 1 loop at bottom
- ೮ (eight): Has 2 connected loops
- ೯ (nine): Has 1 loop at top

This is why we focus on **loop detection** as a key feature!

---

## 3. KNSD Feature Extraction

**File:** `src/knsd_features.py`

This is the **core innovation** of the project. We extract 29 features in 5 categories:

### 3.1 Loop Features (8 features) - NOVEL

```python
def extract_loop_features(binary_img):
    """
    Detect closed loops in the character.
    
    How it works:
    1. Find all contours with cv2.findContours(RETR_TREE)
    2. Check hierarchy - loops are contours WITH a parent
    3. Measure area, position, count
    """
    # Find contours with hierarchy (parent-child relationships)
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # A loop is a contour that has a parent (it's inside something)
    for i, h in enumerate(hierarchy[0]):
        if h[3] != -1:  # Has parent = it's a hole/loop
            # This is a loop!
            area = cv2.contourArea(contours[i])
            # ... extract more properties
```

**Features extracted:**
| # | Name | What it measures |
|---|------|------------------|
| 1 | num_loops | Count of closed loops |
| 2 | total_loop_area | Sum of all loop areas |
| 3 | avg_loop_area | Average loop size |
| 4 | loop_area_ratio | Loop area / total stroke area |
| 5 | loop_cx | X position of largest loop |
| 6 | loop_cy | Y position of largest loop |
| 7 | avg_loop_cy | Average Y of all loops |
| 8 | loop_variance | Variation in loop sizes |

### 3.2 Endpoint Features (5 features)

```python
def extract_endpoint_features(binary_img):
    """
    Find stroke termination points.
    
    How it works:
    1. Skeletonize the image (thin to 1-pixel lines)
    2. For each pixel, count neighbors
    3. If exactly 1 neighbor = endpoint
    """
    # Skeletonize: make strokes 1-pixel thin
    skeleton = skeletonize(binary_img > 0)
    
    # Find endpoints: pixels with exactly 1 neighbor
    for y, x in all_skeleton_pixels:
        neighbors = skeleton[y-1:y+2, x-1:x+2].sum() - 1
        if neighbors == 1:  # Only 1 connected pixel
            # This is an endpoint!
```

**Features:** count, avg_x, avg_y, x_spread, y_spread

### 3.3 Junction Features (3 features)

```python
def extract_junction_features(binary_img):
    """
    Find stroke intersections.
    
    How it works:
    1. Skeletonize the image
    2. For each pixel, count neighbors
    3. If 3+ neighbors = junction
    """
    for y, x in all_skeleton_pixels:
        neighbors = skeleton[y-1:y+2, x-1:x+2].sum() - 1
        if neighbors >= 3:  # 3 or more connections
            # This is a junction!
```

**Features:** count, avg_x, avg_y

### 3.4 Curvature Histogram (8 features)

```python
def extract_curvature_histogram(binary_img, n_bins=8):
    """
    Distribution of stroke directions.
    
    How it works:
    1. For each point on skeleton, measure direction
    2. Direction = arctan2(dy, dx)
    3. Create histogram of directions
    """
    for each_point on skeleton:
        dx = next_point.x - prev_point.x
        dy = next_point.y - prev_point.y
        angle = np.arctan2(dy, dx)  # -π to +π
    
    # Create histogram with 8 bins
    hist = np.histogram(angles, bins=8, range=(-π, π))
```

This tells us: "How much of the stroke goes left? right? up? down?"

### 3.5 Stroke Statistics (5 features)

```python
def extract_stroke_statistics(binary_img):
    """
    General shape metrics.
    """
    # Aspect ratio: width / height
    aspect_ratio = w / h
    
    # Solidity: contour area / convex hull area
    hull = cv2.convexHull(contour)
    solidity = contour_area / hull_area
    
    # Extent: contour area / bounding box area
    extent = contour_area / (w * h)
    
    # Circularity: 4π × area / perimeter²
    # Perfect circle = 1.0
    circularity = 4 * π * area / perimeter²
```

---

## 4. Preprocessing Pipeline

**File:** `src/knsd_features.py` → `preprocess_image()`

### Step-by-Step

```python
def preprocess_image(img):
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Adaptive thresholding (handles varying lighting)
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use local mean
        cv2.THRESH_BINARY_INV,           # Invert: strokes=white
        blockSize=21,                     # Local neighborhood
        C=10                              # Threshold offset
    )
    
    # Step 3: Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    # Step 4: Find largest contour (the character)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, ...)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Step 5: Crop, pad to square, resize to 100×100
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped = binary[y:y+h, x:x+w]
    # ... pad to square, resize to 100×100
```

---

## 5. Training the Model

**File:** `experiments/train_evaluate.py`

### Training Pipeline

```python
def main():
    # 1. Load data
    images, labels = load_kannada_mnist(max_samples=600)
    custom_imgs, custom_labels = load_custom_images()
    
    # 2. Augment data (rotation, scaling, shifting)
    augmented = augment_dataset(images, labels, factor=3)
    
    # 3. Extract features
    for each image:
        processed = preprocess_mnist(img)
        features = extract_knsd_features(processed)  # 29 features
    
    # 4. Train/test split
    X_train, X_test = train_test_split(features, test_size=0.2)
    
    # 5. Scale features (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 6. Train SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    # 7. Evaluate with 5-fold cross-validation
    cv_scores = cross_val_score(svm, X, y, cv=5)
```

### Why SVM?

| Classifier | Pros | Cons |
|------------|------|------|
| **SVM (RBF)** | Works great with 29 features, fast | - |
| CNN | Good for images | Needs more data, black-box |
| Random Forest | Interpretable | Lower accuracy here |
| KNN | Simple | Slower at prediction |

We chose SVM because:
- Works well with small feature vectors (29 features)
- RBF kernel handles non-linear boundaries
- Fast training and prediction

---

## 6. The Streamlit Application

**File:** `app/streamlit_app.py`

### Two Modes

1. **Classify Mode**: Upload image → Get prediction
2. **Learn Mode**: Draw on canvas → Get feedback

### Key Code Sections

```python
# Load the trained model
@st.cache_resource
def load_model():
    data = joblib.load('models/knsd_model.pkl')
    return data['model'], data['scaler']

# For file uploads (Classify Mode)
def predict(image, model, scaler):
    processed = preprocess_uploaded(image)
    features = extract_knsd_features(processed)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    confidence = model.predict_proba(features_scaled)
    return prediction, confidence

# For canvas drawing (Learn Mode)
from streamlit_drawable_canvas import st_canvas
canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280, width=280,
    drawing_mode="freedraw"
)
```

---

## 7. Educational Mode (Quality Feedback)

**File:** `app/reference_features.py`

### How Quality Scoring Works

```python
# Each digit has "ideal" characteristics
IDEAL_FEATURES = {
    0: {'num_loops': 1, 'circularity': 0.8, ...},
    6: {'num_loops': 1, 'num_endpoints': 1, ...},
    8: {'num_loops': 2, 'num_endpoints': 0, ...},
    ...
}

def compute_quality_score(target_digit, user_features):
    ideal = IDEAL_FEATURES[target_digit]
    
    # Compare loops
    if ideal['num_loops'] != user_features['loops']:
        feedback.append("❌ Loop count incorrect")
        score -= 20
    
    # Compare endpoints
    if ideal['num_endpoints'] != user_features['endpoints']:
        feedback.append("⚠️ Check stroke endings")
        score -= 10
    
    return score, feedback
```

This is what makes the app **educational** - not just "wrong" but "your loop is missing"!

---

## 8. How Everything Connects

```
┌─────────────────────────────────────────────────────────────┐
│                    USER DRAWS ON CANVAS                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               PREPROCESSING (preprocess_canvas)             │
│  • RGBA → Grayscale                                        │
│  • Invert colors (black strokes → white)                   │
│  • Threshold to binary                                      │
│  • Resize to 100×100                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           FEATURE EXTRACTION (extract_knsd_features)         │
│  • Loop features (8)     → How many loops? Where?           │
│  • Endpoint features (5) → Stroke terminations              │
│  • Junction features (3) → Intersections                    │
│  • Curvature (8)         → Direction distribution           │
│  • Stroke stats (5)      → Shape metrics                    │
│                          = 29 total features                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                CLASSIFICATION (SVM model)                    │
│  • Scale features with StandardScaler                       │
│  • SVM predicts digit (0-9)                                 │
│  • Returns confidence probability                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              QUALITY FEEDBACK (compute_quality_score)        │
│  • Compare features to ideal reference                      │
│  • Generate specific feedback messages                      │
│  • Calculate quality score (0-100)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         DISPLAY                              │
│  • Show prediction: "You drew ೨"                            │
│  • Show confidence: "92%"                                   │
│  • Show feedback: "✅ Loops correct! ⚠️ Practice curves"    │
│  • Show score: "Quality: 85/100"                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `preprocess_image()` | knsd_features.py | Clean input image |
| `extract_loop_features()` | knsd_features.py | **Novel**: Detect loops |
| `extract_endpoint_features()` | knsd_features.py | Find stroke endings |
| `extract_junction_features()` | knsd_features.py | Find intersections |
| `extract_curvature_histogram()` | knsd_features.py | Direction distribution |
| `extract_stroke_statistics()` | knsd_features.py | Shape metrics |
| `extract_knsd_features()` | knsd_features.py | Combine all 29 features |
| `train_evaluate()` | train_evaluate.py | Train SVM model |
| `predict()` | streamlit_app.py | Make prediction |
| `compute_quality_score()` | reference_features.py | Generate feedback |

---

## Teaching Exercises

### Exercise 1: Understand Loop Detection
1. Print `num_loops` for each digit 0-9
2. Which digits have loops? (Answer: 0, 6, 8, 9)
3. Why is this useful for classification?

### Exercise 2: Modify Features
1. Add a new feature: "stroke_length"
2. Update `extract_knsd_features()` to include it
3. Retrain and compare accuracy

### Exercise 3: Test the App
1. Draw the same digit 5 times
2. Note the confidence variation
3. Why does it vary? (Handwriting variation)

---

*End of Tutorial*

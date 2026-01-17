"""
Preprocessing Visualization - Step-by-step preprocessing pipeline
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import IMAGE_SIZE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_evaluate import load_kannada_mnist


def visualize_preprocessing_steps(img):
    """Visualize each preprocessing step."""
    steps = {}
    
    # Step 0: Original image
    steps['0. Original'] = img.copy()
    
    # Step 1: Resize to 100x100
    resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    steps['1. Resized (100×100)'] = resized
    
    # Step 2: Binary thresholding
    _, binary = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    steps['2. Binary Threshold'] = binary
    
    # Step 3: Morphological noise removal
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    steps['3. Morph Opening'] = opened
    
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps['4. Morph Closing'] = closed
    
    # Step 4: Skeletonization (for visualization)
    skeleton = skeletonize(closed > 0).astype(np.uint8) * 255
    steps['5. Skeleton'] = skeleton
    
    return steps


def create_preprocessing_pipeline_figure(images, labels):
    """Create multi-sample preprocessing visualization."""
    fig, axes = plt.subplots(5, 6, figsize=(16, 14))
    
    sample_indices = []
    for digit in [0, 2, 5, 6, 8]:  # Select varied digits
        idx = np.where(labels == digit)[0][0]
        sample_indices.append(idx)
    
    for row, idx in enumerate(sample_indices):
        img = images[idx]
        steps = visualize_preprocessing_steps(img)
        
        for col, (step_name, step_img) in enumerate(steps.items()):
            ax = axes[row, col]
            ax.imshow(step_img, cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(step_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, f'Digit {labels[idx]}', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', rotation=90)
    
    plt.suptitle('KNSD Preprocessing Pipeline: Step-by-Step Visualization', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_augmentation_examples(images, labels):
    """Visualize data augmentation examples."""
    from train_evaluate import augment_image
    
    fig, axes = plt.subplots(3, 9, figsize=(18, 7))
    
    # Select 3 sample images
    sample_digits = [0, 5, 8]
    
    for row, digit in enumerate(sample_digits):
        idx = np.where(labels == digit)[0][0]
        img = images[idx]
        augmented = augment_image(img)
        
        aug_names = ['Original', 'Rot -10°', 'Rot +10°', 'Scale 85%', 'Scale 115%',
                     'Shift L', 'Shift R', 'Shift U', 'Shift D']
        
        for col, (aug_img, name) in enumerate(zip(augmented, aug_names)):
            ax = axes[row, col]
            ax.imshow(aug_img, cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, f'Digit {digit}', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', rotation=90)
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_sample_images_grid(images, labels):
    """Create grid of sample images per digit class."""
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    
    for digit in range(10):
        digit_images = images[labels == digit][:10]
        for i, img in enumerate(digit_images):
            ax = axes[digit, i]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.text(-0.3, 0.5, f'{digit}', transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='center')
    
    plt.suptitle('Sample Images per Kannada Digit Class (10 samples each)', 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def create_feature_extraction_visualization(img):
    """Visualize what each feature extraction step produces."""
    from knsd_features import extract_loop_features, extract_endpoint_features, extract_junction_features
    from skimage.morphology import skeletonize
    
    # Preprocess
    resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Binary Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Skeleton
    skeleton = skeletonize(binary > 0).astype(np.uint8)
    axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title('Skeleton', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Loops detection
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    loop_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
            if h[3] != -1:  # Has parent (is a hole/loop)
                area = cv2.contourArea(cnt)
                if area > 50:
                    cv2.drawContours(loop_img, [cnt], -1, (0, 255, 0), 2)
    axes[0, 2].imshow(loop_img)
    axes[0, 2].set_title('Loops (Green)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Endpoints
    endpoint_img = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2RGB)
    h, w = skeleton.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x]:
                neighbors = skeleton[y-1:y+2, x-1:x+2].sum() - 1
                if neighbors == 1:  # Endpoint
                    cv2.circle(endpoint_img, (x, y), 3, (255, 0, 0), -1)
    axes[0, 3].imshow(endpoint_img)
    axes[0, 3].set_title('Endpoints (Red)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Junctions
    junction_img = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2RGB)
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x]:
                neighbors = skeleton[y-1:y+2, x-1:x+2].sum() - 1
                if neighbors >= 3:  # Junction
                    cv2.circle(junction_img, (x, y), 3, (0, 0, 255), -1)
    axes[1, 0].imshow(junction_img)
    axes[1, 0].set_title('Junctions (Blue)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Contour/Hull
    contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    main_cnt = max(contours, key=cv2.contourArea) if contours else None
    if main_cnt is not None:
        hull = cv2.convexHull(main_cnt)
        cv2.drawContours(contour_img, [main_cnt], -1, (255, 0, 255), 2)
        cv2.drawContours(contour_img, [hull], -1, (0, 255, 255), 2)
    axes[1, 1].imshow(contour_img)
    axes[1, 1].set_title('Contour & Hull', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Bounding box
    bbox_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    if main_cnt is not None:
        x, y, w, h = cv2.boundingRect(main_cnt)
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (255, 165, 0), 2)
    axes[1, 2].imshow(bbox_img)
    axes[1, 2].set_title('Bounding Box', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # All features combined
    combined = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    combined = cv2.addWeighted(combined, 0.5, loop_img, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.7, endpoint_img, 0.3, 0)
    axes[1, 3].imshow(combined)
    axes[1, 3].set_title('Combined', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')
    
    plt.suptitle('Feature Extraction Visualization', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kannada_mnist')
    train_path = os.path.join(data_dir, 'train.csv')
    
    images, labels = load_kannada_mnist(train_path, max_samples_per_class=50)
    
    print("\n1. Creating preprocessing pipeline figure...")
    fig = create_preprocessing_pipeline_figure(images, labels)
    fig.savefig(os.path.join(output_dir, 'preprocessing_steps.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: preprocessing_steps.png")
    
    print("\n2. Creating augmentation examples...")
    fig = create_augmentation_examples(images, labels)
    fig.savefig(os.path.join(output_dir, 'augmentation_examples.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: augmentation_examples.png")
    
    print("\n3. Creating sample images grid...")
    fig = create_sample_images_grid(images, labels)
    fig.savefig(os.path.join(output_dir, 'sample_images_grid.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: sample_images_grid.png")
    
    print("\n4. Creating feature extraction visualization...")
    # Use a digit with loops (like 0, 6, or 8)
    digit_8_idx = np.where(labels == 8)[0][0]
    fig = create_feature_extraction_visualization(images[digit_8_idx])
    fig.savefig(os.path.join(output_dir, 'feature_extraction_steps.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: feature_extraction_steps.png")
    
    print("\nPreprocessing visualization complete!")


if __name__ == '__main__':
    main()

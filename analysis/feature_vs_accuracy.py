"""
Feature vs Accuracy Analysis
Analyze how accuracy changes with number of features
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from tqdm import tqdm
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import FEATURE_NAMES, extract_knsd_features, IMAGE_SIZE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_evaluate import load_kannada_mnist, preprocess_mnist


def extract_features(images, labels):
    """Extract KNSD features from images."""
    print("Extracting KNSD features...")
    features = []
    
    for img in tqdm(images):
        processed = preprocess_mnist(img)
        feat = extract_knsd_features(processed)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
    
    return np.array(features), labels


def create_feature_vs_accuracy_plot(features, labels):
    """Create plot showing accuracy vs number of features."""
    print("\nComputing feature importance...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Compute mutual information for feature ranking
    mi_scores = mutual_info_classif(X_scaled, labels, random_state=42)
    feature_order = np.argsort(mi_scores)[::-1]  # Best first
    
    print("\nEvaluating accuracy with increasing number of features...")
    num_features_range = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 27, 29]
    accuracies = []
    stds = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    for n_feat in tqdm(num_features_range):
        selected = feature_order[:n_feat]
        X_subset = X_scaled[:, selected]
        
        scores = cross_val_score(svm, X_subset, labels, cv=cv, scoring='accuracy')
        accuracies.append(scores.mean() * 100)
        stds.append(scores.std() * 100)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(num_features_range, accuracies, 'o-', linewidth=2.5, markersize=10,
            color='#2ecc71', label='5-Fold CV Accuracy')
    ax.fill_between(num_features_range, 
                    np.array(accuracies) - np.array(stds),
                    np.array(accuracies) + np.array(stds),
                    alpha=0.2, color='#2ecc71')
    
    # Mark best accuracy
    best_idx = np.argmax(accuracies)
    ax.scatter([num_features_range[best_idx]], [accuracies[best_idx]], 
               s=200, c='red', marker='*', zorder=5, label=f'Best: {accuracies[best_idx]:.2f}%')
    
    # Styling
    ax.set_xlabel('Number of Features (ranked by Mutual Information)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('KNSD Feature Count vs Classification Accuracy', fontsize=16, fontweight='bold')
    ax.set_xticks(num_features_range)
    ax.set_xlim(0, 30)
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    # Annotate key points
    ax.annotate(f'All 29 features\n{accuracies[-1]:.1f}%', 
                xy=(29, accuracies[-1]), xytext=(26, accuracies[-1]-8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    return fig, feature_order, mi_scores


def create_feature_ranking_plot(mi_scores):
    """Create bar chart of feature importance ranking."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by importance
    sorted_idx = np.argsort(mi_scores)[::-1]
    sorted_scores = mi_scores[sorted_idx]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    
    # Color by category
    cat_colors = {
        'Loop': '#27ae60',
        'Endpoint': '#3498db',
        'Junction': '#9b59b6',
        'Curvature': '#e74c3c',
        'Stroke': '#f39c12'
    }
    
    def get_category(name):
        if name.startswith('num_loop') or name.startswith('loop') or name.startswith('avg_loop'):
            return 'Loop'
        elif 'endpoint' in name:
            return 'Endpoint'
        elif 'junction' in name:
            return 'Junction'
        elif 'curv' in name:
            return 'Curvature'
        else:
            return 'Stroke'
    
    colors = [cat_colors[get_category(name)] for name in sorted_names]
    
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_title('KNSD Feature Importance Ranking', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add rank numbers
    for i, (score, bar) in enumerate(zip(sorted_scores, bars)):
        ax.text(score + 0.002, i, f'#{i+1}', va='center', fontsize=8, color='gray')
    
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, label=cat) for cat, color in cat_colors.items()]
    ax.legend(handles=legend_handles, loc='lower right', title='Feature Category')
    
    plt.tight_layout()
    return fig


def create_feature_weightage_plot(features, labels):
    """Create feature weightage visualization using SVM coefficients approximation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Use logistic regression to get approximate feature weights
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
    lr.fit(X_scaled, labels)
    
    # Aggregate absolute weights across all classes
    weights = np.abs(lr.coef_).mean(axis=0)
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    
    # Create gradient colors
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.2, len(sorted_names)))
    
    bars = ax.barh(range(len(sorted_names)), sorted_weights * 100, color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel('Feature Weight (%)', fontsize=12)
    ax.set_title('KNSD Feature Weightage (Logistic Regression Coefficients)', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (weight, bar) in enumerate(zip(sorted_weights, bars)):
        ax.text(weight * 100 + 0.1, i, f'{weight*100:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kannada_mnist')
    train_path = os.path.join(data_dir, 'train.csv')
    
    images, labels = load_kannada_mnist(train_path, max_samples_per_class=300)
    features, labels = extract_features(images, labels)
    
    print("\n1. Creating feature vs accuracy plot...")
    fig, feature_order, mi_scores = create_feature_vs_accuracy_plot(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_vs_accuracy.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: feature_vs_accuracy.png")
    
    print("\n2. Creating feature ranking plot...")
    fig = create_feature_ranking_plot(mi_scores)
    fig.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: feature_importance.png")
    
    print("\n3. Creating feature weightage plot...")
    fig = create_feature_weightage_plot(features, labels)
    fig.savefig(os.path.join(output_dir, 'feature_weightage.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: feature_weightage.png")
    
    print("\nFeature vs accuracy analysis complete!")


if __name__ == '__main__':
    main()

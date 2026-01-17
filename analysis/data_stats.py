"""
Data Statistics - Dataset analysis and visualization
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_evaluate import load_kannada_mnist, load_custom_images


def create_data_distribution(mnist_images, mnist_labels, custom_images, custom_labels):
    """Create data distribution chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # MNIST distribution
    ax = axes[0]
    unique, counts = np.unique(mnist_labels, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))
    ax.bar(unique, counts, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Digit Class', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Kannada-MNIST Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, 0.95, f'Total: {len(mnist_labels)} samples', transform=ax.transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Custom dataset distribution
    ax = axes[1]
    unique, counts = np.unique(custom_labels, return_counts=True)
    ax.bar(unique, counts, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Digit Class', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Custom Dataset Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, 0.95, f'Total: {len(custom_labels)} samples', transform=ax.transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Combined pie chart
    ax = axes[2]
    sizes = [len(mnist_labels), len(custom_labels)]
    labels = ['Kannada-MNIST', 'Custom']
    colors_pie = ['#3498db', '#e74c3c']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90, shadow=True,
                                       textprops={'fontsize': 11})
    ax.set_title('Dataset Composition', fontsize=13, fontweight='bold')
    
    plt.suptitle('KNSD Training Data Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_train_test_split_viz():
    """Create train/test split visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample sizes (from train_evaluate.py)
    categories = ['Total', 'Training (80%)', 'Testing (20%)']
    sizes = [100, 80, 20]  # Percentages
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.barh(categories, sizes, color=colors, edgecolor='white', linewidth=2, height=0.5)
    
    ax.set_xlabel('Percentage of Data (%)', fontsize=12)
    ax.set_title('Train/Test Split Strategy', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, size in zip(bars, sizes):
        ax.text(size + 2, bar.get_y() + bar.get_height()/2, f'{size}%',
                va='center', fontsize=12, fontweight='bold')
    
    # Add annotation
    ax.text(50, -0.5, 'Stratified split ensures equal class proportions',
            ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    return fig


def create_data_summary_table():
    """Create a summary table of dataset statistics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    data = [
        ['Kannada-MNIST', '60,000', '10', '28×28', 'Grayscale', 'Train'],
        ['Kannada-MNIST Test', '10,000', '10', '28×28', 'Grayscale', 'Test reference'],
        ['Custom Dataset', '~250', '10', 'Variable', 'Grayscale', 'Train (augmented)'],
        ['After Augmentation', '~14,000', '10', '28×28', 'Grayscale', 'Combined'],
    ]
    
    columns = ['Dataset', 'Samples', 'Classes', 'Size', 'Format', 'Usage']
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                     colColours=['#3498db']*6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors
    for i in range(1, 5):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax.set_title('Dataset Summary Statistics', fontsize=16, fontweight='bold', y=1.1)
    
    plt.tight_layout()
    return fig


def create_image_size_distribution(images):
    """Create histogram of image pixel intensities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pixel intensity distribution overall
    ax = axes[0]
    all_pixels = images.flatten()
    ax.hist(all_pixels, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Pixel Intensity', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Pixel Intensity Distribution', fontsize=13, fontweight='bold')
    ax.axvline(x=all_pixels.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {all_pixels.mean():.1f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mean intensity per digit
    ax = axes[1]
    # This would require labels, but we'll simulate
    ax.text(0.5, 0.5, 'Requires labels\nfor per-digit analysis', transform=ax.transAxes,
            ha='center', va='center', fontsize=12, style='italic', color='gray')
    ax.set_title('Mean Intensity by Digit', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('Image Statistics Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kannada_mnist')
    custom_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_custom')
    train_path = os.path.join(data_dir, 'train.csv')
    
    print("Loading datasets...")
    mnist_images, mnist_labels = load_kannada_mnist(train_path, max_samples_per_class=200)
    custom_images, custom_labels = load_custom_images(custom_dir)
    
    print("\n1. Creating data distribution plot...")
    fig = create_data_distribution(mnist_images, mnist_labels, custom_images, custom_labels)
    fig.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: data_distribution.png")
    
    print("\n2. Creating train/test split visualization...")
    fig = create_train_test_split_viz()
    fig.savefig(os.path.join(output_dir, 'train_test_split.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: train_test_split.png")
    
    print("\n3. Creating data summary table...")
    fig = create_data_summary_table()
    fig.savefig(os.path.join(output_dir, 'data_summary_table.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: data_summary_table.png")
    
    print("\n4. Creating image statistics...")
    fig = create_image_size_distribution(mnist_images)
    fig.savefig(os.path.join(output_dir, 'image_statistics.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Saved: image_statistics.png")
    
    print("\nData statistics complete!")


if __name__ == '__main__':
    main()

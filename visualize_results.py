#!/usr/bin/env python3
"""
Visualize JEPA-EHR representations and downstream task results

Creates visualizations to understand what the model has learned:
1. t-SNE/UMAP of learned representations
2. Training loss curves
3. Downstream task performance comparisons
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from downstream_tasks import RepresentationExtractor, SyntheticTaskDataset
from src.datasets.mimic_ehr import MIMICEHRDataset


def plot_training_curves(log_file, output_dir):
    """Plot training loss curves from CSV log"""
    logger.info("Plotting training curves...")
    
    # Read CSV and filter out duplicate header rows
    df = pd.read_csv(log_file)
    df = df[df['epoch'] != 'epoch']  # Remove duplicate header rows
    df = df.astype({'epoch': int, 'itr': int, 'loss': float, 
                    'mask-context': float, 'mask-pred': float, 'time (ms)': float})
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    axes[0, 0].plot(df['epoch'] + df['itr'] / df['itr'].max(), df['loss'], alpha=0.6)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Epoch-wise average loss
    epoch_loss = df.groupby('epoch')['loss'].mean()
    axes[0, 1].plot(epoch_loss.index, epoch_loss.values, marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].set_title('Average Loss per Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time per iteration
    axes[1, 0].plot(df['epoch'] + df['itr'] / df['itr'].max(), df['time (ms)'], alpha=0.6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_title('Iteration Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mask statistics
    axes[1, 1].plot(df['epoch'] + df['itr'] / df['itr'].max(), df['mask-context'], label='Context', alpha=0.6)
    axes[1, 1].plot(df['epoch'] + df['itr'] / df['itr'].max(), df['mask-pred'], label='Prediction', alpha=0.6)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Mask Size')
    axes[1, 1].set_title('Mask Statistics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved training curves to {output_path}")
    plt.close()


def plot_representation_tsne(representations, labels, label_names, output_dir, title='t-SNE of Representations'):
    """Plot t-SNE visualization of representations"""
    logger.info(f"Computing t-SNE for {len(representations)} samples...")
    
    # First reduce dimensionality with PCA if needed
    if representations.shape[1] > 50:
        pca = PCA(n_components=50)
        representations = pca.fit_transform(representations)
        logger.info(f"  PCA: {representations.shape[1]} dimensions (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(representations)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[colors[i]], label=label_names.get(label, f'Class {label}'),
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tsne_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved t-SNE visualization to {output_path}")
    plt.close()


def plot_downstream_comparison(results_file, output_dir):
    """Plot comparison of downstream task results"""
    logger.info("Plotting downstream task comparison...")
    
    with open(results_file, 'r') as f:
        results = yaml.load(f, Loader=yaml.FullLoader)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mortality Prediction
    methods = ['JEPA', 'Raw', 'Random']
    aucs = [
        results['mortality']['jepa']['auc'],
        results['mortality']['raw']['auc'],
        results['mortality']['random']['auc']
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = axes[0].bar(methods, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random Chance')
    axes[0].set_ylabel('AUC-ROC', fontsize=12)
    axes[0].set_title('Mortality Prediction', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    # Length of Stay Prediction
    r2_scores = [
        results['length_of_stay']['jepa']['r2'],
        results['length_of_stay']['raw']['r2'],
        results['length_of_stay']['random']['r2']
    ]
    
    bars = axes[1].bar(methods, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline')
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('Length of Stay Prediction', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'downstream_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved downstream comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize JEPA-EHR results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to EHR data CSV')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to training log CSV')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to baseline comparison results')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("JEPA-EHR Visualization")
    logger.info("="*60)
    
    # 1. Plot training curves
    plot_training_curves(args.log_file, args.output_dir)
    
    # 2. Extract representations and visualize
    logger.info("\nExtracting representations for visualization...")
    extractor = RepresentationExtractor(args.checkpoint, args.config, device='cpu')
    
    dataset = MIMICEHRDataset(
        data_path=args.data_path,
        sequence_length=extractor.sequence_length,
        prediction_length=5,
        train=False
    )
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    representations = extractor.extract_representations(data_loader, pool='mean')
    
    # Get patient IDs and create labels based on a task
    patient_to_idx = {}
    for idx, (patient_id, _, _) in enumerate(dataset.sequences):
        if patient_id not in patient_to_idx:
            patient_to_idx[patient_id] = idx
    
    # Create task labels for coloring
    task_data = SyntheticTaskDataset(args.data_path)
    patient_ids, mortality_labels = task_data.create_mortality_task()
    
    # Map labels
    pid_to_label = dict(zip(patient_ids, mortality_labels))
    labels = np.array([pid_to_label.get(pid, -1) for pid in patient_to_idx.keys()])
    
    # Filter to patients with labels
    mask = labels != -1
    representations_labeled = representations[[patient_to_idx[pid] for pid, has_label in zip(patient_to_idx.keys(), mask) if has_label]]
    labels_filtered = labels[mask]
    
    label_names = {0: 'Low Risk', 1: 'High Risk'}
    plot_representation_tsne(
        representations_labeled,
        labels_filtered,
        label_names,
        args.output_dir,
        title='t-SNE of JEPA Representations (Colored by Mortality Risk)'
    )
    
    # 3. Plot downstream task comparison
    plot_downstream_comparison(args.results_file, args.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Visualization complete!")
    logger.info(f"Outputs saved to {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Baseline comparison: Compare JEPA representations with raw features

This script compares:
1. JEPA learned representations
2. Raw averaged features (baseline)
3. Random features (sanity check)

This helps validate that JEPA is learning useful representations.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from downstream_tasks import (
    RepresentationExtractor, SyntheticTaskDataset,
    linear_probe_evaluation, random_forest_evaluation
)
from src.datasets.mimic_ehr import MIMICEHRDataset


def extract_raw_features(data_path, sequence_length=20):
    """
    Extract raw features as baseline
    
    For each patient, average the raw features from their visits
    
    Returns:
        patient_to_features: Dict mapping patient_id to feature vector
    """
    logger.info("Extracting raw feature baseline...")
    
    df = pd.read_csv(data_path)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith('vital_') or c.startswith('lab_')]
    
    # Group by patient and average
    patient_features = {}
    for patient_id, group in df.groupby('patient_id'):
        if len(group) >= 10:
            # Take first sequence_length visits
            visits = group.head(sequence_length)
            
            # Average features
            features = visits[feature_cols].values
            mean_features = np.nanmean(features, axis=0)
            
            # Handle NaNs
            mean_features = np.nan_to_num(mean_features, nan=0.0)
            
            patient_features[patient_id] = mean_features
    
    logger.info(f"Extracted raw features for {len(patient_features)} patients, dim={mean_features.shape[0]}")
    
    return patient_features


def compare_representations(checkpoint_path, config_path, data_path, output_dir):
    """
    Compare JEPA representations with baselines
    """
    logger.info("="*60)
    logger.info("JEPA vs Baseline Comparison")
    logger.info("="*60)
    
    # 1. Extract JEPA representations
    logger.info("\n1. Extracting JEPA representations...")
    extractor = RepresentationExtractor(checkpoint_path, config_path, device='cpu')
    
    dataset = MIMICEHRDataset(
        data_path=data_path,
        sequence_length=extractor.sequence_length,
        prediction_length=5,
        train=False
    )
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    jepa_reps = extractor.extract_representations(data_loader, pool='mean')
    
    # Map to patients
    patient_to_idx = {}
    for idx, (patient_id, _, _) in enumerate(dataset.sequences):
        if patient_id not in patient_to_idx:
            patient_to_idx[patient_id] = idx
    
    logger.info(f"  JEPA representations: {jepa_reps.shape}")
    
    # 2. Extract raw features baseline
    logger.info("\n2. Extracting raw feature baseline...")
    raw_features = extract_raw_features(data_path, extractor.sequence_length)
    
    # Convert to array
    raw_feature_array = np.array([raw_features[pid] for pid in patient_to_idx.keys() if pid in raw_features])
    
    # Normalize
    scaler = StandardScaler()
    raw_feature_array = scaler.fit_transform(raw_feature_array)
    
    logger.info(f"  Raw features: {raw_feature_array.shape}")
    
    # 3. Random features (sanity check)
    logger.info("\n3. Creating random features (sanity check)...")
    np.random.seed(42)
    random_features = np.random.randn(len(jepa_reps), jepa_reps.shape[1])
    logger.info(f"  Random features: {random_features.shape}")
    
    # Create task data
    task_data = SyntheticTaskDataset(data_path)
    
    # Get common patient IDs
    common_pids = [pid for pid in patient_to_idx.keys() if pid in raw_features]
    common_indices = [patient_to_idx[pid] for pid in common_pids]
    
    # Align representations
    X_jepa = jepa_reps[common_indices]
    X_raw = raw_feature_array[:len(common_pids)]
    X_random = random_features[common_indices]
    
    logger.info(f"\nAligned {len(common_pids)} patients across all representations")
    
    results = {}
    
    # ========================================
    # Task 1: Mortality Prediction
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("Task 1: Mortality Prediction")
    logger.info("="*60)
    
    patient_ids, mortality_labels = task_data.create_mortality_task()
    
    # Get labels for common patients
    pid_to_label = dict(zip(patient_ids, mortality_labels))
    y = np.array([pid_to_label[pid] for pid in common_pids if pid in pid_to_label])
    
    # Filter to patients with labels
    mask = np.array([pid in pid_to_label for pid in common_pids])
    X_jepa_task = X_jepa[mask]
    X_raw_task = X_raw[mask]
    X_random_task = X_random[mask]
    
    # Train-test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    
    results_mortality = {}
    
    # JEPA
    logger.info("\nJEPA Representations:")
    metrics, _ = linear_probe_evaluation(
        X_jepa_task[train_idx], y[train_idx],
        X_jepa_task[test_idx], y[test_idx],
        'classification'
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    results_mortality['jepa'] = metrics
    
    # Raw features
    logger.info("\nRaw Features (Baseline):")
    metrics, _ = linear_probe_evaluation(
        X_raw_task[train_idx], y[train_idx],
        X_raw_task[test_idx], y[test_idx],
        'classification'
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    results_mortality['raw'] = metrics
    
    # Random features
    logger.info("\nRandom Features (Sanity Check):")
    metrics, _ = linear_probe_evaluation(
        X_random_task[train_idx], y[train_idx],
        X_random_task[test_idx], y[test_idx],
        'classification'
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    results_mortality['random'] = metrics
    
    results['mortality'] = results_mortality
    
    # ========================================
    # Task 2: Length of Stay
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("Task 2: Length of Stay Prediction")
    logger.info("="*60)
    
    patient_ids, los_values = task_data.create_length_of_stay_task()
    
    pid_to_los = dict(zip(patient_ids, los_values))
    y = np.array([pid_to_los[pid] for pid in common_pids if pid in pid_to_los])
    
    mask = np.array([pid in pid_to_los for pid in common_pids])
    X_jepa_task = X_jepa[mask]
    X_raw_task = X_raw[mask]
    X_random_task = X_random[mask]
    
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    results_los = {}
    
    # JEPA
    logger.info("\nJEPA Representations:")
    metrics, _ = linear_probe_evaluation(
        X_jepa_task[train_idx], y[train_idx],
        X_jepa_task[test_idx], y[test_idx],
        'regression'
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    results_los['jepa'] = metrics
    
    # Raw features
    logger.info("\nRaw Features (Baseline):")
    metrics, _ = linear_probe_evaluation(
        X_raw_task[train_idx], y[train_idx],
        X_raw_task[test_idx], y[test_idx],
        'regression'
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    results_los['raw'] = metrics
    
    # Random features
    logger.info("\nRandom Features (Sanity Check):")
    metrics, _ = linear_probe_evaluation(
        X_random_task[train_idx], y[train_idx],
        X_random_task[test_idx], y[test_idx],
        'regression'
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    results_los['random'] = metrics
    
    results['length_of_stay'] = results_los
    
    # Save results
    results_file = os.path.join(output_dir, 'baseline_comparison.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Print comparison summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    logger.info("\nMortality Prediction (AUC):")
    logger.info(f"  JEPA:   {results['mortality']['jepa']['auc']:.4f}")
    logger.info(f"  Raw:    {results['mortality']['raw']['auc']:.4f}")
    logger.info(f"  Random: {results['mortality']['random']['auc']:.4f}")
    
    jepa_better = results['mortality']['jepa']['auc'] > results['mortality']['raw']['auc']
    logger.info(f"  → JEPA {'BETTER' if jepa_better else 'WORSE'} than raw features")
    
    logger.info("\nLength of Stay Prediction (R²):")
    logger.info(f"  JEPA:   {results['length_of_stay']['jepa']['r2']:.4f}")
    logger.info(f"  Raw:    {results['length_of_stay']['raw']['r2']:.4f}")
    logger.info(f"  Random: {results['length_of_stay']['random']['r2']:.4f}")
    
    jepa_better = results['length_of_stay']['jepa']['r2'] > results['length_of_stay']['raw']['r2']
    logger.info(f"  → JEPA {'BETTER' if jepa_better else 'WORSE'} than raw features")


def main():
    parser = argparse.ArgumentParser(description='Compare JEPA with baseline representations')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to EHR data CSV')
    parser.add_argument('--output_dir', type=str, default='./downstream_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    compare_representations(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

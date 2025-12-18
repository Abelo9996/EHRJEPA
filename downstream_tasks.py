#!/usr/bin/env python3
"""
Downstream tasks for evaluating JEPA-EHR learned representations

This script implements several clinical prediction tasks:
1. Mortality prediction (binary classification)
2. Length of stay prediction (regression)
3. Readmission prediction (binary classification)
4. Patient similarity/clustering

The representations learned by JEPA are used as features for these tasks.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model architecture
import src.models.temporal_transformer as tt
from src.datasets.mimic_ehr import MIMICEHRDataset


class RepresentationExtractor:
    """Extract representations from trained JEPA encoder"""
    
    def __init__(self, checkpoint_path, config_path, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to training config
            device: Device to run on
        """
        self.device = device
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model parameters
        self.num_features = checkpoint.get('num_features', 25)
        self.sequence_length = checkpoint.get('sequence_length', 20)
        model_name = self.config['meta']['model_name']
        
        # Initialize encoder
        logger.info(f"Initializing encoder: {model_name}")
        self.encoder = tt.__dict__[model_name](
            num_features=self.num_features,
            sequence_length=self.sequence_length
        ).to(device)
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder.eval()
        
        logger.info("Encoder loaded successfully")
    
    def extract_representations(self, data_loader, pool='mean'):
        """
        Extract representations from data
        
        Args:
            data_loader: DataLoader with sequences
            pool: How to pool sequence representations ('mean', 'max', 'last')
        
        Returns:
            representations: np.array of shape (N, D)
        """
        all_reps = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get sequences
                if isinstance(batch, (list, tuple)):
                    sequences = batch[0]
                else:
                    sequences = batch
                
                sequences = sequences.to(self.device)
                
                # Forward through encoder
                h = self.encoder(sequences)  # (B, L, D)
                
                # Pool representations
                if pool == 'mean':
                    h = h.mean(dim=1)  # (B, D)
                elif pool == 'max':
                    h = h.max(dim=1)[0]  # (B, D)
                elif pool == 'last':
                    h = h[:, -1, :]  # (B, D)
                else:
                    raise ValueError(f"Unknown pooling: {pool}")
                
                all_reps.append(h.cpu().numpy())
        
        return np.concatenate(all_reps, axis=0)


class SyntheticTaskDataset:
    """Generate synthetic downstream tasks from EHR data"""
    
    def __init__(self, data_path, seed=42):
        """
        Args:
            data_path: Path to EHR CSV data
            seed: Random seed
        """
        np.random.seed(seed)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_csv(data_path)
        
        # Group by patient
        self.patients = self.df.groupby('patient_id')
        
        logger.info(f"Loaded {len(self.patients)} patients")
    
    def create_mortality_task(self):
        """
        Create mortality prediction task
        
        For each patient, predict mortality based on their trajectory.
        We'll simulate this based on trend in vital signs.
        
        Returns:
            patient_ids, labels (1=mortality, 0=survival)
        """
        logger.info("Creating mortality prediction task...")
        
        patient_ids = []
        labels = []
        
        for patient_id, group in self.patients:
            # Use trend in vital signs as proxy for mortality risk
            # In real data, use actual mortality labels
            vital_cols = [c for c in group.columns if c.startswith('vital_')]
            
            if len(vital_cols) > 0 and len(group) >= 10:
                # Calculate mean and trend
                vitals = group[vital_cols].values
                mean_vitals = np.nanmean(vitals)
                
                # Simulate mortality: high mean + high variance = higher risk
                variance = np.nanvar(vitals)
                risk_score = mean_vitals + variance / 10
                
                # Threshold to create binary labels (roughly 30% mortality)
                threshold = np.percentile([mean_vitals + np.nanvar(self.patients.get_group(pid)[vital_cols].values) / 10 
                                          for pid, _ in self.patients if len(self.patients.get_group(pid)) >= 10], 70)
                
                label = 1 if risk_score > threshold else 0
                
                patient_ids.append(patient_id)
                labels.append(label)
        
        labels = np.array(labels)
        logger.info(f"Created {len(labels)} samples, {labels.sum()} positive ({labels.mean()*100:.1f}%)")
        
        return patient_ids, labels
    
    def create_length_of_stay_task(self):
        """
        Create length of stay prediction task
        
        Predict the number of visits/days for each patient.
        
        Returns:
            patient_ids, los_values
        """
        logger.info("Creating length of stay prediction task...")
        
        patient_ids = []
        los_values = []
        
        for patient_id, group in self.patients:
            if len(group) >= 10:
                # Length of stay = number of visits
                los = len(group)
                
                patient_ids.append(patient_id)
                los_values.append(los)
        
        los_values = np.array(los_values)
        logger.info(f"Created {len(los_values)} samples, mean LOS: {los_values.mean():.1f}")
        
        return patient_ids, los_values
    
    def create_readmission_task(self):
        """
        Create 30-day readmission prediction task
        
        Simulate readmission risk based on visit frequency.
        
        Returns:
            patient_ids, labels (1=readmission, 0=no readmission)
        """
        logger.info("Creating readmission prediction task...")
        
        patient_ids = []
        labels = []
        
        for patient_id, group in self.patients:
            if len(group) >= 10:
                # High visit frequency = higher readmission risk
                visit_frequency = len(group) / 30  # visits per month
                
                # Threshold for binary classification
                threshold = 1.0  # 1 visit per month
                label = 1 if visit_frequency > threshold else 0
                
                patient_ids.append(patient_id)
                labels.append(label)
        
        labels = np.array(labels)
        logger.info(f"Created {len(labels)} samples, {labels.sum()} positive ({labels.mean()*100:.1f}%)")
        
        return patient_ids, labels


def linear_probe_evaluation(X_train, y_train, X_test, y_test, task_type='classification'):
    """
    Linear probing: Train a linear model on frozen representations
    
    Args:
        X_train, y_train: Training features and labels
        X_test, y_test: Test features and labels
        task_type: 'classification' or 'regression'
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f"Linear probe evaluation ({task_type})...")
    
    if task_type == 'classification':
        # Logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
    else:
        # Ridge regression
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return metrics, model


def random_forest_evaluation(X_train, y_train, X_test, y_test, task_type='classification'):
    """
    Random Forest evaluation: Non-linear model on representations
    
    Args:
        X_train, y_train: Training features and labels
        X_test, y_test: Test features and labels
        task_type: 'classification' or 'regression'
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f"Random Forest evaluation ({task_type})...")
    
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return metrics, model


def main():
    parser = argparse.ArgumentParser(description='Evaluate JEPA-EHR representations on downstream tasks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to EHR data CSV')
    parser.add_argument('--output_dir', type=str, default='./downstream_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for representation extraction')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize representation extractor
    logger.info("="*60)
    logger.info("JEPA-EHR Downstream Task Evaluation")
    logger.info("="*60)
    
    extractor = RepresentationExtractor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Load data for tasks
    task_data = SyntheticTaskDataset(args.data_path)
    
    # Create dataset for representation extraction
    dataset = MIMICEHRDataset(
        data_path=args.data_path,
        sequence_length=extractor.sequence_length,
        prediction_length=5,
        train=False
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Extract representations
    logger.info("\n" + "="*60)
    logger.info("Extracting representations...")
    logger.info("="*60)
    representations = extractor.extract_representations(data_loader, pool='mean')
    logger.info(f"Extracted representations: {representations.shape}")
    
    # Map representations to patients
    # For simplicity, we'll use first sequence per patient
    patient_to_idx = {}
    for idx, (patient_id, _, _) in enumerate(dataset.sequences):
        if patient_id not in patient_to_idx:
            patient_to_idx[patient_id] = idx
    
    results = {}
    
    # Task 1: Mortality Prediction
    logger.info("\n" + "="*60)
    logger.info("Task 1: Mortality Prediction")
    logger.info("="*60)
    
    patient_ids, mortality_labels = task_data.create_mortality_task()
    
    # Get representations for these patients
    indices = [patient_to_idx[pid] for pid in patient_ids if pid in patient_to_idx]
    X_mortality = representations[indices]
    y_mortality = np.array([mortality_labels[i] for i, pid in enumerate(patient_ids) if pid in patient_to_idx])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_mortality, y_mortality, test_size=0.3, random_state=42, stratify=y_mortality
    )
    
    # Linear probe
    metrics_linear, _ = linear_probe_evaluation(X_train, y_train, X_test, y_test, 'classification')
    logger.info("\nLinear Probe Results:")
    for k, v in metrics_linear.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Random Forest
    metrics_rf, _ = random_forest_evaluation(X_train, y_train, X_test, y_test, 'classification')
    logger.info("\nRandom Forest Results:")
    for k, v in metrics_rf.items():
        logger.info(f"  {k}: {v:.4f}")
    
    results['mortality'] = {
        'linear_probe': metrics_linear,
        'random_forest': metrics_rf
    }
    
    # Task 2: Length of Stay Prediction
    logger.info("\n" + "="*60)
    logger.info("Task 2: Length of Stay Prediction")
    logger.info("="*60)
    
    patient_ids, los_values = task_data.create_length_of_stay_task()
    
    indices = [patient_to_idx[pid] for pid in patient_ids if pid in patient_to_idx]
    X_los = representations[indices]
    y_los = np.array([los_values[i] for i, pid in enumerate(patient_ids) if pid in patient_to_idx])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_los, y_los, test_size=0.3, random_state=42
    )
    
    # Linear probe
    metrics_linear, _ = linear_probe_evaluation(X_train, y_train, X_test, y_test, 'regression')
    logger.info("\nLinear Probe Results:")
    for k, v in metrics_linear.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Random Forest
    metrics_rf, _ = random_forest_evaluation(X_train, y_train, X_test, y_test, 'regression')
    logger.info("\nRandom Forest Results:")
    for k, v in metrics_rf.items():
        logger.info(f"  {k}: {v:.4f}")
    
    results['length_of_stay'] = {
        'linear_probe': metrics_linear,
        'random_forest': metrics_rf
    }
    
    # Task 3: Readmission Prediction
    logger.info("\n" + "="*60)
    logger.info("Task 3: Readmission Prediction")
    logger.info("="*60)
    
    patient_ids, readmission_labels = task_data.create_readmission_task()
    
    indices = [patient_to_idx[pid] for pid in patient_ids if pid in patient_to_idx]
    X_readmission = representations[indices]
    y_readmission = np.array([readmission_labels[i] for i, pid in enumerate(patient_ids) if pid in patient_to_idx])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_readmission, y_readmission, test_size=0.3, random_state=42, stratify=y_readmission
    )
    
    # Linear probe
    metrics_linear, _ = linear_probe_evaluation(X_train, y_train, X_test, y_test, 'classification')
    logger.info("\nLinear Probe Results:")
    for k, v in metrics_linear.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Random Forest
    metrics_rf, _ = random_forest_evaluation(X_train, y_train, X_test, y_test, 'classification')
    logger.info("\nRandom Forest Results:")
    for k, v in metrics_rf.items():
        logger.info(f"  {k}: {v:.4f}")
    
    results['readmission'] = {
        'linear_probe': metrics_linear,
        'random_forest': metrics_rf
    }
    
    # Save results
    results_file = os.path.join(args.output_dir, 'downstream_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    logger.info("\n" + "="*60)
    logger.info(f"Results saved to {results_file}")
    logger.info("="*60)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("\nMortality Prediction:")
    logger.info(f"  Linear Probe AUC: {results['mortality']['linear_probe']['auc']:.4f}")
    logger.info(f"  Random Forest AUC: {results['mortality']['random_forest']['auc']:.4f}")
    logger.info("\nLength of Stay Prediction:")
    logger.info(f"  Linear Probe R²: {results['length_of_stay']['linear_probe']['r2']:.4f}")
    logger.info(f"  Random Forest R²: {results['length_of_stay']['random_forest']['r2']:.4f}")
    logger.info("\nReadmission Prediction:")
    logger.info(f"  Linear Probe AUC: {results['readmission']['linear_probe']['auc']:.4f}")
    logger.info(f"  Random Forest AUC: {results['readmission']['random_forest']['auc']:.4f}")
    

if __name__ == '__main__':
    main()

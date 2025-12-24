"""
Evaluate JEPA-EHR model on visit-level prediction tasks
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Import model and dataset
from src.helper_ehr import init_model_ehr
from src.datasets.mimic_ehr import MIMICEHRDataset


def load_model_checkpoint(checkpoint_path, num_features):
    """Load trained JEPA encoder from checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Initialize model
    encoder, _ = init_model_ehr(
        device=torch.device('cpu'),
        num_features=num_features,
        sequence_length=20,
        model_name='temporal_transformer_base',
        pred_depth=8,
        pred_emb_dim=384
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load encoder weights (target encoder for evaluation)
    if 'target_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['target_encoder'])
        logger.info("Loaded target encoder weights")
    elif 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        logger.info("Loaded encoder weights")
    else:
        logger.warning("No encoder weights found in checkpoint, using random initialization")
    
    encoder.eval()
    return encoder


def extract_representations(encoder, dataset, max_samples=None):
    """Extract representations from the encoder for all sequences"""
    logger.info("Extracting representations...")
    
    representations = []
    patient_ids = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            if max_samples and i >= max_samples:
                break
                
            sequence = dataset[i]  # Shape: (total_length, num_features)
            
            # Take only context (first 15 visits)
            context = sequence[:15].unsqueeze(0)  # Shape: (1, 15, num_features)
            
            # Get encoder output
            output = encoder(context)  # Shape: (1, 15, embed_dim)
            
            # Pool over time dimension (mean pooling)
            pooled = output.mean(dim=1).squeeze(0)  # Shape: (embed_dim,)
            
            representations.append(pooled.cpu().numpy())
            
            # Get patient ID from dataset
            patient_id = dataset.sequences[i][0]
            patient_ids.append(patient_id)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} sequences")
    
    representations = np.array(representations)
    logger.info(f"Extracted representations shape: {representations.shape}")
    
    return representations, patient_ids


def create_prediction_labels(data_path, patient_ids, sequences):
    """Create labels for prediction tasks from the data"""
    logger.info("Creating prediction labels...")
    
    # Load full data
    df = pd.read_csv(data_path)
    
    # Rename columns
    if 'subject_id' in df.columns:
        df = df.rename(columns={'subject_id': 'patient_id'})
    if 'admittime' in df.columns:
        df = df.rename(columns={'admittime': 'visit_time'})
    
    df = df.sort_values(['patient_id', 'visit_time']).reset_index(drop=True)
    
    labels = {}
    
    # For each sequence, create labels based on future visits
    for idx, patient_id in enumerate(patient_ids):
        seq_info = sequences[idx]
        start_idx, end_idx = seq_info[1], seq_info[2]
        
        patient_data = df[df['patient_id'] == patient_id]
        
        # Task 1: 30-day readmission (if there's a visit within 30 days after the sequence)
        if end_idx < len(patient_data):
            last_visit_time = pd.to_datetime(patient_data.iloc[end_idx - 1]['visit_time'])
            next_visits = patient_data.iloc[end_idx:]
            
            if len(next_visits) > 0:
                next_visit_time = pd.to_datetime(next_visits.iloc[0]['visit_time'])
                days_to_next = (next_visit_time - last_visit_time).days
                labels.setdefault('readmission_30d', []).append(1 if days_to_next <= 30 else 0)
            else:
                labels.setdefault('readmission_30d', []).append(0)
        else:
            labels.setdefault('readmission_30d', []).append(0)
        
        # Task 2: In-hospital mortality (if hospital_expire_flag exists)
        if 'hospital_expire_flag' in patient_data.columns:
            # Check if any visit in the sequence had mortality
            seq_data = patient_data.iloc[start_idx:end_idx]
            mortality = seq_data['hospital_expire_flag'].max()
            labels.setdefault('mortality', []).append(int(mortality) if pd.notna(mortality) else 0)
        
        # Task 3: ICU admission (if any ICU-related columns exist)
        # This is a proxy - we'll use if there are ICU features
        labels.setdefault('has_next_visit', []).append(1 if end_idx < len(patient_data) else 0)
    
    # Convert to numpy arrays
    for key in labels:
        labels[key] = np.array(labels[key])
        logger.info(f"Task '{key}': {len(labels[key])} samples, {labels[key].sum()} positive ({100*labels[key].mean():.2f}%)")
    
    return labels


def evaluate_downstream_task(representations, labels, task_name):
    """Evaluate on a downstream task using linear probing"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating task: {task_name}")
    logger.info(f"{'='*60}")
    
    # Check class balance
    pos_rate = labels.mean()
    logger.info(f"Positive rate: {pos_rate:.3f} ({int(labels.sum())}/{len(labels)})")
    
    if pos_rate == 0 or pos_rate == 1:
        logger.warning(f"Task {task_name} has no variance (all same class), skipping")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    results = {}
    
    # Logistic Regression (linear probe)
    logger.info("\n--- Logistic Regression (Linear Probe) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = lr.predict(X_test)
    
    results['lr'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_pred_proba),
        'auprc': average_precision_score(y_test, y_pred_proba)
    }
    
    logger.info(f"Accuracy: {results['lr']['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['lr']['f1']:.4f}")
    logger.info(f"AUROC: {results['lr']['auroc']:.4f}")
    logger.info(f"AUPRC: {results['lr']['auprc']:.4f}")
    
    # Random Forest (non-linear probe)
    logger.info("\n--- Random Forest (Non-linear Probe) ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    
    results['rf'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'auroc': roc_auc_score(y_test, y_pred_proba_rf),
        'auprc': average_precision_score(y_test, y_pred_proba_rf)
    }
    
    logger.info(f"Accuracy: {results['rf']['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['rf']['f1']:.4f}")
    logger.info(f"AUROC: {results['rf']['auroc']:.4f}")
    logger.info(f"AUPRC: {results['rf']['auprc']:.4f}")
    
    return results


def main():
    # Paths
    checkpoint_path = 'logs/jepa_visits/jepa-visits-ep10.pth.tar'
    train_data_path = 'data/mimic_visits/train_mimic_ehr.csv'
    val_data_path = 'data/mimic_visits/val_mimic_ehr.csv'
    
    # Check if using validation set
    if os.path.exists(val_data_path):
        data_path = val_data_path
        logger.info("Using validation set for evaluation")
    else:
        data_path = train_data_path
        logger.info("Using training set for evaluation")
    
    # Load dataset
    logger.info(f"Loading dataset from {data_path}")
    dataset = MIMICEHRDataset(
        data_path=data_path,
        sequence_length=15,  # Only use context
        prediction_length=5,
        use_cache=False
    )
    
    num_features = dataset.num_features
    logger.info(f"Number of features: {num_features}")
    logger.info(f"Number of sequences: {len(dataset)}")
    
    # Load model
    encoder = load_model_checkpoint(checkpoint_path, num_features)
    
    # Extract representations
    representations, patient_ids = extract_representations(encoder, dataset, max_samples=None)
    
    # Create labels
    labels_dict = create_prediction_labels(data_path, patient_ids, dataset.sequences)
    
    # Evaluate on each task
    all_results = {}
    for task_name, labels in labels_dict.items():
        if len(labels) == len(representations):
            results = evaluate_downstream_task(representations, labels, task_name)
            if results:
                all_results[task_name] = results
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    for task_name, results in all_results.items():
        logger.info(f"\n{task_name.upper()}:")
        logger.info(f"  Linear Probe (LR):  AUROC={results['lr']['auroc']:.4f}, AUPRC={results['lr']['auprc']:.4f}, F1={results['lr']['f1']:.4f}")
        logger.info(f"  Non-linear (RF):    AUROC={results['rf']['auroc']:.4f}, AUPRC={results['rf']['auprc']:.4f}, F1={results['rf']['f1']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()

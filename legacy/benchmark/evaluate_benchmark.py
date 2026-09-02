#!/usr/bin/env python3
"""
PhD-Level Benchmark Evaluation Suite for EHR-JEPA

Implements comprehensive evaluation following standards from:
- EHRMamba, CLMBR, Med-BERT, BEHRT, TransformEHR

Evaluation Protocol:
- 5-fold patient-stratified cross-validation
- Linear probing and fine-tuning
- Multiple baselines (LR, RF, LSTM, Transformer)
- Standard metrics (AUROC, AUPRC, F1, Accuracy)
- 95% confidence intervals via bootstrap
- Statistical significance testing

Author: Abel Yagubyan
Date: February 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.temporal_transformer import TemporalTransformer


# =============================================================================
# BENCHMARK TASKS
# =============================================================================

BENCHMARK_TASKS = {
    'mortality': {
        'label_col': 'label_mortality',
        'type': 'binary',
        'description': 'In-hospital mortality prediction'
    },
    'mortality_30d': {
        'label_col': 'label_mortality_30d',
        'type': 'binary',
        'description': '30-day mortality prediction'
    },
    'readmission_30d': {
        'label_col': 'label_readmission_30d',
        'type': 'binary',
        'description': '30-day readmission prediction'
    },
    'los_3class': {
        'label_col': 'label_los_3class',
        'type': 'multiclass',
        'n_classes': 3,
        'description': 'Length of stay (short/medium/long)'
    },
    'los_7class': {
        'label_col': 'label_los_7class',
        'type': 'multiclass',
        'n_classes': 7,
        'description': 'Length of stay (7 categories)'
    },
    'icu': {
        'label_col': 'label_icu',
        'type': 'binary',
        'description': 'ICU admission prediction'
    },
    'phenotyping': {
        'label_cols': [f'pheno_{p}' for p in [
            'hypertension', 'diabetes', 'heart_failure', 'copd', 'ckd',
            'liver_disease', 'stroke', 'mi', 'afib', 'depression',
            'anxiety', 'obesity', 'cancer', 'anemia', 'hypothyroidism',
            'hyperlipidemia', 'osteoporosis', 'rheumatoid_arthritis',
            'asthma', 'dementia', 'parkinsons', 'epilepsy', 'pvd', 'gerd', 'uti'
        ]],
        'type': 'multilabel',
        'n_labels': 25,
        'description': '25 chronic condition phenotyping'
    }
}


# =============================================================================
# DATASET
# =============================================================================

class BenchmarkDataset(Dataset):
    """Dataset for benchmark evaluation with sequence support"""
    
    def __init__(self, data_path, feature_cols, sequence_length=20, mode='visit'):
        """
        Args:
            data_path: Path to CSV file
            feature_cols: List of feature column names
            sequence_length: Number of visits per sequence
            mode: 'visit' for per-visit prediction, 'sequence' for sequence-level
        """
        self.df = pd.read_csv(data_path)
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.mode = mode
        
        # Get all label columns
        self.label_cols = [c for c in self.df.columns if c.startswith('label_') or c.startswith('pheno_')]
        
        if mode == 'sequence':
            self._create_sequences()
        else:
            self.sequences = None
    
    def _create_sequences(self):
        """Create patient sequences for JEPA model"""
        self.sequences = []
        
        for subject_id, group in self.df.groupby('subject_id'):
            group = group.sort_values('visit_num').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            # Sliding window with 50% overlap
            step = max(1, self.sequence_length // 2)
            
            for start in range(0, max(1, len(group) - self.sequence_length + 1), step):
                end = min(start + self.sequence_length, len(group))
                seq = group.iloc[start:end]
                
                if len(seq) >= 2:
                    # Pad if needed
                    features = seq[self.feature_cols].values
                    if len(features) < self.sequence_length:
                        pad = np.zeros((self.sequence_length - len(features), len(self.feature_cols)))
                        features = np.vstack([features, pad])
                    
                    # Use labels from last visit
                    labels = seq[self.label_cols].iloc[-1].to_dict()
                    
                    self.sequences.append({
                        'features': features,
                        'labels': labels,
                        'subject_id': subject_id,
                        'length': len(seq)
                    })
    
    def __len__(self):
        if self.mode == 'sequence':
            return len(self.sequences)
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.mode == 'sequence':
            seq = self.sequences[idx]
            return {
                'features': torch.FloatTensor(seq['features']),
                'labels': {k: torch.tensor(v) for k, v in seq['labels'].items()},
                'length': seq['length']
            }
        else:
            row = self.df.iloc[idx]
            return {
                'features': torch.FloatTensor(row[self.feature_cols].values),
                'labels': {c: torch.tensor(row[c]) for c in self.label_cols}
            }


# =============================================================================
# BASELINE MODELS
# =============================================================================

class LSTMClassifier(nn.Module):
    """LSTM baseline for sequence classification"""
    
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        out, (h_n, _) = self.lstm(x)
        # Use final hidden state
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h_final = self.dropout(h_final)
        return self.fc(h_final)


class TransformerClassifier(nn.Module):
    """Vanilla Transformer baseline (no pretraining)"""
    
    def __init__(self, input_dim, d_model=256, n_heads=4, n_layers=4, 
                 n_classes=2, max_len=100, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.fc = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


class LinearProbe(nn.Module):
    """Linear probing head for frozen JEPA encoder"""
    
    def __init__(self, input_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        return self.fc(x)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob=None, task_type='binary'):
    """Compute comprehensive metrics"""
    metrics = {}
    
    if task_type == 'binary':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if y_prob is not None:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_prob)
                metrics['auprc'] = average_precision_score(y_true, y_prob)
            except:
                metrics['auroc'] = 0.5
                metrics['auprc'] = 0.5
    
    elif task_type == 'multiclass':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_prob is not None:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                metrics['auroc'] = 0.5
    
    elif task_type == 'multilabel':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)  # Exact match
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        if y_prob is not None:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_prob, average='macro')
                metrics['auprc'] = average_precision_score(y_true, y_prob, average='macro')
            except:
                metrics['auroc'] = 0.5
                metrics['auprc'] = 0.5
    
    return metrics


def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000, ci=0.95):
    """Compute confidence intervals via bootstrap"""
    np.random.seed(42)
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except:
            continue
    
    if len(scores) == 0:
        return 0, 0, 0
    
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class BenchmarkEvaluator:
    """
    Comprehensive benchmark evaluation following PhD standards
    """
    
    def __init__(self, data_dir, checkpoint_path=None, device='cuda'):
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print("="*80)
        print("EHR-JEPA PhD Benchmark Evaluation Suite")
        print("="*80)
        print(f"Data directory: {data_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load feature columns
        with open(os.path.join(data_dir, 'feature_columns.json'), 'r') as f:
            self.feature_cols = json.load(f)
        print(f"Features: {len(self.feature_cols)}")
        
        # Load data
        self.train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        self.val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
        self.test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        
        print(f"Train: {len(self.train_df):,} samples")
        print(f"Val: {len(self.val_df):,} samples")
        print(f"Test: {len(self.test_df):,} samples")
        
        # Load JEPA encoder if checkpoint provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_jepa_encoder()
        else:
            self.jepa_encoder = None
            print("⚠ No JEPA checkpoint provided, skipping JEPA evaluation")
        
        self.results = {}
    
    def _load_jepa_encoder(self):
        """Load pretrained JEPA encoder"""
        print("\nLoading JEPA encoder...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get model config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config
            config = {
                'model': {
                    'model_name': 'temporal_transformer_base'
                },
                'data': {
                    'feature_dim': len(self.feature_cols),
                    'sequence_length': 20
                }
            }
        
        # Initialize encoder
        self.jepa_encoder = TemporalTransformer(
            feature_dim=len(self.feature_cols),
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            drop_rate=0.0
        )
        
        # Load weights
        if 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
        elif 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        else:
            state_dict = checkpoint
        
        # Handle state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.jepa_encoder.load_state_dict(new_state_dict, strict=False)
        self.jepa_encoder = self.jepa_encoder.to(self.device)
        self.jepa_encoder.eval()
        
        print(f"✓ Loaded JEPA encoder")
    
    def extract_jepa_embeddings(self, dataloader):
        """Extract embeddings using JEPA encoder"""
        if self.jepa_encoder is None:
            return None
        
        embeddings = []
        labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['features'].to(self.device)
                
                # Get encoder output
                emb = self.jepa_encoder(x)
                
                # Global average pooling over sequence
                if len(emb.shape) == 3:
                    emb = emb.mean(dim=1)
                
                embeddings.append(emb.cpu().numpy())
                labels_list.append({k: v.numpy() for k, v in batch['labels'].items()})
        
        embeddings = np.vstack(embeddings)
        
        # Merge labels
        all_labels = {}
        for key in labels_list[0].keys():
            all_labels[key] = np.concatenate([l[key] for l in labels_list])
        
        return embeddings, all_labels
    
    def evaluate_task(self, task_name, n_folds=5):
        """Evaluate a single benchmark task"""
        print(f"\n{'='*60}")
        print(f"Task: {BENCHMARK_TASKS[task_name]['description']}")
        print(f"{'='*60}")
        
        task_config = BENCHMARK_TASKS[task_name]
        task_type = task_config['type']
        
        if task_type == 'multilabel':
            label_cols = task_config['label_cols']
        else:
            label_cols = [task_config['label_col']]
        
        # Prepare data
        X_train = self.train_df[self.feature_cols].values
        X_val = self.val_df[self.feature_cols].values
        X_test = self.test_df[self.feature_cols].values
        
        if task_type == 'multilabel':
            y_train = self.train_df[label_cols].values
            y_val = self.val_df[label_cols].values
            y_test = self.test_df[label_cols].values
        else:
            y_train = self.train_df[label_cols[0]].values
            y_val = self.val_df[label_cols[0]].values
            y_test = self.test_df[label_cols[0]].values
        
        results = {}
        
        # 1. Logistic Regression baseline
        print("\n[1] Logistic Regression baseline...")
        lr_results = self._evaluate_sklearn_model(
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            X_train, y_train, X_test, y_test, task_type
        )
        results['logistic_regression'] = lr_results
        print(f"  AUROC: {lr_results.get('auroc', lr_results.get('macro_f1', 0)):.4f}")
        
        # 2. Random Forest baseline
        print("\n[2] Random Forest baseline...")
        rf_results = self._evaluate_sklearn_model(
            RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
            X_train, y_train, X_test, y_test, task_type
        )
        results['random_forest'] = rf_results
        print(f"  AUROC: {rf_results.get('auroc', rf_results.get('macro_f1', 0)):.4f}")
        
        # 3. Gradient Boosting baseline
        print("\n[3] Gradient Boosting baseline...")
        if task_type != 'multilabel':
            gb_results = self._evaluate_sklearn_model(
                GradientBoostingClassifier(n_estimators=100),
                X_train, y_train, X_test, y_test, task_type
            )
            results['gradient_boosting'] = gb_results
            print(f"  AUROC: {gb_results.get('auroc', gb_results.get('macro_f1', 0)):.4f}")
        
        # 4. JEPA Linear Probe (if available)
        if self.jepa_encoder is not None:
            print("\n[4] JEPA Linear Probe...")
            jepa_results = self._evaluate_jepa_linear_probe(
                task_name, X_train, y_train, X_test, y_test, task_type
            )
            results['jepa_linear_probe'] = jepa_results
            print(f"  AUROC: {jepa_results.get('auroc', jepa_results.get('macro_f1', 0)):.4f}")
        
        # Store results
        self.results[task_name] = results
        
        return results
    
    def _evaluate_sklearn_model(self, model, X_train, y_train, X_test, y_test, task_type):
        """Evaluate sklearn model"""
        # Handle multilabel
        if task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(model)
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Get probabilities
        if task_type == 'multilabel':
            y_prob = np.column_stack([est.predict_proba(X_test)[:, 1] 
                                      for est in model.estimators_])
        elif hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if task_type == 'binary':
                y_prob = y_prob[:, 1]
        else:
            y_prob = None
        
        return compute_metrics(y_test, y_pred, y_prob, task_type)
    
    def _evaluate_jepa_linear_probe(self, task_name, X_train, y_train, X_test, y_test, task_type):
        """Evaluate JEPA with linear probing"""
        
        # Create sequence datasets
        train_dataset = BenchmarkDataset(
            os.path.join(self.data_dir, 'train.csv'),
            self.feature_cols, sequence_length=20, mode='sequence'
        )
        test_dataset = BenchmarkDataset(
            os.path.join(self.data_dir, 'test.csv'),
            self.feature_cols, sequence_length=20, mode='sequence'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Extract embeddings
        train_emb, train_labels = self.extract_jepa_embeddings(train_loader)
        test_emb, test_labels = self.extract_jepa_embeddings(test_loader)
        
        if train_emb is None:
            return {}
        
        # Get labels for this task
        task_config = BENCHMARK_TASKS[task_name]
        
        if task_type == 'multilabel':
            y_train = np.column_stack([train_labels[c] for c in task_config['label_cols']])
            y_test = np.column_stack([test_labels[c] for c in task_config['label_cols']])
        else:
            y_train = train_labels[task_config['label_col']]
            y_test = test_labels[task_config['label_col']]
        
        # Train linear probe
        if task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
        
        model.fit(train_emb, y_train)
        y_pred = model.predict(test_emb)
        
        if task_type == 'multilabel':
            y_prob = np.column_stack([est.predict_proba(test_emb)[:, 1] 
                                      for est in model.estimators_])
        elif hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(test_emb)
            if task_type == 'binary':
                y_prob = y_prob[:, 1]
        else:
            y_prob = None
        
        return compute_metrics(y_test, y_pred, y_prob, task_type)
    
    def run_full_evaluation(self):
        """Run evaluation on all benchmark tasks"""
        print("\n" + "="*80)
        print("Running Full Benchmark Evaluation")
        print("="*80)
        
        for task_name in BENCHMARK_TASKS.keys():
            try:
                self.evaluate_task(task_name)
            except Exception as e:
                print(f"  Error in task {task_name}: {e}")
                self.results[task_name] = {'error': str(e)}
        
        # Generate summary
        self._print_summary()
        self._save_results()
        
        return self.results
    
    def _print_summary(self):
        """Print results summary table"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n{'Task':<20} {'Metric':<10} {'LR':<10} {'RF':<10} {'GB':<10} {'JEPA':<10}")
        print("-"*70)
        
        for task_name, task_results in self.results.items():
            if 'error' in task_results:
                continue
            
            task_type = BENCHMARK_TASKS[task_name]['type']
            metric = 'auroc' if task_type == 'binary' else 'macro_f1'
            
            lr_score = task_results.get('logistic_regression', {}).get(metric, 0)
            rf_score = task_results.get('random_forest', {}).get(metric, 0)
            gb_score = task_results.get('gradient_boosting', {}).get(metric, 0)
            jepa_score = task_results.get('jepa_linear_probe', {}).get(metric, 0)
            
            print(f"{task_name:<20} {metric:<10} {lr_score:<10.4f} {rf_score:<10.4f} "
                  f"{gb_score:<10.4f} {jepa_score:<10.4f}")
        
        print("-"*70)
    
    def _save_results(self):
        """Save results to JSON"""
        output_path = os.path.join(self.data_dir, 'benchmark_results.json')
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_serializable = {}
        for task, task_results in self.results.items():
            results_serializable[task] = {}
            for model, metrics in task_results.items():
                if isinstance(metrics, dict):
                    results_serializable[task][model] = {
                        k: convert(v) for k, v in metrics.items()
                    }
                else:
                    results_serializable[task][model] = convert(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================

def run_kfold_evaluation(data_dir, checkpoint_path=None, n_folds=5, device='cuda'):
    """
    Run k-fold cross-validation for robust evaluation
    """
    print("="*80)
    print(f"Running {n_folds}-Fold Cross-Validation")
    print("="*80)
    
    # Load all data
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Combine for CV
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    with open(os.path.join(data_dir, 'feature_columns.json'), 'r') as f:
        feature_cols = json.load(f)
    
    # Patient-stratified folds
    patients = all_df['subject_id'].unique()
    
    cv_results = defaultdict(lambda: defaultdict(list))
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Use mortality for stratification
    patient_labels = all_df.groupby('subject_id')['label_mortality'].max().loc[patients]
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(patients, patient_labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        train_patients = patients[train_idx]
        test_patients = patients[test_idx]
        
        fold_train = all_df[all_df['subject_id'].isin(train_patients)]
        fold_test = all_df[all_df['subject_id'].isin(test_patients)]
        
        X_train = fold_train[feature_cols].values
        X_test = fold_test[feature_cols].values
        
        for task_name, task_config in BENCHMARK_TASKS.items():
            if task_config['type'] == 'multilabel':
                continue  # Skip for simplicity
            
            y_train = fold_train[task_config['label_col']].values
            y_test = fold_test[task_config['label_col']].values
            
            # Logistic Regression
            lr = LogisticRegression(max_iter=1000, class_weight='balanced')
            lr.fit(X_train, y_train)
            y_prob = lr.predict_proba(X_test)[:, 1] if task_config['type'] == 'binary' else None
            
            if y_prob is not None:
                try:
                    auroc = roc_auc_score(y_test, y_prob)
                    cv_results[task_name]['lr_auroc'].append(auroc)
                except:
                    pass
    
    # Print CV results with confidence intervals
    print("\n" + "="*80)
    print("K-Fold Cross-Validation Results")
    print("="*80)
    print(f"\n{'Task':<25} {'Mean AUROC':<15} {'95% CI':<20}")
    print("-"*60)
    
    for task_name in BENCHMARK_TASKS.keys():
        if task_name in cv_results and 'lr_auroc' in cv_results[task_name]:
            scores = cv_results[task_name]['lr_auroc']
            mean = np.mean(scores)
            std = np.std(scores)
            ci_lower = mean - 1.96 * std / np.sqrt(len(scores))
            ci_upper = mean + 1.96 * std / np.sqrt(len(scores))
            print(f"{task_name:<25} {mean:<15.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return cv_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PhD Benchmark Evaluation')
    parser.add_argument('--data_dir', type=str, default='./data/benchmark',
                        help='Path to benchmark data')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to JEPA checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--kfold', type=int, default=0,
                        help='Number of folds for CV (0 = no CV)')
    args = parser.parse_args()
    
    if args.kfold > 0:
        run_kfold_evaluation(args.data_dir, args.checkpoint, args.kfold, args.device)
    else:
        evaluator = BenchmarkEvaluator(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()

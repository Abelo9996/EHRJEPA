#!/usr/bin/env python3
"""
Neural Network Baseline Training for EHR-JEPA Benchmark

Trains LSTM and Transformer baselines (without pretraining)
for fair comparison against JEPA.

Baselines:
1. LSTM (bidirectional, 2 layers)
2. Transformer (4 layers, no pretraining)
3. GRU-D (with decay for missing data)

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MODELS
# =============================================================================

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequence classification"""
    
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention pooling
        attn_weights = self.attention(out).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        context = self.dropout(context)
        
        return self.classifier(context)


class TransformerClassifier(nn.Module):
    """Vanilla Transformer for sequence classification"""
    
    def __init__(self, input_dim, d_model=256, n_heads=4, n_layers=4,
                 n_classes=2, max_len=100, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        if mask is not None:
            # Convert to attention mask
            attn_mask = mask.float()
            attn_mask = attn_mask.masked_fill(mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(mask == 1, 0.0)
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)
        
        # Mean pooling
        x = x.mean(dim=1)
        return self.classifier(x)


class GRUDCell(nn.Module):
    """GRU-D cell with decay for missing data handling"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Decay parameters
        self.W_gamma_x = nn.Linear(input_dim, input_dim)
        self.W_gamma_h = nn.Linear(input_dim, hidden_dim)
        
        # GRU parameters
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(self, x, h, mask, delta):
        # delta: time since last observation
        # mask: observation mask (1=observed, 0=missing)
        
        # Input decay
        gamma_x = torch.exp(-F.relu(self.W_gamma_x(delta)))
        x_decay = mask * x + (1 - mask) * (gamma_x * x)
        
        # Hidden decay
        gamma_h = torch.exp(-F.relu(self.W_gamma_h(delta)))
        h_decay = gamma_h * h
        
        # GRU update
        combined = torch.cat([x_decay, h_decay], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        h_tilde = torch.tanh(self.W_h(torch.cat([x_decay, r * h_decay], dim=-1)))
        h_new = (1 - z) * h_decay + z * h_tilde
        
        return h_new


class GRUDClassifier(nn.Module):
    """GRU-D for handling missing data"""
    
    def __init__(self, input_dim, hidden_dim=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grud_cell = GRUDCell(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Create delta (time since last obs) - simplified: just use position
        delta = torch.ones_like(x)
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones_like(x)
        
        for t in range(seq_len):
            h = self.grud_cell(x[:, t], h, mask[:, t] if len(mask.shape) > 2 else mask[:, t:t+1].expand(-1, x.size(-1)), delta[:, t])
        
        return self.classifier(h)


# =============================================================================
# DATASET
# =============================================================================

class SequenceDataset(Dataset):
    """Dataset for sequence classification"""
    
    def __init__(self, df, feature_cols, label_col, sequence_length=20):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.sequence_length = sequence_length
        
        self.sequences = []
        self.labels = []
        
        for subject_id, group in df.groupby('subject_id'):
            group = group.sort_values('visit_num').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            step = max(1, self.sequence_length // 2)
            
            for start in range(0, max(1, len(group) - self.sequence_length + 1), step):
                end = min(start + self.sequence_length, len(group))
                seq = group.iloc[start:end]
                
                features = seq[self.feature_cols].values.astype(np.float32)
                label = seq[self.label_col].iloc[-1]
                
                # Pad if needed
                if len(features) < self.sequence_length:
                    pad = np.zeros((self.sequence_length - len(features), len(self.feature_cols)), dtype=np.float32)
                    features = np.vstack([features, pad])
                
                self.sequences.append(features)
                self.labels.append(label)
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.sequences[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0]
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                n_epochs, device, early_stop_patience=10):
    """Train neural network model"""
    
    best_val_auroc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(device)
                y = batch['label']
                
                logits = model(x)
                probs = F.softmax(logits, dim=-1)[:, 1]
                
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        try:
            val_auroc = roc_auc_score(val_labels, val_preds)
        except:
            val_auroc = 0.5
        
        print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f} | Val AUROC: {val_auroc:.4f}")
        
        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_auroc


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(device)
            y = batch['label']
            
            logits = model(x)
            probs = F.softmax(logits, dim=-1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {}
    try:
        metrics['auroc'] = roc_auc_score(all_labels, all_preds)
        metrics['auprc'] = average_precision_score(all_labels, all_preds)
    except:
        metrics['auroc'] = 0.5
        metrics['auprc'] = 0.5
    
    pred_labels = (all_preds > 0.5).astype(int)
    metrics['f1'] = f1_score(all_labels, pred_labels, zero_division=0)
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Neural Network Baselines')
    parser.add_argument('--data_dir', type=str, default='./data/benchmark',
                        help='Path to benchmark data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/baselines',
                        help='Output directory for models')
    parser.add_argument('--task', type=str, default='mortality',
                        help='Task to train on')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lstm', 'transformer', 'grud', 'all'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("Neural Network Baseline Training")
    print("="*80)
    
    # Load feature columns
    with open(os.path.join(args.data_dir, 'feature_columns.json'), 'r') as f:
        feature_cols = json.load(f)
    
    print(f"Features: {len(feature_cols)}")
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    
    # Task label column mapping
    task_labels = {
        'mortality': 'label_mortality',
        'mortality_30d': 'label_mortality_30d',
        'readmission_30d': 'label_readmission_30d',
        'icu': 'label_icu'
    }
    
    label_col = task_labels.get(args.task, f'label_{args.task}')
    print(f"Task: {args.task} | Label: {label_col}")
    
    # Create datasets
    train_dataset = SequenceDataset(train_df, feature_cols, label_col)
    val_dataset = SequenceDataset(val_df, feature_cols, label_col)
    test_dataset = SequenceDataset(test_df, feature_cols, label_col)
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Class weights for imbalanced data
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_dim = len(feature_cols)
    results = {}
    
    # Train models
    models_to_train = ['lstm', 'transformer', 'grud'] if args.model == 'all' else [args.model]
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        if model_name == 'lstm':
            model = LSTMClassifier(input_dim, args.hidden_dim, n_layers=2, n_classes=2)
        elif model_name == 'transformer':
            model = TransformerClassifier(input_dim, d_model=args.hidden_dim, n_classes=2)
        elif model_name == 'grud':
            model = GRUDClassifier(input_dim, args.hidden_dim, n_classes=2)
        
        model = model.to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        
        # Train
        model, val_auroc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, device, early_stop_patience=10
        )
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        results[model_name] = metrics
        
        print(f"\n{model_name.upper()} Test Results:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': vars(args),
            'metrics': metrics
        }, os.path.join(args.output_dir, f'{model_name}_{args.task}.pt'))
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Model':<15} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
    print("-"*45)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['auroc']:<10.4f} {metrics['auprc']:<10.4f} {metrics['f1']:<10.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, f'baseline_results_{args.task}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

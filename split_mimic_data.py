#!/usr/bin/env python3
"""
Split MIMIC-IV processed data into train/validation/test sets

Splits at the PATIENT level to prevent data leakage.
Each patient appears in only one split.
"""

import argparse
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime


def split_data(input_file, output_dir, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Split data by patient ID
    
    Args:
        input_file: Path to preprocessed MIMIC data
        output_dir: Directory to save splits
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
        seed: Random seed for reproducibility
    """
    
    print("="*80)
    print("MIMIC-IV Data Splitting")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Split: {train_frac:.1%} train, {val_frac:.1%} val, {test_frac:.1%} test")
    print("="*80)
    
    # Validate split fractions
    if abs(train_frac + val_frac + test_frac - 1.0) > 0.001:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_file)
    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  ✓ {df['subject_id'].nunique():,} unique patients")
    print(f"  ✓ {df['hadm_id'].nunique():,} unique admissions")
    
    # Get unique patients
    patients = df['subject_id'].unique()
    n_patients = len(patients)
    
    # Set random seed
    np.random.seed(seed)
    
    # Shuffle patients
    np.random.shuffle(patients)
    
    # Calculate split indices
    train_end = int(n_patients * train_frac)
    val_end = train_end + int(n_patients * val_frac)
    
    # Split patient IDs
    train_patients = set(patients[:train_end])
    val_patients = set(patients[train_end:val_end])
    test_patients = set(patients[val_end:])
    
    print(f"\nSplitting by patient (n={n_patients:,}):")
    print(f"  Train: {len(train_patients):,} patients ({len(train_patients)/n_patients:.1%})")
    print(f"  Val:   {len(val_patients):,} patients ({len(val_patients)/n_patients:.1%})")
    print(f"  Test:  {len(test_patients):,} patients ({len(test_patients)/n_patients:.1%})")
    
    # Split data
    print("\nCreating splits...")
    train_df = df[df['subject_id'].isin(train_patients)]
    val_df = df[df['subject_id'].isin(val_patients)]
    test_df = df[df['subject_id'].isin(test_patients)]
    
    print(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df):.1%})")
    print(f"  Val:   {len(val_df):,} rows ({len(val_df)/len(df):.1%})")
    print(f"  Test:  {len(test_df):,} rows ({len(test_df)/len(df):.1%})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    print("\nSaving splits...")
    train_path = os.path.join(output_dir, 'train_mimic_ehr.csv')
    val_path = os.path.join(output_dir, 'val_mimic_ehr.csv')
    test_path = os.path.join(output_dir, 'test_mimic_ehr.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ {train_path}")
    print(f"  ✓ {val_path}")
    print(f"  ✓ {test_path}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    
    feature_cols = [c for c in df.columns 
                   if c.startswith('chart_') or c.startswith('lab_')]
    
    stats = {
        'split_info': {
            'train_frac': train_frac,
            'val_frac': val_frac,
            'test_frac': test_frac,
            'random_seed': seed,
            'split_date': datetime.now().isoformat(),
        },
        'train': {
            'n_patients': int(train_df['subject_id'].nunique()),
            'n_admissions': int(train_df['hadm_id'].nunique()),
            'n_rows': int(len(train_df)),
            'avg_seq_length': float(len(train_df) / train_df['hadm_id'].nunique()),
        },
        'val': {
            'n_patients': int(val_df['subject_id'].nunique()),
            'n_admissions': int(val_df['hadm_id'].nunique()),
            'n_rows': int(len(val_df)),
            'avg_seq_length': float(len(val_df) / val_df['hadm_id'].nunique()),
        },
        'test': {
            'n_patients': int(test_df['subject_id'].nunique()),
            'n_admissions': int(test_df['hadm_id'].nunique()),
            'n_rows': int(len(test_df)),
            'avg_seq_length': float(len(test_df) / test_df['hadm_id'].nunique()),
        },
        'features': {
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
        },
    }
    
    # Feature statistics (for normalization)
    stats['normalization'] = {}
    for col in feature_cols:
        if col in train_df.columns:
            stats['normalization'][col] = {
                'mean': float(train_df[col].mean()),
                'std': float(train_df[col].std()),
                'min': float(train_df[col].min()),
                'max': float(train_df[col].max()),
                'missing_pct': float(train_df[col].isna().mean() * 100),
            }
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'split_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SPLIT COMPLETE!")
    print("="*80)
    print(f"Total patients: {n_patients:,}")
    print(f"Total rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nTrain set: {len(train_df):,} rows from {stats['train']['n_patients']:,} patients")
    print(f"Val set:   {len(val_df):,} rows from {stats['val']['n_patients']:,} patients")
    print(f"Test set:  {len(test_df):,} rows from {stats['test']['n_patients']:,} patients")
    print("="*80)
    print("\nNext step: Update config file with:")
    print(f"  data_path: {train_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Split MIMIC-IV data into train/val/test')
    
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input CSV file (preprocessed MIMIC data)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for splits')
    parser.add_argument('--train_frac', type=float, default=0.8,
                       help='Fraction for training (default: 0.8)')
    parser.add_argument('--val_frac', type=float, default=0.1,
                       help='Fraction for validation (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.1,
                       help='Fraction for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

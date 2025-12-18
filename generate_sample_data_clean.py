#!/usr/bin/env python3
"""
Generate sample MIMIC-IV-like EHR data for testing JEPA-EHR

This script creates synthetic patient visit data that mimics the structure
of MIMIC-IV hourly data for testing the JEPA-EHR model.

Usage:
    python generate_sample_data_clean.py --output_dir ./data/sample_ehr --num_patients 500
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_ehr_data(
    num_patients=500,
    min_visits_per_patient=15,
    max_visits_per_patient=50,
    num_vital_features=10,
    num_lab_features=15,
    output_dir='./data/sample_ehr',
    seed=42
):
    """
    Generate synthetic EHR data
    
    Args:
        num_patients: Number of patients to generate
        min_visits_per_patient: Minimum number of visits per patient
        max_visits_per_patient: Maximum number of visits per patient
        num_vital_features: Number of vital sign features
        num_lab_features: Number of lab test features
        output_dir: Directory to save the data
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print(f"Generating sample EHR data for {num_patients} patients...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature names
    vital_features = [f'vital_{i}' for i in range(num_vital_features)]
    lab_features = [f'lab_{i}' for i in range(num_lab_features)]
    feature_names = vital_features + lab_features
    
    all_data = []
    
    for patient_id in range(1, num_patients + 1):
        # Random number of visits for this patient
        num_visits = np.random.randint(min_visits_per_patient, max_visits_per_patient + 1)
        
        # Generate visit times (hourly intervals with some noise)
        start_time = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365))
        visit_times = [start_time + timedelta(hours=i) + timedelta(minutes=np.random.randint(-30, 30))
                      for i in range(num_visits)]
        
        # Generate patient-specific baseline values
        vital_baseline = np.random.randn(num_vital_features) * 10 + 50
        lab_baseline = np.random.randn(num_lab_features) * 5 + 20
        
        # Add some temporal trends and noise
        for visit_idx, visit_time in enumerate(visit_times):
            # Temporal trend (slight drift over time)
            trend = 0.1 * visit_idx / num_visits
            
            # Vital signs (with temporal autocorrelation)
            vitals = vital_baseline + np.random.randn(num_vital_features) * 3 + trend * 5
            
            # Lab values (with more noise and some missing values)
            labs = lab_baseline + np.random.randn(num_lab_features) * 2 + trend * 3
            # Randomly set some lab values to NaN (missing data)
            lab_mask = np.random.rand(num_lab_features) < 0.3  # 30% missing
            labs[lab_mask] = np.nan
            
            # Combine features
            features = np.concatenate([vitals, labs])
            
            # Create row
            row = {
                'patient_id': patient_id,
                'visit_time': visit_time,
                'hadm_id': patient_id * 1000 + visit_idx,  # Hospital admission ID
            }
            
            # Add feature values
            for feat_name, feat_val in zip(feature_names, features):
                row[feat_name] = feat_val
            
            all_data.append(row)
        
        if (patient_id % 100) == 0:
            print(f"  Generated data for {patient_id}/{num_patients} patients")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nDataset created with {len(df)} total visits")
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'sample_ehr_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  - Number of patients: {df['patient_id'].nunique()}")
    print(f"  - Number of visits: {len(df)}")
    print(f"  - Avg visits per patient: {len(df) / df['patient_id'].nunique():.1f}")
    print(f"  - Number of features: {len(feature_names)}")
    print(f"  - Missing data ratio: {df[feature_names].isna().sum().sum() / (len(df) * len(feature_names)):.2%}")
    
    return df, output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sample EHR data for JEPA-EHR testing')
    parser.add_argument('--num_patients', type=int, default=500,
                        help='Number of patients to generate')
    parser.add_argument('--min_visits', type=int, default=15,
                        help='Minimum visits per patient')
    parser.add_argument('--max_visits', type=int, default=50,
                        help='Maximum visits per patient')
    parser.add_argument('--num_vital_features', type=int, default=10,
                        help='Number of vital sign features')
    parser.add_argument('--num_lab_features', type=int, default=15,
                        help='Number of lab test features')
    parser.add_argument('--output_dir', type=str, default='./data/sample_ehr',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    df, output_file = generate_sample_ehr_data(
        num_patients=args.num_patients,
        min_visits_per_patient=args.min_visits,
        max_visits_per_patient=args.max_visits,
        num_vital_features=args.num_vital_features,
        num_lab_features=args.num_lab_features,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\nDone!")

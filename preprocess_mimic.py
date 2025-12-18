#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data for JEPA-EHR training

This script processes raw MIMIC-IV data into sequences suitable for training
the JEPA model on EHR data.

Strategy:
1. Load patient admissions and ICU stays
2. Extract relevant features from chartevents (vital signs) and labevents
3. Create hourly aggregated sequences per patient admission
4. Save processed data as CSV

Usage:
    python preprocess_mimic.py --mimic_dir ./mimic-iv-2.1 --output_dir ./data/processed_mimic
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_admissions(mimic_dir):
    """Load patient admissions data"""
    print("Loading admissions data...")
    admissions = pd.read_csv(os.path.join(mimic_dir, 'hosp', 'admissions.csv'))
    patients = pd.read_csv(os.path.join(mimic_dir, 'hosp', 'patients.csv'))
    
    # Merge with patient demographics
    admissions = admissions.merge(patients, on='subject_id', how='left')
    
    # Parse dates
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    
    # Calculate length of stay in hours
    admissions['los_hours'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 3600
    
    print(f"  Loaded {len(admissions)} admissions for {admissions['subject_id'].nunique()} patients")
    return admissions


def load_icu_stays(mimic_dir):
    """Load ICU stays data"""
    print("Loading ICU stays data...")
    try:
        icustays = pd.read_csv(os.path.join(mimic_dir, 'icu', 'icustays.csv'))
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])
        print(f"  Loaded {len(icustays)} ICU stays")
        return icustays
    except Exception as e:
        print(f"  Could not load ICU stays: {e}")
        return None


def load_chartevents_sample(mimic_dir, sample_frac=0.1, item_ids=None):
    """Load a sample of chart events (vital signs)"""
    print(f"Loading chart events (sampling {sample_frac*100}% of data)...")
    
    # Common vital sign item IDs
    if item_ids is None:
        item_ids = [
            220045,  # Heart Rate
            220050,  # Arterial Blood Pressure systolic
            220051,  # Arterial Blood Pressure diastolic
            220052,  # Arterial Blood Pressure mean
            220179,  # Non Invasive Blood Pressure systolic
            220180,  # Non Invasive Blood Pressure diastolic
            220181,  # Non Invasive Blood Pressure mean
            223761,  # Temperature Fahrenheit
            223762,  # Temperature Celsius
            220210,  # Respiratory Rate
            220277,  # O2 saturation pulseoxymetry
            224690,  # Respiratory Rate (Total)
        ]
    
    try:
        chartevents_file = os.path.join(mimic_dir, 'icu', 'chartevents.csv')
        
        # Load in chunks to handle large file
        print("  Reading chartevents in chunks (this may take a while)...")
        chunks = []
        chunk_size = 100000
        
        for i, chunk in enumerate(pd.read_csv(chartevents_file, chunksize=chunk_size)):
            # Filter by item IDs
            chunk = chunk[chunk['itemid'].isin(item_ids)]
            
            # Sample
            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            
            if len(chunk) > 0:
                chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {(i+1)*chunk_size} rows...")
        
        if chunks:
            chartevents = pd.concat(chunks, ignore_index=True)
            chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
            print(f"  Loaded {len(chartevents)} chart events")
            return chartevents
        else:
            print("  No chart events found")
            return None
            
    except Exception as e:
        print(f"  Could not load chart events: {e}")
        return None


def load_labevents_sample(mimic_dir, sample_frac=0.1, item_ids=None):
    """Load a sample of lab events"""
    print(f"Loading lab events (sampling {sample_frac*100}% of data)...")
    
    # Common lab test item IDs
    if item_ids is None:
        item_ids = [
            50861,  # Albumin
            50862,  # Albumin
            50863,  # Alkaline Phosphatase
            50868,  # Anion Gap
            50878,  # Asparate Aminotransferase (AST)
            50882,  # Bicarbonate
            50885,  # Bilirubin, Total
            50893,  # Calcium, Total
            50902,  # Chloride
            50912,  # Creatinine
            50931,  # Glucose
            50960,  # Magnesium
            50970,  # Phosphate
            50971,  # Potassium
            50983,  # Sodium
            51006,  # Urea Nitrogen
            51221,  # Hematocrit
            51222,  # Hemoglobin
            51248,  # MCH
            51249,  # MCHC
            51250,  # MCV
            51265,  # Platelet Count
            51277,  # RDW
            51279,  # Red Blood Cells
            51301,  # White Blood Cells
        ]
    
    try:
        labevents_file = os.path.join(mimic_dir, 'hosp', 'labevents.csv')
        
        # Load in chunks
        print("  Reading labevents in chunks (this may take a while)...")
        chunks = []
        chunk_size = 100000
        
        for i, chunk in enumerate(pd.read_csv(labevents_file, chunksize=chunk_size)):
            # Filter by item IDs
            chunk = chunk[chunk['itemid'].isin(item_ids)]
            
            # Sample
            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            
            if len(chunk) > 0:
                chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {(i+1)*chunk_size} rows...")
        
        if chunks:
            labevents = pd.concat(chunks, ignore_index=True)
            labevents['charttime'] = pd.to_datetime(labevents['charttime'])
            print(f"  Loaded {len(labevents)} lab events")
            return labevents
        else:
            print("  No lab events found")
            return None
            
    except Exception as e:
        print(f"  Could not load lab events: {e}")
        return None


def create_hourly_sequences(admissions, chartevents, labevents, min_seq_length=10):
    """
    Create hourly aggregated sequences per admission
    
    Args:
        admissions: DataFrame of admissions
        chartevents: DataFrame of chart events
        labevents: DataFrame of lab events
        min_seq_length: Minimum sequence length to keep
    
    Returns:
        DataFrame with hourly aggregated features per admission
    """
    print("\nCreating hourly sequences...")
    
    # Get unique subject_ids and hadm_ids from the events data
    if chartevents is not None:
        chart_subjects = set(chartevents['subject_id'].unique())
        chart_hadms = set(chartevents['hadm_id'].dropna().unique())
    else:
        chart_subjects = set()
        chart_hadms = set()
    
    if labevents is not None:
        lab_subjects = set(labevents['subject_id'].unique())
        lab_hadms = set(labevents['hadm_id'].dropna().unique())
    else:
        lab_subjects = set()
        lab_hadms = set()
    
    # Combine available subjects and hadms
    available_subjects = chart_subjects | lab_subjects
    available_hadms = chart_hadms | lab_hadms
    
    print(f"  Found {len(available_subjects)} patients with events")
    print(f"  Found {len(available_hadms)} admissions with events")
    
    # Filter admissions to only those with some events
    admissions = admissions[admissions['hadm_id'].isin(available_hadms)].copy()
    print(f"  Processing {len(admissions)} admissions with available events...")
    
    all_sequences = []
    
    # Process each admission
    for idx, adm in admissions.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processing admission {idx+1}/{len(admissions)}...")
        
        subject_id = adm['subject_id']
        hadm_id = adm['hadm_id']
        admit_time = adm['admittime']
        disch_time = adm['dischtime']
        
        # Skip very short admissions
        if adm['los_hours'] < min_seq_length:
            continue
        
        # Create hourly time bins
        hours = int(min(adm['los_hours'], 168))  # Cap at 1 week (168 hours)
        if hours < min_seq_length:
            continue
        
        time_bins = [admit_time + timedelta(hours=h) for h in range(hours + 1)]
        
        # Initialize features DataFrame
        sequence_df = pd.DataFrame({
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'hour': range(hours),
            'timestamp': time_bins[:-1]
        })
        
        # Extract chart events for this admission
        if chartevents is not None:
            adm_charts = chartevents[
                (chartevents['subject_id'] == subject_id) &
                (chartevents['hadm_id'] == hadm_id) &
                (chartevents['charttime'] >= admit_time) &
                (chartevents['charttime'] <= disch_time)
            ].copy()
            
            if len(adm_charts) > 0:
                # Bin by hour
                adm_charts['hour'] = ((adm_charts['charttime'] - admit_time).dt.total_seconds() / 3600).astype(int)
                adm_charts = adm_charts[adm_charts['hour'] < hours]
                
                # Pivot to get features
                chart_pivot = adm_charts.pivot_table(
                    index='hour',
                    columns='itemid',
                    values='valuenum',
                    aggfunc='mean'
                )
                
                # Rename columns
                chart_pivot.columns = [f'chart_{col}' for col in chart_pivot.columns]
                
                # Merge with sequence
                sequence_df = sequence_df.merge(chart_pivot, left_on='hour', right_index=True, how='left')
        
        # Extract lab events for this admission
        if labevents is not None:
            adm_labs = labevents[
                (labevents['subject_id'] == subject_id) &
                (labevents['hadm_id'] == hadm_id) &
                (labevents['charttime'] >= admit_time) &
                (labevents['charttime'] <= disch_time)
            ].copy()
            
            if len(adm_labs) > 0:
                # Bin by hour
                adm_labs['hour'] = ((adm_labs['charttime'] - admit_time).dt.total_seconds() / 3600).astype(int)
                adm_labs = adm_labs[adm_labs['hour'] < hours]
                
                # Pivot to get features
                lab_pivot = adm_labs.pivot_table(
                    index='hour',
                    columns='itemid',
                    values='valuenum',
                    aggfunc='mean'
                )
                
                # Rename columns
                lab_pivot.columns = [f'lab_{col}' for col in lab_pivot.columns]
                
                # Merge with sequence
                sequence_df = sequence_df.merge(lab_pivot, left_on='hour', right_index=True, how='left')
        
        # Add demographic features
        if 'anchor_age' in adm:
            sequence_df['age'] = adm['anchor_age']
        
        # Only keep if we have enough non-null values
        feature_cols = [c for c in sequence_df.columns if c.startswith('chart_') or c.startswith('lab_')]
        if len(feature_cols) > 0:
            non_null_ratio = sequence_df[feature_cols].notna().sum().sum() / (len(sequence_df) * len(feature_cols))
            if non_null_ratio > 0.1:  # At least 10% non-null
                all_sequences.append(sequence_df)
    
    if not all_sequences:
        print("  No sequences created!")
        return None
    
    # Combine all sequences
    print(f"\n  Created sequences for {len(all_sequences)} admissions")
    combined = pd.concat(all_sequences, ignore_index=True)
    
    print(f"  Total timepoints: {len(combined)}")
    print(f"  Total features: {len([c for c in combined.columns if c.startswith('chart_') or c.startswith('lab_')])}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-IV data for JEPA-EHR')
    parser.add_argument('--mimic_dir', type=str, default='./mimic-iv-2.1',
                        help='Path to MIMIC-IV data directory')
    parser.add_argument('--output_dir', type=str, default='./data/processed_mimic',
                        help='Output directory for processed data')
    parser.add_argument('--sample_frac', type=float, default=0.1,
                        help='Fraction of events data to sample (0.0-1.0)')
    parser.add_argument('--min_seq_length', type=int, default=10,
                        help='Minimum sequence length to keep')
    parser.add_argument('--max_admissions', type=int, default=None,
                        help='Maximum number of admissions to process (for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MIMIC-IV Data Preprocessing for JEPA-EHR")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load admissions
    admissions = load_admissions(args.mimic_dir)
    
    # Limit admissions if specified
    if args.max_admissions is not None:
        print(f"\nLimiting to {args.max_admissions} admissions for testing...")
        admissions = admissions.head(args.max_admissions)
    
    # Load events
    chartevents = load_chartevents_sample(args.mimic_dir, sample_frac=args.sample_frac)
    labevents = load_labevents_sample(args.mimic_dir, sample_frac=args.sample_frac)
    
    # Create sequences
    sequences = create_hourly_sequences(
        admissions=admissions,
        chartevents=chartevents,
        labevents=labevents,
        min_seq_length=args.min_seq_length
    )
    
    if sequences is not None:
        # Save processed data
        output_file = os.path.join(args.output_dir, 'mimic_hourly_sequences.csv')
        print(f"\nSaving processed data to {output_file}...")
        sequences.to_csv(output_file, index=False)
        
        # Save feature statistics
        feature_cols = [c for c in sequences.columns if c.startswith('chart_') or c.startswith('lab_')]
        stats = sequences[feature_cols].describe()
        stats_file = os.path.join(args.output_dir, 'feature_statistics.csv')
        stats.to_csv(stats_file)
        
        print("\n" + "="*60)
        print("Preprocessing complete!")
        print("="*60)
        print(f"Output file: {output_file}")
        print(f"Feature statistics: {stats_file}")
        print(f"Total sequences: {sequences['hadm_id'].nunique()}")
        print(f"Total timepoints: {len(sequences)}")
        print(f"Total features: {len(feature_cols)}")
    else:
        print("\nNo sequences created. Check your data!")


if __name__ == '__main__':
    main()

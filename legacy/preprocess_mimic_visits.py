#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data for VISIT-LEVEL sequences (not hourly)

This approach is designed for learning:
- Long-term disease progression
- Patient phenotypes and cohorts
- Readmission patterns
- Clinical trajectories over weeks/months

Each timestep = 1 hospital admission/visit
Features = aggregated summary of that encounter (diagnoses, procedures, labs, etc.)

Usage:
    python preprocess_mimic_visits.py \
        --mimic_dir ./mimic-iv-2.1 \
        --output_dir ./data/mimic_visits \
        --min_visits_per_patient 10
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')


class VisitLevelPreprocessor:
    """
    Create visit-level sequences for long-term clinical concept learning
    
    Key Differences from Hourly Approach:
    - Each row = 1 complete hospital admission (not 1 hour)
    - Features = visit summary (diagnoses, procedures, lab trends, vitals summary)
    - Time gaps = days/weeks/months between visits (realistic)
    - Learns disease progression, not hourly vital fluctuations
    """
    
    def __init__(self, mimic_dir, output_dir, min_visits_per_patient=10):
        self.mimic_dir = mimic_dir
        self.output_dir = output_dir
        self.min_visits_per_patient = min_visits_per_patient
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("MIMIC-IV Visit-Level Preprocessing")
        print("For Long-Term Clinical Concept Learning")
        print("="*80)
        print(f"Input directory: {mimic_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum visits per patient: {min_visits_per_patient}")
        print("="*80)
    
    def load_core_data(self):
        """Load admissions, patients, and compute basic features"""
        print("\n[1/8] Loading core admission data...")
        
        # Load admissions
        adm_path = os.path.join(self.mimic_dir, 'hosp', 'admissions.csv')
        admissions = pd.read_csv(adm_path)
        print(f"  ✓ Loaded {len(admissions):,} admissions")
        
        # Load patients
        pat_path = os.path.join(self.mimic_dir, 'hosp', 'patients.csv')
        patients = pd.read_csv(pat_path)
        print(f"  ✓ Loaded {len(patients):,} patients")
        
        # Merge
        admissions = admissions.merge(patients, on='subject_id', how='left')
        
        # Parse dates
        admissions['admittime'] = pd.to_datetime(admissions['admittime'])
        admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
        admissions['dod'] = pd.to_datetime(admissions['dod'])
        
        # Calculate length of stay (days, not hours)
        admissions['los_days'] = (
            admissions['dischtime'] - admissions['admittime']
        ).dt.total_seconds() / 86400
        
        # Calculate age at admission
        admissions['anchor_year'] = pd.to_datetime(admissions['anchor_year'], format='%Y')
        # Extract year range start
        admissions['anchor_year_start'] = admissions['anchor_year_group'].str.split(' - ').str[0].astype(int)
        admissions['age'] = admissions['anchor_year'].dt.year - admissions['anchor_year_start']
        
        # 30-day mortality
        admissions['death_30d'] = (
            (admissions['dod'].notna()) & 
            ((admissions['dod'] - admissions['dischtime']).dt.days <= 30)
        ).astype(int)
        
        # 1-year mortality
        admissions['death_1yr'] = (
            (admissions['dod'].notna()) & 
            ((admissions['dod'] - admissions['dischtime']).dt.days <= 365)
        ).astype(int)
        
        # Sort by patient and time
        admissions = admissions.sort_values(['subject_id', 'admittime'])
        
        # Calculate days since last admission (for each patient)
        admissions['days_since_last_admission'] = admissions.groupby('subject_id')['admittime'].diff().dt.days
        admissions['days_since_last_admission'] = admissions['days_since_last_admission'].fillna(0)
        
        # Visit number for each patient
        admissions['visit_number'] = admissions.groupby('subject_id').cumcount() + 1
        
        print(f"  ✓ Computed temporal features")
        print(f"  ✓ Mean LOS: {admissions['los_days'].mean():.1f} days")
        print(f"  ✓ 30-day mortality: {admissions['death_30d'].mean()*100:.2f}%")
        
        return admissions
    
    def load_diagnoses(self):
        """Load and encode diagnosis codes"""
        print("\n[2/8] Loading diagnosis codes...")
        
        diag_path = os.path.join(self.mimic_dir, 'hosp', 'diagnoses_icd.csv')
        diagnoses = pd.read_csv(diag_path)
        
        print(f"  ✓ Loaded {len(diagnoses):,} diagnosis records")
        
        # Get top N most common diagnoses
        top_diagnoses = diagnoses['icd_code'].value_counts().head(100).index.tolist()
        print(f"  ✓ Selected top 100 most common ICD codes")
        
        # Create binary indicators per admission
        diag_features = []
        for hadm_id, group in tqdm(diagnoses.groupby('hadm_id'), desc="  Processing diagnoses"):
            codes = group['icd_code'].values
            features = {f'diag_{code}': 1 for code in codes if code in top_diagnoses}
            features['hadm_id'] = hadm_id
            features['n_diagnoses'] = len(group)  # Total number of diagnoses
            diag_features.append(features)
        
        diag_df = pd.DataFrame(diag_features)
        
        # Fill missing with 0
        for code in top_diagnoses:
            col = f'diag_{code}'
            if col not in diag_df.columns:
                diag_df[col] = 0
            else:
                diag_df[col] = diag_df[col].fillna(0)
        
        print(f"  ✓ Created {len([c for c in diag_df.columns if c.startswith('diag_')])} diagnosis features")
        
        return diag_df
    
    def load_procedures(self):
        """Load and encode procedure codes"""
        print("\n[3/8] Loading procedure codes...")
        
        proc_path = os.path.join(self.mimic_dir, 'hosp', 'procedures_icd.csv')
        if not os.path.exists(proc_path):
            print("  ⚠ Procedures file not found, skipping...")
            return pd.DataFrame({'hadm_id': []})
        
        procedures = pd.read_csv(proc_path)
        print(f"  ✓ Loaded {len(procedures):,} procedure records")
        
        # Get top N most common procedures
        top_procedures = procedures['icd_code'].value_counts().head(50).index.tolist()
        print(f"  ✓ Selected top 50 most common procedure codes")
        
        # Create binary indicators per admission
        proc_features = []
        for hadm_id, group in tqdm(procedures.groupby('hadm_id'), desc="  Processing procedures"):
            codes = group['icd_code'].values
            features = {f'proc_{code}': 1 for code in codes if code in top_procedures}
            features['hadm_id'] = hadm_id
            features['n_procedures'] = len(group)
            proc_features.append(features)
        
        proc_df = pd.DataFrame(proc_features)
        
        # Fill missing with 0
        for code in top_procedures:
            col = f'proc_{code}'
            if col not in proc_df.columns:
                proc_df[col] = 0
            else:
                proc_df[col] = proc_df[col].fillna(0)
        
        print(f"  ✓ Created {len([c for c in proc_df.columns if c.startswith('proc_')])} procedure features")
        
        return proc_df
    
    def aggregate_vitals(self, sample_frac=0.3):
        """
        Aggregate vital signs PER ADMISSION (not hourly)
        Summary statistics: mean, std, min, max
        """
        print(f"\n[4/8] Aggregating vital signs per admission (sampling {sample_frac*100}%)...")
        
        chart_path = os.path.join(self.mimic_dir, 'icu', 'chartevents.csv')
        if not os.path.exists(chart_path):
            print("  ⚠ Chartevents file not found, skipping vitals...")
            return pd.DataFrame({'hadm_id': []})
        
        # Vital sign item IDs
        vital_items = {
            220045: 'heart_rate',
            220050: 'sbp',
            220051: 'dbp',
            220052: 'mbp',
            220179: 'sbp_ni',
            220180: 'dbp_ni',
            220277: 'spo2',
            223761: 'temperature',
            220210: 'respiratory_rate',
        }
        
        # Load in chunks and aggregate
        chunks = []
        chunk_size = 1_000_000
        
        for i, chunk in enumerate(pd.read_csv(chart_path, chunksize=chunk_size,
                                              usecols=['hadm_id', 'itemid', 'valuenum'])):
            # Filter for vital signs
            chunk = chunk[chunk['itemid'].isin(vital_items.keys())]
            chunk = chunk[chunk['valuenum'].notna()]
            
            # Sample
            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            
            if len(chunk) > 0:
                chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {(i+1)*chunk_size:,} rows...")
            
            # Memory management
            if len(chunks) > 30:
                chunks = [pd.concat(chunks, ignore_index=True)]
        
        if not chunks:
            print("  ⚠ No vital signs found")
            return pd.DataFrame({'hadm_id': []})
        
        chartevents = pd.concat(chunks, ignore_index=True)
        chartevents['feature'] = chartevents['itemid'].map(vital_items)
        
        print(f"  ✓ Loaded {len(chartevents):,} vital sign measurements")
        
        # Aggregate per admission
        print("  Aggregating to admission level...")
        vital_agg = chartevents.groupby(['hadm_id', 'feature'])['valuenum'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Pivot to wide format
        vital_features = []
        for hadm_id, group in vital_agg.groupby('hadm_id'):
            features = {'hadm_id': hadm_id}
            for _, row in group.iterrows():
                feat = row['feature']
                features[f'vital_{feat}_mean'] = row['mean']
                features[f'vital_{feat}_std'] = row['std']
                features[f'vital_{feat}_min'] = row['min']
                features[f'vital_{feat}_max'] = row['max']
            vital_features.append(features)
        
        vital_df = pd.DataFrame(vital_features)
        
        print(f"  ✓ Created {len([c for c in vital_df.columns if c.startswith('vital_')])} vital features")
        
        return vital_df
    
    def aggregate_labs(self, sample_frac=0.3):
        """
        Aggregate lab values PER ADMISSION
        Summary statistics: mean, std, min, max, trend
        """
        print(f"\n[5/8] Aggregating lab values per admission (sampling {sample_frac*100}%)...")
        
        lab_path = os.path.join(self.mimic_dir, 'hosp', 'labevents.csv')
        if not os.path.exists(lab_path):
            print("  ⚠ Labevents file not found, skipping labs...")
            return pd.DataFrame({'hadm_id': []})
        
        # Important lab test item IDs
        lab_items = {
            50912: 'creatinine',
            50971: 'potassium',
            50983: 'sodium',
            50902: 'chloride',
            50882: 'bicarbonate',
            51006: 'bun',
            50931: 'glucose',
            51221: 'hematocrit',
            51222: 'hemoglobin',
            51265: 'platelet',
            51301: 'wbc',
            50861: 'albumin',
            50878: 'ast',
            50885: 'bilirubin',
            51279: 'rbc',
        }
        
        # Load in chunks and aggregate
        chunks = []
        chunk_size = 1_000_000
        
        for i, chunk in enumerate(pd.read_csv(lab_path, chunksize=chunk_size,
                                              usecols=['hadm_id', 'itemid', 'valuenum'])):
            # Filter for selected labs
            chunk = chunk[chunk['itemid'].isin(lab_items.keys())]
            chunk = chunk[chunk['valuenum'].notna()]
            chunk = chunk[chunk['valuenum'] > 0]
            
            # Sample
            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            
            if len(chunk) > 0:
                chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {(i+1)*chunk_size:,} rows...")
            
            # Memory management
            if len(chunks) > 30:
                chunks = [pd.concat(chunks, ignore_index=True)]
        
        if not chunks:
            print("  ⚠ No lab values found")
            return pd.DataFrame({'hadm_id': []})
        
        labevents = pd.concat(chunks, ignore_index=True)
        labevents['feature'] = labevents['itemid'].map(lab_items)
        
        print(f"  ✓ Loaded {len(labevents):,} lab measurements")
        
        # Aggregate per admission
        print("  Aggregating to admission level...")
        lab_agg = labevents.groupby(['hadm_id', 'feature'])['valuenum'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Pivot to wide format
        lab_features = []
        for hadm_id, group in lab_agg.groupby('hadm_id'):
            features = {'hadm_id': hadm_id}
            for _, row in group.iterrows():
                feat = row['feature']
                features[f'lab_{feat}_mean'] = row['mean']
                features[f'lab_{feat}_std'] = row['std']
                features[f'lab_{feat}_min'] = row['min']
                features[f'lab_{feat}_max'] = row['max']
            lab_features.append(features)
        
        lab_df = pd.DataFrame(lab_features)
        
        print(f"  ✓ Created {len([c for c in lab_df.columns if c.startswith('lab_')])} lab features")
        
        return lab_df
    
    def merge_all_features(self, admissions, diag_df, proc_df, vital_df, lab_df):
        """Merge all feature sets into single visit-level dataframe"""
        print("\n[6/8] Merging all features...")
        
        # Start with admissions
        visits = admissions.copy()
        print(f"  Starting with {len(visits):,} admissions")
        
        # Merge diagnoses
        if len(diag_df) > 0:
            visits = visits.merge(diag_df, on='hadm_id', how='left')
            print(f"  ✓ Merged diagnoses")
        
        # Merge procedures
        if len(proc_df) > 0:
            visits = visits.merge(proc_df, on='hadm_id', how='left')
            print(f"  ✓ Merged procedures")
        
        # Merge vitals
        if len(vital_df) > 0:
            visits = visits.merge(vital_df, on='hadm_id', how='left')
            print(f"  ✓ Merged vitals")
        
        # Merge labs
        if len(lab_df) > 0:
            visits = visits.merge(lab_df, on='hadm_id', how='left')
            print(f"  ✓ Merged labs")
        
        # Fill missing values
        # For binary features (diagnoses, procedures): fill with 0
        for col in visits.columns:
            if col.startswith('diag_') or col.startswith('proc_'):
                visits[col] = visits[col].fillna(0)
        
        # For continuous features (vitals, labs): fill with median
        for col in visits.columns:
            if col.startswith('vital_') or col.startswith('lab_'):
                visits[col] = visits[col].fillna(visits[col].median())
        
        print(f"  ✓ Final dataset: {len(visits):,} visits")
        print(f"  ✓ Total features: {len(visits.columns)}")
        
        return visits
    
    def create_patient_sequences(self, visits):
        """
        Create sequences of visits per patient
        Each sequence = N consecutive visits for one patient
        """
        print(f"\n[7/8] Creating patient visit sequences (min {self.min_visits_per_patient} visits)...")
        
        # Filter patients with enough visits
        patient_visit_counts = visits.groupby('subject_id').size()
        valid_patients = patient_visit_counts[
            patient_visit_counts >= self.min_visits_per_patient
        ].index
        
        visits = visits[visits['subject_id'].isin(valid_patients)]
        
        print(f"  ✓ {len(valid_patients):,} patients with ≥{self.min_visits_per_patient} visits")
        print(f"  ✓ {len(visits):,} total visits for these patients")
        
        # Sort by patient and time
        visits = visits.sort_values(['subject_id', 'admittime'])
        
        # Calculate statistics
        visits_per_patient = visits.groupby('subject_id').size()
        print(f"  ✓ Mean visits per patient: {visits_per_patient.mean():.1f}")
        print(f"  ✓ Median visits per patient: {visits_per_patient.median():.0f}")
        print(f"  ✓ Max visits per patient: {visits_per_patient.max():.0f}")
        
        return visits
    
    def save_processed_data(self, visits):
        """Save processed visit-level data"""
        print(f"\n[8/8] Saving processed data to {self.output_dir}...")
        
        # Save main file
        output_file = os.path.join(self.output_dir, 'mimic_visit_sequences.csv')
        visits.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
        
        # Calculate and save statistics
        feature_cols = [c for c in visits.columns if c.startswith(('diag_', 'proc_', 'vital_', 'lab_'))]
        
        stats = {
            'preprocessing_info': {
                'date': datetime.now().isoformat(),
                'level': 'visit',  # vs 'hour'
                'min_visits_per_patient': self.min_visits_per_patient,
            },
            'data_stats': {
                'n_patients': int(visits['subject_id'].nunique()),
                'n_visits': int(len(visits)),
                'avg_visits_per_patient': float(visits.groupby('subject_id').size().mean()),
                'median_visits_per_patient': float(visits.groupby('subject_id').size().median()),
                'n_features': len(feature_cols),
            },
            'feature_breakdown': {
                'diagnosis_features': len([c for c in feature_cols if c.startswith('diag_')]),
                'procedure_features': len([c for c in feature_cols if c.startswith('proc_')]),
                'vital_features': len([c for c in feature_cols if c.startswith('vital_')]),
                'lab_features': len([c for c in feature_cols if c.startswith('lab_')]),
            },
            'temporal_stats': {
                'mean_days_between_visits': float(visits['days_since_last_admission'].mean()),
                'median_days_between_visits': float(visits['days_since_last_admission'].median()),
                'mean_los_days': float(visits['los_days'].mean()),
            },
            'outcome_rates': {
                'mortality_30d': float(visits.groupby('subject_id')['death_30d'].first().mean()),
                'mortality_1yr': float(visits.groupby('subject_id')['death_1yr'].first().mean()),
            },
        }
        
        stats_file = os.path.join(self.output_dir, 'visit_preprocessing_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved: {stats_file}")
        
        # Save feature statistics
        feature_stats = visits[feature_cols].describe()
        feature_stats_file = os.path.join(self.output_dir, 'visit_feature_statistics.csv')
        feature_stats.to_csv(feature_stats_file)
        print(f"  ✓ Saved: {feature_stats_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("VISIT-LEVEL PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"Total patients: {stats['data_stats']['n_patients']:,}")
        print(f"Total visits: {stats['data_stats']['n_visits']:,}")
        print(f"Avg visits/patient: {stats['data_stats']['avg_visits_per_patient']:.1f}")
        print(f"Total features: {stats['data_stats']['n_features']}")
        print(f"  - Diagnoses: {stats['feature_breakdown']['diagnosis_features']}")
        print(f"  - Procedures: {stats['feature_breakdown']['procedure_features']}")
        print(f"  - Vitals: {stats['feature_breakdown']['vital_features']}")
        print(f"  - Labs: {stats['feature_breakdown']['lab_features']}")
        print(f"Mean days between visits: {stats['temporal_stats']['mean_days_between_visits']:.1f}")
        print(f"30-day mortality: {stats['outcome_rates']['mortality_30d']*100:.2f}%")
        print("="*80)
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MIMIC-IV for visit-level (not hourly) sequences'
    )
    
    parser.add_argument('--mimic_dir', type=str, required=True,
                       help='Path to MIMIC-IV directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--min_visits_per_patient', type=int, default=10,
                       help='Minimum visits required per patient')
    parser.add_argument('--sample_frac', type=float, default=0.3,
                       help='Fraction of chartevents/labevents to sample (for speed)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = VisitLevelPreprocessor(
        mimic_dir=args.mimic_dir,
        output_dir=args.output_dir,
        min_visits_per_patient=args.min_visits_per_patient
    )
    
    # Run preprocessing pipeline
    admissions = preprocessor.load_core_data()
    diag_df = preprocessor.load_diagnoses()
    proc_df = preprocessor.load_procedures()
    vital_df = preprocessor.aggregate_vitals(sample_frac=args.sample_frac)
    lab_df = preprocessor.aggregate_labs(sample_frac=args.sample_frac)
    
    visits = preprocessor.merge_all_features(admissions, diag_df, proc_df, vital_df, lab_df)
    visits = preprocessor.create_patient_sequences(visits)
    output_file = preprocessor.save_processed_data(visits)
    
    print("\n✅ Visit-level preprocessing complete!")
    print(f"\nNext steps:")
    print(f"1. Split data:")
    print(f"   python split_mimic_data.py --input_file {output_file} --output_dir {args.output_dir}")
    print(f"\n2. Train model:")
    print(f"   python main_ehr.py --fname configs/mimic_visits.yaml")


if __name__ == '__main__':
    main()

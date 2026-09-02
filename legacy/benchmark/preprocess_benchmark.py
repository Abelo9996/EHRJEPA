#!/usr/bin/env python3
"""
MIMIC-IV Benchmark Preprocessing for PhD-Level Evaluation

Creates standardized datasets for comparison with:
- EHRMamba, CLMBR, Med-BERT, BEHRT, TransformEHR

Features:
- Demographics (age, gender, ethnicity, insurance)
- Diagnoses (Top 100 ICD-10 codes, one-hot)
- Procedures (Top 50 CPT codes, one-hot)
- Lab values (15 common labs with mean/std/min/max)
- Vital signs (6 vitals with mean/std/min/max)
- Medications (Top 50 drug categories)
- Admission info (admission type, location, etc.)

Total: ~300-400 features per visit

Standard Benchmark Tasks:
1. In-hospital mortality
2. 30-day readmission  
3. Length of stay (3-class and 7-class)
4. Phenotyping (25 chronic conditions)
5. ICU admission prediction

Author: Abel Yagubyan
Date: February 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STANDARD BENCHMARK PHENOTYPE DEFINITIONS (MIMIC-Extract / BEHRT style)
# =============================================================================

# 25 chronic condition phenotypes based on CCS categories
PHENOTYPE_ICD_MAPPING = {
    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15'],
    'diabetes': ['E10', 'E11', 'E13', 'E14'],
    'heart_failure': ['I50', 'I110', 'I130', 'I132'],
    'copd': ['J40', 'J41', 'J42', 'J43', 'J44'],
    'ckd': ['N18', 'N19'],
    'liver_disease': ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77'],
    'stroke': ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'],
    'mi': ['I21', 'I22', 'I23', 'I24', 'I25'],
    'afib': ['I48'],
    'depression': ['F32', 'F33', 'F34'],
    'anxiety': ['F40', 'F41'],
    'obesity': ['E66'],
    'cancer': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09',
               'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
               'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'],
    'anemia': ['D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59', 
               'D60', 'D61', 'D62', 'D63', 'D64'],
    'hypothyroidism': ['E00', 'E01', 'E02', 'E03'],
    'hyperlipidemia': ['E78'],
    'osteoporosis': ['M80', 'M81'],
    'rheumatoid_arthritis': ['M05', 'M06'],
    'asthma': ['J45', 'J46'],
    'dementia': ['F00', 'F01', 'F02', 'F03', 'G30', 'G31'],
    'parkinsons': ['G20', 'G21', 'G22'],
    'epilepsy': ['G40', 'G41'],
    'pvd': ['I70', 'I71', 'I72', 'I73', 'I74', 'I77'],
    'gerd': ['K21'],
    'uti': ['N30', 'N34', 'N39']
}


class BenchmarkPreprocessor:
    """
    Create benchmark-ready datasets following MIMIC-Extract standards
    """
    
    def __init__(self, mimic_dir, output_dir, min_visits=3):
        self.mimic_dir = mimic_dir
        self.output_dir = output_dir
        self.min_visits = min_visits
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("MIMIC-IV PhD Benchmark Preprocessing")
        print("="*80)
        print(f"Input: {mimic_dir}")
        print(f"Output: {output_dir}")
        print(f"Min visits per patient: {min_visits}")
        print("="*80)
        
        # Feature counts
        self.feature_info = {
            'demographics': 0,
            'diagnoses': 0,
            'procedures': 0,
            'labs': 0,
            'vitals': 0,
            'medications': 0,
            'admission': 0
        }
    
    def run(self):
        """Execute full preprocessing pipeline"""
        # Step 1: Load and merge core data
        admissions = self._load_admissions()
        
        # Step 2: Add diagnosis features
        admissions = self._add_diagnoses(admissions)
        
        # Step 3: Add procedure features
        admissions = self._add_procedures(admissions)
        
        # Step 4: Add lab values
        admissions = self._add_labs(admissions)
        
        # Step 5: Add vital signs
        admissions = self._add_vitals(admissions)
        
        # Step 6: Add medications
        admissions = self._add_medications(admissions)
        
        # Step 7: Add phenotype labels
        admissions = self._add_phenotype_labels(admissions)
        
        # Step 8: Add outcome labels
        admissions = self._add_outcome_labels(admissions)
        
        # Step 9: Create train/val/test splits
        self._create_splits(admissions)
        
        # Step 10: Save metadata
        self._save_metadata()
        
        print("\n" + "="*80)
        print("✅ Preprocessing complete!")
        print("="*80)
        
        return admissions
    
    def _load_admissions(self):
        """Load and process admissions with demographics"""
        print("\n[1/10] Loading admissions and demographics...")
        
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
        for col in ['admittime', 'dischtime', 'deathtime', 'dod']:
            if col in admissions.columns:
                admissions[col] = pd.to_datetime(admissions[col])
        
        # Calculate features
        admissions['los_hours'] = (
            admissions['dischtime'] - admissions['admittime']
        ).dt.total_seconds() / 3600
        
        admissions['los_days'] = admissions['los_hours'] / 24
        
        # Age at admission
        admissions['age'] = admissions['anchor_age']
        
        # Gender encoding
        admissions['gender_M'] = (admissions['gender'] == 'M').astype(int)
        admissions['gender_F'] = (admissions['gender'] == 'F').astype(int)
        
        # Race encoding (top categories)
        race_dummies = pd.get_dummies(admissions['race'], prefix='race')
        race_cols = [c for c in race_dummies.columns if race_dummies[c].sum() > 1000]
        admissions = pd.concat([admissions, race_dummies[race_cols[:10]]], axis=1)
        
        # Insurance encoding
        ins_dummies = pd.get_dummies(admissions['insurance'], prefix='insurance')
        admissions = pd.concat([admissions, ins_dummies], axis=1)
        
        # Admission type encoding
        adm_type_dummies = pd.get_dummies(admissions['admission_type'], prefix='adm_type')
        admissions = pd.concat([admissions, adm_type_dummies], axis=1)
        
        # Sort by patient and time
        admissions = admissions.sort_values(['subject_id', 'admittime']).reset_index(drop=True)
        
        # Add visit number
        admissions['visit_num'] = admissions.groupby('subject_id').cumcount() + 1
        
        # Days since last admission
        admissions['days_since_last'] = admissions.groupby('subject_id')['admittime'].diff().dt.days.fillna(0)
        
        # Count demographics features
        demo_cols = ['age', 'gender_M', 'gender_F'] + [c for c in admissions.columns if c.startswith('race_')] + \
                    [c for c in admissions.columns if c.startswith('insurance_')] + \
                    [c for c in admissions.columns if c.startswith('adm_type_')] + \
                    ['visit_num', 'days_since_last', 'los_days']
        self.feature_info['demographics'] = len(demo_cols)
        print(f"  ✓ Added {len(demo_cols)} demographic features")
        
        return admissions
    
    def _add_diagnoses(self, admissions):
        """Add diagnosis code features (top 100 ICD codes)"""
        print("\n[2/10] Adding diagnosis features...")
        
        diag_path = os.path.join(self.mimic_dir, 'hosp', 'diagnoses_icd.csv')
        diagnoses = pd.read_csv(diag_path)
        print(f"  ✓ Loaded {len(diagnoses):,} diagnosis records")
        
        # Get top 100 most common codes
        top_codes = diagnoses['icd_code'].value_counts().head(100).index.tolist()
        
        # Create binary features per admission
        diag_pivot = diagnoses[diagnoses['icd_code'].isin(top_codes)].copy()
        diag_pivot['value'] = 1
        diag_pivot = diag_pivot.pivot_table(
            index='hadm_id', 
            columns='icd_code', 
            values='value',
            aggfunc='max',
            fill_value=0
        ).reset_index()
        
        # Rename columns
        diag_pivot.columns = ['hadm_id'] + [f'diag_{c}' for c in diag_pivot.columns[1:]]
        
        # Add total diagnosis count
        diag_count = diagnoses.groupby('hadm_id').size().reset_index(name='n_diagnoses')
        diag_pivot = diag_pivot.merge(diag_count, on='hadm_id', how='left')
        
        # Merge with admissions
        admissions = admissions.merge(diag_pivot, on='hadm_id', how='left')
        
        # Fill missing
        diag_cols = [c for c in admissions.columns if c.startswith('diag_')]
        admissions[diag_cols] = admissions[diag_cols].fillna(0)
        admissions['n_diagnoses'] = admissions['n_diagnoses'].fillna(0)
        
        self.feature_info['diagnoses'] = len(diag_cols) + 1
        print(f"  ✓ Added {len(diag_cols) + 1} diagnosis features")
        
        return admissions
    
    def _add_procedures(self, admissions):
        """Add procedure code features (top 50 CPT/ICD procedure codes)"""
        print("\n[3/10] Adding procedure features...")
        
        proc_path = os.path.join(self.mimic_dir, 'hosp', 'procedures_icd.csv')
        if not os.path.exists(proc_path):
            print("  ⚠ Procedures file not found, skipping...")
            self.feature_info['procedures'] = 0
            return admissions
            
        procedures = pd.read_csv(proc_path)
        print(f"  ✓ Loaded {len(procedures):,} procedure records")
        
        # Get top 50 most common codes
        top_codes = procedures['icd_code'].value_counts().head(50).index.tolist()
        
        # Create binary features per admission
        proc_pivot = procedures[procedures['icd_code'].isin(top_codes)].copy()
        proc_pivot['value'] = 1
        proc_pivot = proc_pivot.pivot_table(
            index='hadm_id',
            columns='icd_code',
            values='value',
            aggfunc='max',
            fill_value=0
        ).reset_index()
        
        proc_pivot.columns = ['hadm_id'] + [f'proc_{c}' for c in proc_pivot.columns[1:]]
        
        # Add total procedure count
        proc_count = procedures.groupby('hadm_id').size().reset_index(name='n_procedures')
        proc_pivot = proc_pivot.merge(proc_count, on='hadm_id', how='left')
        
        # Merge
        admissions = admissions.merge(proc_pivot, on='hadm_id', how='left')
        
        proc_cols = [c for c in admissions.columns if c.startswith('proc_')]
        admissions[proc_cols] = admissions[proc_cols].fillna(0)
        admissions['n_procedures'] = admissions['n_procedures'].fillna(0)
        
        self.feature_info['procedures'] = len(proc_cols) + 1
        print(f"  ✓ Added {len(proc_cols) + 1} procedure features")
        
        return admissions
    
    def _add_labs(self, admissions):
        """Add aggregated lab value features"""
        print("\n[4/10] Adding lab value features...")
        
        lab_path = os.path.join(self.mimic_dir, 'hosp', 'labevents.csv')
        if not os.path.exists(lab_path):
            print("  ⚠ Lab events file not found, skipping...")
            self.feature_info['labs'] = 0
            return admissions
        
        # Common labs to extract
        lab_items = {
            'glucose': [50809, 50931],
            'creatinine': [50912],
            'bun': [51006],
            'sodium': [50824, 50983],
            'potassium': [50822, 50971],
            'chloride': [50806, 50902],
            'bicarbonate': [50803, 50882],
            'hemoglobin': [51222],
            'hematocrit': [51221],
            'wbc': [51300, 51301],
            'platelets': [51265],
            'lactate': [50813],
            'bilirubin': [50885],
            'albumin': [50862],
            'troponin': [51002, 51003]
        }
        
        all_item_ids = [item for items in lab_items.values() for item in items]
        
        # Load labs in chunks (large file)
        print("  Loading lab events (this may take a while)...")
        lab_features = {}
        
        chunk_size = 1000000
        for chunk in tqdm(pd.read_csv(lab_path, chunksize=chunk_size, 
                                       usecols=['hadm_id', 'itemid', 'valuenum']),
                          desc="  Processing labs"):
            chunk = chunk[chunk['itemid'].isin(all_item_ids)]
            chunk = chunk.dropna(subset=['valuenum', 'hadm_id'])
            
            for hadm_id, group in chunk.groupby('hadm_id'):
                if hadm_id not in lab_features:
                    lab_features[hadm_id] = defaultdict(list)
                
                for lab_name, item_ids in lab_items.items():
                    values = group[group['itemid'].isin(item_ids)]['valuenum'].values
                    if len(values) > 0:
                        lab_features[hadm_id][lab_name].extend(values.tolist())
        
        # Create aggregated features
        lab_rows = []
        for hadm_id, labs in tqdm(lab_features.items(), desc="  Aggregating"):
            row = {'hadm_id': hadm_id}
            for lab_name in lab_items.keys():
                values = labs.get(lab_name, [])
                if len(values) > 0:
                    arr = np.array(values)
                    row[f'lab_{lab_name}_mean'] = np.mean(arr)
                    row[f'lab_{lab_name}_std'] = np.std(arr)
                    row[f'lab_{lab_name}_min'] = np.min(arr)
                    row[f'lab_{lab_name}_max'] = np.max(arr)
                else:
                    row[f'lab_{lab_name}_mean'] = np.nan
                    row[f'lab_{lab_name}_std'] = np.nan
                    row[f'lab_{lab_name}_min'] = np.nan
                    row[f'lab_{lab_name}_max'] = np.nan
            lab_rows.append(row)
        
        lab_df = pd.DataFrame(lab_rows)
        admissions = admissions.merge(lab_df, on='hadm_id', how='left')
        
        lab_cols = [c for c in admissions.columns if c.startswith('lab_')]
        self.feature_info['labs'] = len(lab_cols)
        print(f"  ✓ Added {len(lab_cols)} lab features")
        
        return admissions
    
    def _add_vitals(self, admissions):
        """Add vital sign features from chartevents"""
        print("\n[5/10] Adding vital sign features...")
        
        chart_path = os.path.join(self.mimic_dir, 'icu', 'chartevents.csv')
        if not os.path.exists(chart_path):
            print("  ⚠ Chart events file not found, skipping...")
            self.feature_info['vitals'] = 0
            return admissions
        
        # Common vitals itemids
        vital_items = {
            'heart_rate': [220045],
            'sbp': [220050, 220179],
            'dbp': [220051, 220180],
            'resp_rate': [220210, 224690],
            'spo2': [220277],
            'temperature': [223761, 223762]
        }
        
        all_item_ids = [item for items in vital_items.values() for item in items]
        
        print("  Loading chart events (this may take a while)...")
        vital_features = {}
        
        chunk_size = 2000000
        for chunk in tqdm(pd.read_csv(chart_path, chunksize=chunk_size,
                                       usecols=['hadm_id', 'itemid', 'valuenum']),
                          desc="  Processing vitals"):
            chunk = chunk[chunk['itemid'].isin(all_item_ids)]
            chunk = chunk.dropna(subset=['valuenum', 'hadm_id'])
            
            for hadm_id, group in chunk.groupby('hadm_id'):
                if hadm_id not in vital_features:
                    vital_features[hadm_id] = defaultdict(list)
                
                for vital_name, item_ids in vital_items.items():
                    values = group[group['itemid'].isin(item_ids)]['valuenum'].values
                    if len(values) > 0:
                        vital_features[hadm_id][vital_name].extend(values.tolist())
        
        # Aggregate
        vital_rows = []
        for hadm_id, vitals in tqdm(vital_features.items(), desc="  Aggregating"):
            row = {'hadm_id': hadm_id}
            for vital_name in vital_items.keys():
                values = vitals.get(vital_name, [])
                if len(values) > 0:
                    arr = np.array(values)
                    row[f'vital_{vital_name}_mean'] = np.mean(arr)
                    row[f'vital_{vital_name}_std'] = np.std(arr)
                    row[f'vital_{vital_name}_min'] = np.min(arr)
                    row[f'vital_{vital_name}_max'] = np.max(arr)
                else:
                    row[f'vital_{vital_name}_mean'] = np.nan
                    row[f'vital_{vital_name}_std'] = np.nan
                    row[f'vital_{vital_name}_min'] = np.nan
                    row[f'vital_{vital_name}_max'] = np.nan
            vital_rows.append(row)
        
        vital_df = pd.DataFrame(vital_rows)
        admissions = admissions.merge(vital_df, on='hadm_id', how='left')
        
        vital_cols = [c for c in admissions.columns if c.startswith('vital_')]
        self.feature_info['vitals'] = len(vital_cols)
        print(f"  ✓ Added {len(vital_cols)} vital features")
        
        return admissions
    
    def _add_medications(self, admissions):
        """Add medication features (top 50 drug categories)"""
        print("\n[6/10] Adding medication features...")
        
        # Try different medication files
        med_paths = [
            os.path.join(self.mimic_dir, 'hosp', 'prescriptions.csv'),
            os.path.join(self.mimic_dir, 'hosp', 'emar.csv'),
            os.path.join(self.mimic_dir, 'icu', 'inputevents.csv')
        ]
        
        med_path = None
        for p in med_paths:
            if os.path.exists(p):
                med_path = p
                break
        
        if med_path is None:
            print("  ⚠ Medication file not found, skipping...")
            self.feature_info['medications'] = 0
            return admissions
        
        print(f"  Loading medications from {os.path.basename(med_path)}...")
        
        if 'prescriptions' in med_path:
            meds = pd.read_csv(med_path, usecols=['hadm_id', 'drug'])
            meds = meds.dropna()
            
            # Get top 50 drugs
            top_drugs = meds['drug'].value_counts().head(50).index.tolist()
            
            # Create binary features
            med_pivot = meds[meds['drug'].isin(top_drugs)].copy()
            med_pivot['value'] = 1
            med_pivot = med_pivot.drop_duplicates(['hadm_id', 'drug'])
            med_pivot = med_pivot.pivot(
                index='hadm_id',
                columns='drug',
                values='value'
            ).fillna(0).reset_index()
            
            med_pivot.columns = ['hadm_id'] + [f'med_{c[:30]}' for c in med_pivot.columns[1:]]
            
            # Add medication count
            med_count = meds.groupby('hadm_id').size().reset_index(name='n_medications')
            med_pivot = med_pivot.merge(med_count, on='hadm_id', how='left')
            
            admissions = admissions.merge(med_pivot, on='hadm_id', how='left')
            
            med_cols = [c for c in admissions.columns if c.startswith('med_')]
            admissions[med_cols] = admissions[med_cols].fillna(0)
            admissions['n_medications'] = admissions['n_medications'].fillna(0)
            
            self.feature_info['medications'] = len(med_cols) + 1
            print(f"  ✓ Added {len(med_cols) + 1} medication features")
        else:
            self.feature_info['medications'] = 0
            print("  ⚠ Unsupported medication format, skipping...")
        
        return admissions
    
    def _add_phenotype_labels(self, admissions):
        """Add 25 chronic condition phenotype labels"""
        print("\n[7/10] Adding phenotype labels...")
        
        diag_path = os.path.join(self.mimic_dir, 'hosp', 'diagnoses_icd.csv')
        diagnoses = pd.read_csv(diag_path)
        
        # For each phenotype, check if any matching ICD code exists
        phenotype_labels = defaultdict(lambda: defaultdict(int))
        
        for _, row in tqdm(diagnoses.iterrows(), total=len(diagnoses), desc="  Processing"):
            hadm_id = row['hadm_id']
            icd_code = str(row['icd_code'])
            
            for phenotype, prefixes in PHENOTYPE_ICD_MAPPING.items():
                for prefix in prefixes:
                    if icd_code.startswith(prefix):
                        phenotype_labels[hadm_id][f'pheno_{phenotype}'] = 1
                        break
        
        # Convert to DataFrame
        pheno_rows = []
        for hadm_id in admissions['hadm_id'].unique():
            row = {'hadm_id': hadm_id}
            for phenotype in PHENOTYPE_ICD_MAPPING.keys():
                row[f'pheno_{phenotype}'] = phenotype_labels[hadm_id].get(f'pheno_{phenotype}', 0)
            pheno_rows.append(row)
        
        pheno_df = pd.DataFrame(pheno_rows)
        admissions = admissions.merge(pheno_df, on='hadm_id', how='left')
        
        pheno_cols = [c for c in admissions.columns if c.startswith('pheno_')]
        admissions[pheno_cols] = admissions[pheno_cols].fillna(0)
        
        print(f"  ✓ Added {len(pheno_cols)} phenotype labels")
        
        # Print phenotype prevalence
        print("\n  Phenotype prevalence:")
        for col in pheno_cols[:5]:
            prev = admissions[col].mean() * 100
            print(f"    {col}: {prev:.2f}%")
        print(f"    ... and {len(pheno_cols)-5} more")
        
        return admissions
    
    def _add_outcome_labels(self, admissions):
        """Add outcome labels for benchmark tasks"""
        print("\n[8/10] Adding outcome labels...")
        
        # In-hospital mortality
        admissions['label_mortality'] = admissions['hospital_expire_flag'].astype(int)
        
        # 30-day mortality
        admissions['label_mortality_30d'] = (
            (admissions['dod'].notna()) &
            ((admissions['dod'] - admissions['dischtime']).dt.days <= 30)
        ).astype(int)
        
        # 30-day readmission
        admissions = admissions.sort_values(['subject_id', 'admittime']).reset_index(drop=True)
        admissions['next_admit'] = admissions.groupby('subject_id')['admittime'].shift(-1)
        admissions['label_readmission_30d'] = (
            (admissions['next_admit'].notna()) &
            ((admissions['next_admit'] - admissions['dischtime']).dt.days <= 30)
        ).astype(int)
        
        # Length of stay (3-class)
        admissions['label_los_3class'] = pd.cut(
            admissions['los_days'],
            bins=[-np.inf, 3, 7, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        # Length of stay (7-class)
        admissions['label_los_7class'] = pd.cut(
            admissions['los_days'],
            bins=[-np.inf, 1, 2, 3, 4, 7, 14, np.inf],
            labels=[0, 1, 2, 3, 4, 5, 6]
        ).astype(int)
        
        # ICU admission (for patients with ICU stays)
        icu_path = os.path.join(self.mimic_dir, 'icu', 'icustays.csv')
        if os.path.exists(icu_path):
            icu_stays = pd.read_csv(icu_path, usecols=['hadm_id']).drop_duplicates()
            icu_stays['label_icu'] = 1
            admissions = admissions.merge(icu_stays, on='hadm_id', how='left')
            admissions['label_icu'] = admissions['label_icu'].fillna(0).astype(int)
        else:
            admissions['label_icu'] = 0
        
        # Print outcome statistics
        print("\n  Outcome label statistics:")
        print(f"    Mortality: {admissions['label_mortality'].mean()*100:.2f}%")
        print(f"    30-day mortality: {admissions['label_mortality_30d'].mean()*100:.2f}%")
        print(f"    30-day readmission: {admissions['label_readmission_30d'].mean()*100:.2f}%")
        print(f"    ICU admission: {admissions['label_icu'].mean()*100:.2f}%")
        print(f"    LOS distribution: {admissions['label_los_3class'].value_counts().to_dict()}")
        
        return admissions
    
    def _create_splits(self, admissions):
        """Create patient-stratified train/val/test splits"""
        print("\n[9/10] Creating train/val/test splits...")
        
        # Filter patients with minimum visits
        patient_counts = admissions.groupby('subject_id').size()
        valid_patients = patient_counts[patient_counts >= self.min_visits].index.tolist()
        print(f"  Patients with >= {self.min_visits} visits: {len(valid_patients):,}")
        
        # Patient-level split (80/10/10)
        np.random.seed(42)
        np.random.shuffle(valid_patients)
        
        n_train = int(0.8 * len(valid_patients))
        n_val = int(0.1 * len(valid_patients))
        
        train_patients = valid_patients[:n_train]
        val_patients = valid_patients[n_train:n_train+n_val]
        test_patients = valid_patients[n_train+n_val:]
        
        # Create splits
        train_df = admissions[admissions['subject_id'].isin(train_patients)].copy()
        val_df = admissions[admissions['subject_id'].isin(val_patients)].copy()
        test_df = admissions[admissions['subject_id'].isin(test_patients)].copy()
        
        print(f"  Train: {len(train_df):,} admissions, {len(train_patients):,} patients")
        print(f"  Val: {len(val_df):,} admissions, {len(val_patients):,} patients")
        print(f"  Test: {len(test_df):,} admissions, {len(test_patients):,} patients")
        
        # Identify feature columns (exclude labels, IDs, dates)
        exclude_patterns = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 
                           'deathtime', 'dod', 'label_', 'pheno_', 'next_admit',
                           'gender', 'race', 'insurance', 'admission_type',
                           'admission_location', 'discharge_location', 'marital_status',
                           'language', 'anchor_year', 'anchor_year_group']
        
        feature_cols = [c for c in train_df.columns 
                       if not any(p in c for p in exclude_patterns)
                       and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        print(f"  Feature columns: {len(feature_cols)}")
        
        # Compute normalization statistics from training set
        norm_stats = {}
        for col in feature_cols:
            mean = train_df[col].mean()
            std = train_df[col].std()
            if std == 0 or np.isnan(std):
                std = 1
            norm_stats[col] = {'mean': float(mean), 'std': float(std)}
        
        # Apply normalization
        for col in feature_cols:
            mean, std = norm_stats[col]['mean'], norm_stats[col]['std']
            train_df[col] = (train_df[col].fillna(mean) - mean) / std
            val_df[col] = (val_df[col].fillna(mean) - mean) / std
            test_df[col] = (test_df[col].fillna(mean) - mean) / std
        
        # Save splits
        train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        
        # Save feature list
        with open(os.path.join(self.output_dir, 'feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f)
        
        # Save normalization statistics
        with open(os.path.join(self.output_dir, 'norm_stats.json'), 'w') as f:
            json.dump(norm_stats, f)
        
        print(f"  ✓ Saved splits to {self.output_dir}")
        
        return train_df, val_df, test_df
    
    def _save_metadata(self):
        """Save preprocessing metadata"""
        print("\n[10/10] Saving metadata...")
        
        metadata = {
            'preprocessing_date': datetime.now().isoformat(),
            'mimic_dir': self.mimic_dir,
            'min_visits': self.min_visits,
            'feature_counts': self.feature_info,
            'total_features': sum(self.feature_info.values()),
            'phenotype_labels': list(PHENOTYPE_ICD_MAPPING.keys()),
            'outcome_labels': [
                'label_mortality',
                'label_mortality_30d', 
                'label_readmission_30d',
                'label_los_3class',
                'label_los_7class',
                'label_icu'
            ]
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata")
        print(f"\n  Feature summary:")
        for category, count in self.feature_info.items():
            print(f"    {category}: {count}")
        print(f"    Total: {sum(self.feature_info.values())}")


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV Benchmark Preprocessing')
    parser.add_argument('--mimic_dir', type=str, default='./mimic-iv-2.1',
                        help='Path to MIMIC-IV directory')
    parser.add_argument('--output_dir', type=str, default='./data/benchmark',
                        help='Output directory')
    parser.add_argument('--min_visits', type=int, default=3,
                        help='Minimum visits per patient')
    args = parser.parse_args()
    
    preprocessor = BenchmarkPreprocessor(
        mimic_dir=args.mimic_dir,
        output_dir=args.output_dir,
        min_visits=args.min_visits
    )
    
    preprocessor.run()


if __name__ == '__main__':
    main()

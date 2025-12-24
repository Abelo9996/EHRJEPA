# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for EHR data from MIMIC-IV

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from logging import getLogger

logger = getLogger()


class MIMICEHRDataset(Dataset):
    """
    Dataset class for MIMIC-IV EHR data
    
    Expected data format:
    - CSV files with patient visit sequences
    - Each row represents a visit with features
    - Columns: patient_id, visit_time, [feature_columns...]
    
    The dataset loads sequences of visits per patient and creates
    fixed-length sequences for training.
    """
    
    def __init__(
        self,
        data_path,
        sequence_length=20,
        prediction_length=5,
        feature_columns=None,
        train=True,
        transform=None,
        use_cache=False
    ):
        """
        Args:
            data_path: Path to the CSV file(s) containing EHR data
            sequence_length: Number of visits in context (encoder input)
            prediction_length: Number of visits to predict
            feature_columns: List of feature column names to use
            train: Whether this is training data
            transform: Optional transform to apply to features
            use_cache: Whether to cache processed sequences in memory
        """
        super(MIMICEHRDataset, self).__init__()
        
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.train = train
        self.transform = transform
        self.use_cache = use_cache
        
        # Load and preprocess data
        logger.info(f'Loading EHR data from {data_path}')
        self.data = self._load_data(data_path)
        
        # Get feature columns
        if feature_columns is None:
            # Select only numeric columns and exclude identifiers/timestamps
            exclude_cols = ['patient_id', 'visit_time', 'hadm_id', 'subject_id', 
                          'admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime',
                          'dod', 'charttime', 'intime', 'outtime']
            
            # Get numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude identifier columns
            self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        
        self.num_features = len(self.feature_columns)
        logger.info(f'Number of features: {self.num_features}')
        
        # Create sequences
        self.sequences = self._create_sequences()
        logger.info(f'Created {len(self.sequences)} sequences')
        
        # Cache for processed sequences
        self.cache = {} if use_cache else None
    
    def _load_data(self, data_path):
        """Load EHR data from CSV file(s)"""
        if os.path.isfile(data_path):
            df = pd.read_csv(data_path)
        elif os.path.isdir(data_path):
            # Load all CSV files in directory
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            dfs = [pd.read_csv(os.path.join(data_path, f)) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"Data path {data_path} not found")
        
        # Sort by patient and time
        if 'patient_id' in df.columns and 'visit_time' in df.columns:
            df = df.sort_values(['patient_id', 'visit_time']).reset_index(drop=True)
        elif 'subject_id' in df.columns:
            # MIMIC-IV uses subject_id
            df = df.rename(columns={'subject_id': 'patient_id'})
            # Handle different time columns
            if 'charttime' in df.columns:
                df = df.rename(columns={'charttime': 'visit_time'})
            elif 'admittime' in df.columns:
                df = df.rename(columns={'admittime': 'visit_time'})
            elif 'intime' in df.columns:
                df = df.rename(columns={'intime': 'visit_time'})
            
            if 'visit_time' in df.columns:
                df = df.sort_values(['patient_id', 'visit_time']).reset_index(drop=True)
        
        return df
    
    def _create_sequences(self):
        """
        Create fixed-length sequences from patient visit data
        
        Returns:
            List of (patient_id, start_idx, end_idx) tuples
        """
        sequences = []
        total_length = self.sequence_length + self.prediction_length
        
        # Group by patient
        grouped = self.data.groupby('patient_id')
        
        for patient_id, group in grouped:
            n_visits = len(group)
            
            # Create overlapping sequences if we have enough visits
            if n_visits >= total_length:
                # Create multiple sequences per patient with stride
                stride = self.sequence_length // 2  # 50% overlap
                for start_idx in range(0, n_visits - total_length + 1, stride):
                    end_idx = start_idx + total_length
                    sequences.append((patient_id, start_idx, end_idx))
        
        return sequences
    
    def _normalize_features(self, features):
        """
        Normalize features
        Currently using simple standardization, can be extended with
        learned normalization parameters
        """
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardization (can be replaced with MinMax or other methods)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std
        
        return features
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns a sequence of visits as a tensor
        
        Returns:
            sequence: Tensor of shape (total_length, num_features)
        """
        # Check cache
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        patient_id, start_idx, end_idx = self.sequences[idx]
        
        # Get patient data
        patient_data = self.data[self.data['patient_id'] == patient_id].iloc[start_idx:end_idx]
        
        # Extract features
        features = patient_data[self.feature_columns].values.astype(np.float32)
        
        # Normalize
        features = self._normalize_features(features)
        
        # Apply transform if provided
        if self.transform is not None:
            features = self.transform(features)
        
        # Convert to tensor
        sequence = torch.from_numpy(features).float()
        
        # Cache if enabled
        if self.use_cache:
            self.cache[idx] = sequence
        
        return sequence


def make_mimic_ehr(
    data_path,
    batch_size,
    sequence_length=20,
    prediction_length=5,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    feature_columns=None
):
    """
    Create MIMIC-IV EHR data loaders
    
    Args:
        data_path: Path to EHR data
        batch_size: Batch size
        sequence_length: Number of visits in context
        prediction_length: Number of visits to predict
        collator: Optional collator for masking
        pin_mem: Whether to pin memory
        num_workers: Number of data loading workers
        world_size: Number of distributed processes
        rank: Rank of current process
        training: Whether this is for training
        drop_last: Whether to drop last incomplete batch
        feature_columns: List of feature columns to use
    
    Returns:
        dataset: EHR dataset
        data_loader: Data loader
        sampler: Distributed sampler (if world_size > 1)
    """
    
    dataset = MIMICEHRDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        feature_columns=feature_columns,
        train=training,
        use_cache=False  # Set to True for faster training with small datasets
    )
    
    logger.info(f'MIMIC-IV EHR dataset created')
    logger.info(f'  - Number of sequences: {len(dataset)}')
    logger.info(f'  - Number of features: {dataset.num_features}')
    logger.info(f'  - Sequence length: {sequence_length}')
    logger.info(f'  - Prediction length: {prediction_length}')
    
    # Create distributed sampler if needed
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=drop_last
        )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and training),
        sampler=sampler,
        collate_fn=collator,
        pin_memory=pin_mem,
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    return dataset, data_loader, sampler

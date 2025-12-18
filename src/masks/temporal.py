# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for temporal EHR sequences

import torch
import numpy as np
from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


class TemporalMaskCollator(object):
    """
    Temporal masking collator for EHR sequences
    
    This creates masks for:
    1. Context timesteps (encoder input) - earlier visits
    2. Target timesteps (prediction targets) - later visits to predict
    
    The masking strategy predicts future visits given past context.
    """
    
    def __init__(
        self,
        sequence_length=20,
        prediction_length=5,
        context_ratio=0.75,  # Ratio of sequence to use as context
        num_context_blocks=1,
        num_pred_blocks=1,
        allow_overlap=False,
        block_size_range=(1, 5),  # Range for contiguous block sizes
    ):
        """
        Args:
            sequence_length: Total length of input sequence
            prediction_length: Number of future timesteps to predict
            context_ratio: Ratio of sequence to use as context (rest is masked)
            num_context_blocks: Number of context blocks to sample
            num_pred_blocks: Number of prediction blocks to sample
            allow_overlap: Whether to allow overlap between context and targets
            block_size_range: (min, max) size for contiguous temporal blocks
        """
        super(TemporalMaskCollator, self).__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.context_ratio = context_ratio
        self.num_context_blocks = num_context_blocks
        self.num_pred_blocks = num_pred_blocks
        self.allow_overlap = allow_overlap
        self.block_size_range = block_size_range
        
        # Calculate context and prediction regions
        self.context_length = int(sequence_length * context_ratio)
        
        logger.info(f'Initialized TemporalMaskCollator')
        logger.info(f'  - Sequence length: {sequence_length}')
        logger.info(f'  - Context length: {self.context_length}')
        logger.info(f'  - Prediction length: {prediction_length}')
    
    def _sample_block_mask(self, region_length, num_blocks, block_size_range):
        """
        Sample block masks within a region
        
        Args:
            region_length: Length of the region to sample from
            num_blocks: Number of blocks to sample
            block_size_range: (min, max) block size
        
        Returns:
            List of indices in the masked region
        """
        min_block, max_block = block_size_range
        max_block = min(max_block, region_length)
        
        masks = []
        available_indices = set(range(region_length))
        
        for _ in range(num_blocks):
            if not available_indices:
                break
            
            # Sample block size
            block_size = np.random.randint(min_block, max_block + 1)
            
            # Sample start position
            max_start = max(0, region_length - block_size)
            if max_start <= 0:
                start = 0
            else:
                # Try to find a valid start position
                valid_starts = [s for s in range(max_start + 1) 
                              if any(s + i in available_indices for i in range(block_size))]
                if not valid_starts:
                    continue
                start = np.random.choice(valid_starts)
            
            # Add block indices
            block_indices = list(range(start, min(start + block_size, region_length)))
            block_indices = [idx for idx in block_indices if idx in available_indices]
            
            if block_indices:
                masks.extend(block_indices)
                # Remove sampled indices from available set
                available_indices -= set(block_indices)
        
        return sorted(masks)
    
    def _sample_context_masks(self, batch_size):
        """
        Sample context masks (encoder input)
        These are the timesteps the model sees as input
        """
        masks = []
        
        for _ in range(batch_size):
            # Sample from earlier part of sequence
            mask = self._sample_block_mask(
                region_length=self.context_length,
                num_blocks=self.num_context_blocks,
                block_size_range=self.block_size_range
            )
            
            # Ensure we have at least some context
            if len(mask) < 3:
                # Take first few timesteps if sampling failed
                mask = list(range(min(5, self.context_length)))
            
            masks.append(torch.tensor(mask, dtype=torch.long))
        
        return masks
    
    def _sample_prediction_masks(self, batch_size, context_masks=None):
        """
        Sample prediction masks (target timesteps to predict)
        These are the future timesteps the model should predict
        """
        masks = []
        
        for b in range(batch_size):
            if self.allow_overlap or context_masks is None:
                # Sample from later part of sequence
                start_region = self.context_length
                region_length = self.sequence_length - start_region
                
                mask = self._sample_block_mask(
                    region_length=region_length,
                    num_blocks=self.num_pred_blocks,
                    block_size_range=self.block_size_range
                )
                # Offset by start region
                mask = [idx + start_region for idx in mask]
            else:
                # Don't overlap with context
                context_set = set(context_masks[b].tolist())
                available = [i for i in range(self.context_length, self.sequence_length) 
                           if i not in context_set]
                
                if len(available) < self.prediction_length:
                    # Fall back to last N timesteps
                    mask = list(range(
                        self.sequence_length - self.prediction_length,
                        self.sequence_length
                    ))
                else:
                    # Sample prediction_length consecutive timesteps
                    max_start = len(available) - self.prediction_length
                    start_idx = np.random.randint(0, max_start + 1)
                    mask = available[start_idx:start_idx + self.prediction_length]
            
            # Ensure we have prediction targets
            if len(mask) < 1:
                # Predict last timesteps if sampling failed
                mask = list(range(
                    self.sequence_length - min(self.prediction_length, 3),
                    self.sequence_length
                ))
            
            masks.append(torch.tensor(mask, dtype=torch.long))
        
        return masks
    
    def __call__(self, batch):
        """
        Collate a batch of sequences and create masks
        
        Args:
            batch: List of tensors, each of shape (sequence_length, num_features)
        
        Returns:
            Tuple of:
                - Stacked batch tensor
                - Context masks (encoder input)
                - Prediction masks (decoder targets)
        """
        batch_size = len(batch)
        
        # Stack sequences
        sequences = torch.stack(batch, dim=0)  # (B, L, F)
        
        # Sample context masks
        context_masks = self._sample_context_masks(batch_size)
        
        # Sample prediction masks
        pred_masks = self._sample_prediction_masks(batch_size, context_masks)
        
        # Convert to format expected by model
        # Each mask should be a list of lists (one per sample in batch)
        context_masks_out = [[mask] for mask in context_masks]
        pred_masks_out = [[mask] for mask in pred_masks]
        
        return (sequences,), context_masks_out, pred_masks_out
    
    def step(self):
        """
        Update internal state (for compatibility with training loop)
        Can be used to implement curriculum learning on masking strategy
        """
        pass


class SimpleFuturePredictionMask(object):
    """
    Simple masking strategy: predict next N visits given previous M visits
    
    This is a straightforward temporal prediction task where:
    - Context: First M timesteps
    - Target: Next N timesteps
    """
    
    def __init__(
        self,
        sequence_length=20,
        context_length=15,
        prediction_length=5,
    ):
        """
        Args:
            sequence_length: Total sequence length
            context_length: Number of context timesteps
            prediction_length: Number of timesteps to predict
        """
        super(SimpleFuturePredictionMask, self).__init__()
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        assert context_length + prediction_length <= sequence_length, \
            "Context + Prediction length must be <= sequence length"
        
        logger.info(f'Initialized SimpleFuturePredictionMask')
        logger.info(f'  - Context: timesteps 0-{context_length-1}')
        logger.info(f'  - Predict: timesteps {context_length}-{context_length + prediction_length - 1}')
    
    def __call__(self, batch):
        """
        Create simple future prediction masks
        
        Args:
            batch: List of tensors, each of shape (sequence_length, num_features)
        
        Returns:
            Tuple of (sequences, context_masks, pred_masks)
        """
        batch_size = len(batch)
        
        # Stack sequences
        sequences = torch.stack(batch, dim=0)  # (B, L, F)
        
        # Create fixed masks
        context_mask = torch.arange(self.context_length, dtype=torch.long)
        pred_mask = torch.arange(
            self.context_length,
            self.context_length + self.prediction_length,
            dtype=torch.long
        )
        
        # Replicate for batch
        context_masks = [[context_mask] for _ in range(batch_size)]
        pred_masks = [[pred_mask] for _ in range(batch_size)]
        
        return (sequences,), context_masks, pred_masks
    
    def step(self):
        """Compatibility with training loop"""
        pass

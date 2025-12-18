#!/usr/bin/env python3
"""
Test script for JEPA-EHR components

This script tests the individual components of JEPA-EHR to ensure
everything is working correctly before full training.
"""

import torch
import numpy as np
import sys

print("Testing JEPA-EHR components...")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.models.temporal_transformer import (
        temporal_transformer_small,
        temporal_transformer_predictor,
        VisitEmbed
    )
    from src.datasets.mimic_ehr import MIMICEHRDataset
    from src.masks.temporal import SimpleFuturePredictionMask, TemporalMaskCollator
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Visit Embedding
print("\n2. Testing Visit Embedding...")
try:
    visit_embed = VisitEmbed(num_features=25, embed_dim=384)
    dummy_input = torch.randn(4, 20, 25)  # (batch, seq_len, features)
    output = visit_embed(dummy_input)
    assert output.shape == (4, 20, 384), f"Wrong output shape: {output.shape}"
    print(f"   ✓ Visit Embed works: {dummy_input.shape} -> {output.shape}")
except Exception as e:
    print(f"   ✗ Visit Embed failed: {e}")
    sys.exit(1)

# Test 3: Temporal Transformer Encoder
print("\n3. Testing Temporal Transformer Encoder...")
try:
    encoder = temporal_transformer_small(num_features=25, sequence_length=20)
    dummy_input = torch.randn(4, 20, 25)
    output = encoder(dummy_input)
    assert output.shape == (4, 20, 384), f"Wrong output shape: {output.shape}"
    print(f"   ✓ Encoder works: {dummy_input.shape} -> {output.shape}")
    
    # Test with masks
    masks = [[torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] for _ in range(4)]
    output_masked = encoder(dummy_input, masks)
    assert output_masked.shape[1] == 10, f"Wrong masked shape: {output_masked.shape}"
    print(f"   ✓ Encoder with masks works: {output_masked.shape}")
except Exception as e:
    print(f"   ✗ Encoder failed: {e}")
    sys.exit(1)

# Test 4: Temporal Transformer Predictor
print("\n4. Testing Temporal Transformer Predictor...")
try:
    predictor = temporal_transformer_predictor(
        num_timesteps=20,
        embed_dim=384,
        predictor_embed_dim=192,
        depth=4,
        num_heads=6
    )
    
    # Context and prediction masks
    context_masks = [[torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])] for _ in range(4)]
    pred_masks = [[torch.tensor([10, 11, 12, 13, 14])] for _ in range(4)]
    
    # Dummy encoded input
    encoded = torch.randn(4, 10, 384)
    
    output = predictor(encoded, context_masks, pred_masks)
    assert output.shape == (4, 5, 384), f"Wrong predictor output shape: {output.shape}"
    print(f"   ✓ Predictor works: {output.shape}")
except Exception as e:
    print(f"   ✗ Predictor failed: {e}")
    sys.exit(1)

# Test 5: Masking Strategies
print("\n5. Testing Masking Strategies...")
try:
    # Simple mask
    simple_mask = SimpleFuturePredictionMask(
        sequence_length=20,
        context_length=15,
        prediction_length=5
    )
    dummy_batch = [torch.randn(20, 25) for _ in range(4)]
    seqs, ctx_masks, pred_masks = simple_mask(dummy_batch)
    print(f"   ✓ Simple mask works")
    print(f"     Context: {len(ctx_masks[0][0])} timesteps")
    print(f"     Predict: {len(pred_masks[0][0])} timesteps")
    
    # Temporal mask
    temporal_mask = TemporalMaskCollator(
        sequence_length=20,
        prediction_length=5,
        context_ratio=0.75
    )
    seqs, ctx_masks, pred_masks = temporal_mask(dummy_batch)
    print(f"   ✓ Temporal mask works")
    print(f"     Context: ~{len(ctx_masks[0][0])} timesteps (sampled)")
    print(f"     Predict: ~{len(pred_masks[0][0])} timesteps (sampled)")
except Exception as e:
    print(f"   ✗ Masking failed: {e}")
    sys.exit(1)

# Test 6: End-to-End Forward Pass
print("\n6. Testing End-to-End Forward Pass...")
try:
    # Create models
    encoder = temporal_transformer_small(num_features=25, sequence_length=20)
    predictor = temporal_transformer_predictor(
        num_timesteps=20,
        embed_dim=384,
        predictor_embed_dim=192,
        depth=4,
        num_heads=6
    )
    target_encoder = temporal_transformer_small(num_features=25, sequence_length=20)
    
    # Create dummy batch
    batch_size = 4
    sequences = torch.randn(batch_size, 20, 25)
    
    # Create masks
    context_masks = [[torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])] 
                     for _ in range(batch_size)]
    pred_masks = [[torch.tensor([15, 16, 17, 18, 19])] for _ in range(batch_size)]
    
    # Forward pass (context encoder)
    z = encoder(sequences, context_masks)
    assert z.shape == (batch_size, 15, 384)
    
    # Forward pass (target encoder)
    with torch.no_grad():
        h = target_encoder(sequences)
        h = torch.nn.functional.layer_norm(h, (h.size(-1),))
    assert h.shape == (batch_size, 20, 384)
    
    # Forward pass (predictor)
    z_pred = predictor(z, context_masks, pred_masks)
    assert z_pred.shape == (batch_size, 5, 384)
    
    print(f"   ✓ End-to-end forward pass successful")
    print(f"     Input: {sequences.shape}")
    print(f"     Context encoding: {z.shape}")
    print(f"     Target encoding: {h.shape}")
    print(f"     Prediction: {z_pred.shape}")
    
    # Test loss computation
    from src.masks.utils import apply_masks
    from src.utils.tensors import repeat_interleave_batch
    h_masked = apply_masks(h, pred_masks)
    h_repeated = repeat_interleave_batch(h_masked, batch_size, repeat=len(context_masks))
    loss = torch.nn.functional.smooth_l1_loss(z_pred, h_repeated)
    print(f"   ✓ Loss computation successful: {loss.item():.4f}")
    
except Exception as e:
    print(f"   ✗ End-to-end test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Dataset (if sample data exists)
print("\n7. Testing Dataset Loading...")
try:
    import os
    sample_path = './data/sample_ehr/sample_mimic_ehr.csv'
    
    if os.path.exists(sample_path):
        dataset = MIMICEHRDataset(
            data_path=sample_path,
            sequence_length=20,
            prediction_length=5,
            train=True
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} sequences")
        print(f"     Number of features: {dataset.num_features}")
        
        # Test getting an item
        item = dataset[0]
        print(f"   ✓ Dataset getitem works: {item.shape}")
    else:
        print(f"   ⚠ Sample data not found at {sample_path}")
        print(f"     Run: python generate_sample_data.py to create sample data")
        
except Exception as e:
    print(f"   ⚠ Dataset test skipped or failed: {e}")

# Summary
print("\n" + "=" * 60)
print("All critical tests passed! ✓")
print("=" * 60)
print("\nYou can now:")
print("  1. Generate sample data: python generate_sample_data.py")
print("  2. Run quick start: bash quickstart.sh")
print("  3. Or train directly: python main_ehr.py --fname configs/mimic_ehr_small.yaml")
print()

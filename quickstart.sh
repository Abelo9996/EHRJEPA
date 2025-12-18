#!/bin/bash
# Quick start script for JEPA-EHR
# This script generates sample data and starts training

set -e  # Exit on error

echo "========================================="
echo "JEPA-EHR Quick Start Script"
echo "========================================="
echo ""

# Step 1: Generate sample data
echo "Step 1: Generating sample EHR data..."
python generate_sample_data.py \
    --output_dir ./data/sample_ehr \
    --num_patients 1000 \
    --min_visits 15 \
    --max_visits 50 \
    --num_vitals 10 \
    --num_labs 15 \
    --train_test_split \
    --seed 42

echo ""
echo "Step 2: Updating configuration..."

# Create a temporary config file
cat > configs/mimic_ehr_quickstart.yaml << EOF
data:
  data_path: ./data/sample_ehr/train_mimic_ehr.csv
  batch_size: 64
  sequence_length: 20
  context_length: 15
  prediction_length: 5
  feature_columns: null
  num_workers: 4
  pin_mem: true
  drop_last: true

logging:
  folder: ./logs/jepa_ehr_quickstart/
  write_tag: jepa-ehr-test

mask:
  masking_strategy: simple
  allow_overlap: false
  num_context_blocks: 1
  num_pred_blocks: 1
  block_size_range: [1, 5]
  context_ratio: 0.75

meta:
  model_name: temporal_transformer_small
  pred_depth: 4
  pred_emb_dim: 192
  load_checkpoint: false
  read_checkpoint: null
  use_bfloat16: false

optimization:
  epochs: 10
  start_lr: 0.0001
  lr: 0.0008
  final_lr: 1.0e-06
  warmup: 2
  weight_decay: 0.03
  final_weight_decay: 0.3
  ema: [0.996, 1.0]
  ipe_scale: 1.0
EOF

echo "Created quick-start configuration: configs/mimic_ehr_quickstart.yaml"
echo ""

# Step 3: Start training
echo "Step 3: Starting training (10 epochs for testing)..."
echo ""
python main_ehr.py --fname configs/mimic_ehr_quickstart.yaml

echo ""
echo "========================================="
echo "Quick start completed!"
echo "========================================="
echo ""
echo "Checkpoints saved to: ./logs/jepa_ehr_quickstart/"
echo ""
echo "Next steps:"
echo "  1. Check training logs: cat ./logs/jepa_ehr_quickstart/jepa-ehr-test_r0.csv"
echo "  2. For longer training, edit configs/mimic_ehr_quickstart.yaml and increase epochs"
echo "  3. Use the base model for better performance: configs/mimic_ehr_base.yaml"
echo ""

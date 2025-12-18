#!/bin/bash
# Complete pipeline to preprocess MIMIC-IV data and train JEPA-EHR

echo "======================================"
echo "JEPA-EHR Training Pipeline"
echo "======================================"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Step 1: Preprocess MIMIC-IV data
echo ""
echo "Step 1: Preprocessing MIMIC-IV data..."
echo "--------------------------------------"
python preprocess_mimic.py \
    --mimic_dir ./mimic-iv-2.1 \
    --output_dir ./data/processed_mimic \
    --sample_frac 0.05 \
    --min_seq_length 10 \
    --max_admissions 500

if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

# Step 2: Verify data was created
echo ""
echo "Step 2: Verifying processed data..."
echo "--------------------------------------"
if [ -f "./data/processed_mimic/mimic_hourly_sequences.csv" ]; then
    echo "✓ Processed data file found"
    wc -l ./data/processed_mimic/mimic_hourly_sequences.csv
else
    echo "ERROR: Processed data file not found!"
    exit 1
fi

# Step 3: Run training
echo ""
echo "Step 3: Starting JEPA-EHR training..."
echo "--------------------------------------"
python main_ehr.py --fname configs/mimic_ehr_base.yaml

echo ""
echo "======================================"
echo "Pipeline complete!"
echo "======================================"

#!/bin/bash
#
# EHR-JEPA PhD Benchmark Pipeline
# Complete evaluation following standards from EHRMamba, CLMBR, Med-BERT
#
# Author: Abel Yagubyan
# Date: February 2025

set -e

echo "========================================"
echo "EHR-JEPA PhD Benchmark Pipeline"
echo "========================================"

# Configuration
MIMIC_DIR="${MIMIC_DIR:-./mimic-iv-2.1}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/benchmark}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/jepa-visits_latest.pt}"
DEVICE="${DEVICE:-cuda}"
MIN_VISITS="${MIN_VISITS:-3}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mimic_dir)
            MIMIC_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --min_visits)
            MIN_VISITS="$2"
            shift 2
            ;;
        --skip_preprocessing)
            SKIP_PREPROCESSING=1
            shift
            ;;
        --skip_baselines)
            SKIP_BASELINES=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  MIMIC_DIR: $MIMIC_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  CHECKPOINT: $CHECKPOINT"
echo "  DEVICE: $DEVICE"
echo "  MIN_VISITS: $MIN_VISITS"
echo ""

# Step 1: Preprocessing
if [ -z "$SKIP_PREPROCESSING" ]; then
    echo "========================================"
    echo "Step 1: Data Preprocessing"
    echo "========================================"
    
    python benchmark/preprocess_benchmark.py \
        --mimic_dir "$MIMIC_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --min_visits "$MIN_VISITS"
else
    echo "Skipping preprocessing..."
fi

# Step 2: Train Neural Network Baselines
if [ -z "$SKIP_BASELINES" ]; then
    echo ""
    echo "========================================"
    echo "Step 2: Training Neural Network Baselines"
    echo "========================================"
    
    for TASK in mortality readmission_30d icu; do
        echo ""
        echo "--- Task: $TASK ---"
        
        python benchmark/train_baselines.py \
            --data_dir "$OUTPUT_DIR" \
            --output_dir ./checkpoints/baselines \
            --task "$TASK" \
            --model all \
            --epochs 30 \
            --batch_size 64 \
            --device "$DEVICE"
    done
else
    echo "Skipping baseline training..."
fi

# Step 3: Full Benchmark Evaluation
echo ""
echo "========================================"
echo "Step 3: Full Benchmark Evaluation"
echo "========================================"

python benchmark/evaluate_benchmark.py \
    --data_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

# Step 4: K-Fold Cross-Validation
echo ""
echo "========================================"
echo "Step 4: 5-Fold Cross-Validation"
echo "========================================"

python benchmark/evaluate_benchmark.py \
    --data_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --kfold 5

# Step 5: Generate Report
echo ""
echo "========================================"
echo "Step 5: Generating Benchmark Report"
echo "========================================"

python -c "
import json
import os

output_dir = '$OUTPUT_DIR'

# Load results
results_path = os.path.join(output_dir, 'benchmark_results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    
    print()
    print('='*80)
    print('FINAL BENCHMARK REPORT')
    print('='*80)
    print()
    
    # Compare with SOTA
    sota = {
        'mortality': {'EHRMamba': 0.89, 'CLMBR': 0.86, 'Med-BERT': 0.85},
        'readmission_30d': {'EHRMamba': 0.72, 'CLMBR': 0.70, 'Med-BERT': 0.68},
        'phenotyping': {'EHRMamba': 0.82, 'CLMBR': 0.79, 'Med-BERT': 0.77}
    }
    
    for task, task_results in results.items():
        if 'error' in task_results:
            continue
        
        print(f'Task: {task}')
        print('-'*40)
        
        for model, metrics in task_results.items():
            if isinstance(metrics, dict):
                auroc = metrics.get('auroc', metrics.get('macro_f1', 0))
                print(f'  {model}: AUROC={auroc:.4f}')
        
        if task in sota:
            print()
            print('  SOTA Comparison:')
            for model, score in sota[task].items():
                print(f'    {model}: {score:.4f}')
        print()
else:
    print('No results found!')
"

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/benchmark_results.json"
echo ""

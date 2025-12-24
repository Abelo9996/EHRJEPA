#!/bin/bash
# Run visit-level MIMIC-IV preprocessing
# For learning long-term clinical concepts (not hourly vitals)

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      MIMIC-IV Visit-Level Preprocessing                     ║"
echo "║      For Long-Term Clinical Concept Learning                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This preprocessing creates VISIT-LEVEL sequences where:"
echo "  • Each timestep = 1 complete hospital admission"
echo "  • Features = diagnoses, procedures, lab trends, vital summaries"
echo "  • Time gaps = days/weeks/months (realistic)"
echo "  • Learns = disease progression, patient phenotypes, readmissions"
echo ""
echo "vs. Hourly approach which captures:"
echo "  • Each timestep = 1 hour of ICU monitoring"
echo "  • Features = raw vital signs and labs"
echo "  • Time gaps = 1 hour"
echo "  • Learns = short-term physiological dynamics"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

# Configuration
MIMIC_DIR="./mimic-iv-2.1"
OUTPUT_DIR="./data/mimic_visits"
MIN_VISITS=10        # Minimum visits per patient
SAMPLE_FRAC=0.3      # Sample 30% of chartevents/labevents for speed

echo ""
echo "Configuration:"
echo "  MIMIC-IV directory: $MIMIC_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Min visits per patient: $MIN_VISITS"
echo "  Sampling fraction: $SAMPLE_FRAC (30%)"
echo ""

# Check if MIMIC directory exists
if [ ! -d "$MIMIC_DIR" ]; then
    echo "❌ Error: MIMIC directory not found: $MIMIC_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting preprocessing..."
echo "⏱️  Estimated time: 30-90 minutes (faster than hourly approach)"
echo ""

# Run preprocessing
python preprocess_mimic_visits.py \
    --mimic_dir "$MIMIC_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --min_visits_per_patient "$MIN_VISITS" \
    --sample_frac "$SAMPLE_FRAC"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Visit-Level Preprocessing Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check output
if [ -f "$OUTPUT_DIR/mimic_visit_sequences.csv" ]; then
    echo "✓ Output file created: $OUTPUT_DIR/mimic_visit_sequences.csv"
    
    # Count lines
    LINES=$(wc -l < "$OUTPUT_DIR/mimic_visit_sequences.csv")
    echo "  Total visits: $((LINES - 1))"
    
    # Get file size
    SIZE=$(du -h "$OUTPUT_DIR/mimic_visit_sequences.csv" | cut -f1)
    echo "  File size: $SIZE"
    
    echo ""
    echo "📊 What you can now learn:"
    echo "  ✓ Disease progression over months/years"
    echo "  ✓ Patient phenotypes and cohorts"
    echo "  ✓ Readmission patterns (30-day, 1-year)"
    echo "  ✓ Long-term clinical trajectories"
    echo "  ✓ Chronic disease evolution"
    echo ""
    echo "Next steps:"
    echo "─────────────────────────────────────────────────────────────"
    echo "1. Split data into train/val/test:"
    echo "   python split_mimic_data.py \\"
    echo "       --input_file $OUTPUT_DIR/mimic_visit_sequences.csv \\"
    echo "       --output_dir $OUTPUT_DIR"
    echo ""
    echo "2. Update config (if needed):"
    echo "   configs/mimic_visits.yaml"
    echo "   → data_path: $OUTPUT_DIR/train_mimic_ehr.csv"
    echo ""
    echo "3. Train the model:"
    echo "   python main_ehr.py --fname configs/mimic_visits.yaml"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "❌ Error: Output file not created"
    exit 1
fi

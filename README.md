# JEPA-EHR: Self-Supervised Learning for Electronic Health Records

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A self-supervised learning framework for Electronic Health Records (EHR) based on Joint-Embedding Predictive Architecture (JEPA). This implementation learns rich, temporally-aware patient representations from visit-level clinical data for downstream tasks such as readmission prediction, mortality forecasting, and disease progression modeling.

## Overview

JEPA-EHR adapts the I-JEPA (Image-based Joint-Embedding Predictive Architecture) framework to learn representations from longitudinal patient visit data. Unlike traditional supervised approaches, this method learns meaningful patient embeddings through self-supervised prediction of future visits.

### Key Features

- **Visit-Level Learning**: Operates on hospital visits/encounters rather than hourly measurements, capturing long-term clinical patterns
- **Rich Feature Engineering**: Incorporates ~250+ features per visit including:
  - ICD-10 diagnosis codes (top 100)
  - CPT procedure codes (top 50)
  - Aggregated lab values and vital signs
  - Demographics and admission details
- **Temporal Transformer Architecture**: 86.65M parameter encoder + 14.80M parameter predictor
- **Self-Supervised Pretraining**: Learns representations by predicting future visits from context
- **Flexible Downstream Tasks**: Representations can be used for readmission prediction, mortality risk, length of stay, etc.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Patient Visit Sequence (20 visits)                         │
│  [V1, V2, V3, ..., V14, V15 | V16, V17, V18, V19, V20]     │
│         Context (15)        |      Targets (5)              │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Encoder (Context)    │
        │  Temporal Transformer │
        │  768-dim embeddings   │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Target Encoder (EMA) │
        │  Process all visits   │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Predictor            │
        │  Predict target reps  │
        └───────────────────────┘
                    │
                    ▼
            [ Smooth L1 Loss ]
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Access to MIMIC-IV dataset (requires PhysioNet credentialing)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Abelo9996/EHRJEPA.git
cd EHRJEPA
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch pandas numpy PyYAML tqdm scikit-learn
# Or use the requirements file:
pip install -r requirements_ehr.txt
```

4. **Download MIMIC-IV data**:
   - Obtain access at https://physionet.org/content/mimiciv/
   - Download and extract to `./mimic-iv-2.1/`

## Data Preprocessing

### Visit-Level Preprocessing

Transform raw MIMIC-IV data into visit-level sequences:

```bash
./run_visit_preprocessing.sh
```

This script:
1. Loads admissions, diagnoses, procedures, chartevents, and labevents
2. Aggregates features per hospital visit
3. Extracts top diagnosis/procedure codes and vital/lab summaries
4. Creates patient sequences with 20 consecutive visits
5. Splits data into train/val/test sets (80/10/10)

**Output**:
- `data/mimic_visits/train_mimic_ehr.csv` - Training data
- `data/mimic_visits/val_mimic_ehr.csv` - Validation data  
- `data/mimic_visits/test_mimic_ehr.csv` - Test data
- Statistics files with feature distributions

**Configuration** (edit `preprocess_mimic_visits.py`):
```python
NUM_DIAG_CODES = 100        # Top diagnosis codes
NUM_PROC_CODES = 50         # Top procedure codes
MIN_VISITS_PER_PATIENT = 3  # Minimum visits required
```

## Training

### Quick Start

Train on visit-level data:

```bash
python main_ehr.py --fname configs/mimic_visits.yaml
```

### Configuration

Key parameters in `configs/mimic_visits.yaml`:

```yaml
data:
  batch_size: 64
  sequence_length: 20      # Total visits per sequence
  context_length: 15       # Visits used as context
  prediction_length: 5     # Visits to predict

meta:
  model_name: temporal_transformer_base  # 86M params
  pred_depth: 8                          # Predictor depth
  pred_emb_dim: 384                      # Predictor dimension

optimization:
  epochs: 100
  lr: 0.0005               # Peak learning rate
  warmup: 20               # Warmup epochs
  weight_decay: 0.05
  ema: [0.996, 1.0]       # EMA for target encoder
```

### Training Outputs

- **Checkpoints**: `logs/jepa_visits/jepa-visits-ep{N}.pth.tar`
- **Training logs**: `logs/jepa_visits/jepa-visits_r0.csv`
- **Config snapshot**: `logs/jepa_visits/params-jepa-ehr.yaml`

### Monitoring Training

```bash
# View training progress
tail -f logs/jepa_visits/jepa-visits_r0.csv

# Check loss curve
python -c "import pandas as pd; df=pd.read_csv('logs/jepa_visits/jepa-visits_r0.csv'); print(df.groupby('epoch')['loss'].mean())"
```

## Evaluation

Evaluate learned representations on downstream tasks:

```bash
python evaluate_visit_model.py
```

This script:
1. Loads the trained encoder from checkpoint
2. Extracts 768-dimensional representations for each patient sequence
3. Evaluates on multiple clinical prediction tasks using linear probes

### Downstream Tasks

**1. 30-Day Readmission Prediction**
- Predicts if patient returns within 30 days after sequence
- Binary classification task

**2. In-Hospital Mortality**
- Predicts mortality during any visit in the sequence
- High-stakes clinical prediction

**3. Future Visit Prediction**
- Predicts if patient has follow-up visits after sequence
- Proxy for patient engagement/health trajectory

### Sample Results (10 epochs)

```
Task: 30-Day Readmission
  Linear Probe:  AUROC=0.6232, AUPRC=0.5678, F1=0.5405
  Random Forest: AUROC=0.5966, AUPRC=0.5430, F1=0.4138

Task: Future Visit Prediction
  Linear Probe:  AUROC=0.6857, AUPRC=0.9167, F1=0.9167
  Random Forest: AUROC=0.6595, AUPRC=0.9251, F1=0.9211
```

*Note: These are preliminary results from early training (10/100 epochs)*

## Repository Structure

```
EHRJEPA/
├── configs/
│   ├── mimic_visits.yaml         # Visit-level training config
│   └── sample_ehr_test.yaml      # Sample/test config
├── src/
│   ├── datasets/
│   │   └── mimic_ehr.py          # EHR dataset loader
│   ├── models/
│   │   └── temporal_transformer.py  # Model architectures
│   ├── masks/
│   │   └── masking.py            # Temporal masking strategies
│   ├── helper_ehr.py             # Training utilities
│   └── train.py                  # Core training logic
├── data/                         # Data directory (gitignored)
├── logs/                         # Training logs (gitignored)
├── main_ehr.py                   # Training entry point
├── train_ehr.py                  # Main training loop
├── preprocess_mimic_visits.py    # Visit-level preprocessing
├── run_visit_preprocessing.sh    # Preprocessing script
├── split_mimic_data.py           # Train/val/test splitting
├── evaluate_visit_model.py       # Evaluation script
├── requirements_ehr.txt          # Python dependencies
└── README.md                     # This file
```

## Model Details

### Encoder Architecture
- **Type**: Temporal Transformer
- **Parameters**: 86.65M
- **Embedding dim**: 768
- **Depth**: 12 layers
- **Attention heads**: 12
- **Sequence length**: 20 visits
- **Feature input**: 257 numeric features per visit

### Predictor Architecture
- **Parameters**: 14.80M
- **Embedding dim**: 384
- **Depth**: 8 layers
- **Purpose**: Predict target visit representations from context

### Training Strategy
- **Loss**: Smooth L1 Loss (Huber loss)
- **Optimizer**: AdamW with cosine annealing
- **Target encoder**: Exponential Moving Average (τ=0.996→1.0)
- **Masking**: Simple future prediction (context: 0-14, predict: 15-19)

## Dataset Statistics

### MIMIC-IV Visit-Level Data
- **Training sequences**: 1,008
- **Validation sequences**: 203
- **Test sequences**: ~200
- **Features per visit**: 257 numeric
- **Sequence length**: 20 visits
- **Time span**: Varies per patient (days to years between visits)

### Feature Breakdown
- **Diagnosis codes**: 100 one-hot ICD-10 codes
- **Procedure codes**: 50 one-hot CPT codes
- **Demographics**: Age, gender, race, insurance (10 features)
- **Aggregated vitals**: Heart rate, BP, temp, SpO2 (means, stds, counts)
- **Aggregated labs**: WBC, hemoglobin, glucose, creatinine (means, stds, counts)
- **Admission info**: Admission type, location, discharge location
- **Temporal**: Length of stay, time to death (if applicable)

## Citation

If you use this code for your research, please cite:

```bibtex
@software{ehrjepa2024,
  author = {Yagubyan, Abel},
  title = {JEPA-EHR: Self-Supervised Learning for Electronic Health Records},
  year = {2024},
  url = {https://github.com/Abelo9996/EHRJEPA}
}
```

### Related Work

This work builds upon:
- **I-JEPA**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023)
- **MIMIC-IV**: Johnson et al., "MIMIC-IV: A freely accessible electronic health record dataset" (2023)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_ehr.txt

# Run tests (if any)
python -m pytest tests/
```

## Acknowledgments

- **MIMIC-IV**: Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.1). PhysioNet.
- **I-JEPA**: Original architecture from Meta AI Research
- **PhysioNet**: For providing access to clinical datasets

## Contact

- **Author**: Abel Yagubyan
- **Email**: abelyagubyan@berkeley.edu
- **GitHub**: [@Abelo9996](https://github.com/Abelo9996)

## Troubleshooting

### Common Issues

**1. Out of Memory**
```yaml
# Reduce batch size in config
data:
  batch_size: 32  # or 16
```

**2. Slow Training on CPU**
```yaml
# Use smaller model
meta:
  model_name: temporal_transformer_small
```

**3. Data Loading Errors**
- Ensure MIMIC-IV data is in `./mimic-iv-2.1/`
- Check preprocessing completed successfully
- Verify CSV files exist in `./data/mimic_visits/`

**4. Poor Evaluation Performance**
- Train for more epochs (currently only 10/100 shown)
- Increase dataset size (use more patients)
- Try task-specific fine-tuning instead of linear probes

## Roadmap

- [ ] Add more downstream evaluation tasks
- [ ] Implement multi-modal learning (text + structured data)
- [ ] Support for other EHR datasets (eICU, OMOP)
- [ ] Distributed training support
- [ ] Pre-trained model checkpoints
- [ ] Interactive visualization dashboard

---

**Last Updated**: December 2025

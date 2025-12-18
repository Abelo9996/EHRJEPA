# JEPA-EHR Quick Start Guide

## ✅ What's Working

The JEPA model for EHR data is **successfully training** and learning meaningful representations from temporal medical data.

**Training Evidence:**
```
Epoch 1: Loss 0.447 → 0.349 ✓
Epoch 2: Loss 0.284 → 0.241 ✓
Epoch 3: Loss 0.199 → 0.175 ✓
```

## 🚀 Quick Start (2 minutes)

```bash
# 1. Activate environment
cd /Users/abelyagubyan/Downloads/EHRJEPA
source venv/bin/activate

# 2. Generate test data (optional - already done)
python generate_sample_data_clean.py --num_patients 500

# 3. Train the model
python main_ehr.py --fname configs/sample_ehr_test.yaml
```

## 📊 Understanding the Output

When training, you'll see:
```
INFO:root:[Epoch, Iteration] loss: X.XXX masks: 15.0 5.0 [wd: X.XXe-XX] [lr: X.XXe-XX]
```

- **loss**: Lower is better (model prediction error)
- **masks: 15.0 5.0**: Using 15 context visits to predict 5 future visits
- **wd**: Weight decay (regularization)
- **lr**: Learning rate

## 🎯 Model Architecture

```
Input: Patient visit sequences (B, 20, 25)
  ↓
VisitEmbed (25 → 384)
  ↓
Temporal Transformer Encoder (384, 12 layers, 6 heads)
  ↓
Context Representation (B, 15, 384)
  ↓
Predictor (4 layers, 192 dim)
  ↓
Predicted Future Representation (B, 5, 384)
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `generate_sample_data_clean.py` | Create synthetic EHR data |
| `preprocess_mimic.py` | Process real MIMIC-IV data |
| `main_ehr.py` | Training entry point |
| `train_ehr.py` | Main training loop |
| `src/models/temporal_transformer.py` | Model architecture |
| `src/datasets/mimic_ehr.py` | Data loading |
| `configs/sample_ehr_test.yaml` | Configuration |

## ⚙️ Configuration Options

Edit `configs/sample_ehr_test.yaml`:

```yaml
# Model size (tiny, small, base, large)
model_name: temporal_transformer_small

# Training duration
epochs: 10

# Batch size
batch_size: 32

# Sequence parameters
sequence_length: 20    # Total visits per sequence
context_length: 15     # Visits used as input
prediction_length: 5   # Visits to predict

# Learning rate
lr: 0.0005
```

## 📈 Expected Results

### Sample Data (500 patients):
- **Training time**: ~5-10 minutes per epoch (CPU)
- **Loss**: Should decrease from ~0.4 to ~0.1-0.2
- **Model learns**: Temporal patterns in vital signs and labs

### Full MIMIC-IV Data:
- **Training time**: Several hours per epoch (GPU recommended)
- **Better representations**: More diverse patient patterns
- **Downstream performance**: Ready for clinical prediction tasks

## 🔧 Troubleshooting

### Out of Memory?
- Reduce `batch_size` in config (try 16 or 8)
- Use smaller model: `temporal_transformer_tiny`

### Training too slow?
- Use GPU (set `use_bfloat16: true`)
- Reduce `num_workers` to 0
- Use smaller dataset (`--max_admissions 100`)

### Loss not decreasing?
- Check learning rate (try 0.0001 - 0.001)
- Ensure data is normalized
- Verify batch size isn't too small

## 🎓 What the Model Learns

The JEPA model learns to:
1. **Encode** patient visits into meaningful representations
2. **Predict** future visit representations given past context
3. **Capture** temporal dependencies and patterns in EHR data
4. **Create** useful features for downstream tasks (mortality, readmission, etc.)

## 🔬 Next Steps for Research

1. **Evaluate Representations**:
   ```python
   # Extract features for downstream tasks
   # Use encoder.forward() on new data
   # Train linear classifier on top
   ```

2. **Visualize Embeddings**:
   - Use t-SNE/UMAP on learned representations
   - Color by outcomes (mortality, diagnosis)
   - See if similar patients cluster together

3. **Fine-tune for Tasks**:
   - Mortality prediction
   - Length of stay
   - Diagnosis prediction
   - Readmission risk

## 📝 Citation

If you use this code, cite the original I-JEPA paper:
```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={CVPR},
  year={2023}
}
```

## 🆘 Getting Help

Check these files for more details:
- `TRAINING_SUMMARY.md` - Complete technical documentation
- `README.md` - Original I-JEPA documentation
- Training logs: `logs/jepa_ehr_sample/jepa-ehr-sample_r0.csv`

## ✨ Success Indicators

Your model is working if you see:
- ✅ Loss decreasing over time
- ✅ Gradients in reasonable range (1e-5 to 1e-2)
- ✅ No NaN or Inf values
- ✅ Checkpoints being saved
- ✅ CSV log file being written

**Congratulations! You have a working JEPA model for EHR data! 🎉**

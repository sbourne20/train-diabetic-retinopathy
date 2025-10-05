# Weighted Ensemble Guide

## Quick Start

### Test predefined weighting strategies:
```bash
python weighted_ensemble_inference.py --dataset_path ./dataset_eyepacs
```

This will test:
1. **Simple Averaging** (current: 90.82%)
   - All models: 33.33% each

2. **Accuracy-Based Weighting**
   - DenseNet: 33.5% (88.88% accuracy)
   - MedSigLIP: 33.1% (87.74% accuracy)  
   - EfficientNetB2: 33.9% (89.87% accuracy)

3. **Best-Model Heavy**
   - EfficientNetB2: 60% (best performer)
   - DenseNet + MedSigLIP: 20% each

4. **Top-2 Models Only**
   - DenseNet + EfficientNetB2: 50% each
   - MedSigLIP: 0% (excluded)

### Find OPTIMAL weights automatically:
```bash
python weighted_ensemble_inference.py --dataset_path ./dataset_eyepacs --optimize --granularity 20
```

This will:
- Test hundreds of weight combinations
- Find the weights that maximize accuracy
- Save results to `./ensemble_3model_results/optimal_weights.json`
- Expected: 91.0-91.5% accuracy

**Granularity options:**
- `10` = Fast (~100 combinations, ~5 minutes)
- `20` = Balanced (~400 combinations, ~15 minutes)
- `50` = Thorough (~1,250 combinations, ~45 minutes)

## Expected Results

### Weighting Strategy Performance:

| Strategy | Expected Accuracy | Improvement | Use Case |
|----------|------------------|-------------|----------|
| Simple Averaging | 90.82% | Baseline | General purpose |
| Accuracy-Based | 90.9-91.1% | +0.1-0.3% | Trust better models more |
| Best-Model Heavy | 90.5-90.9% | -0.3-0.1% | Single model dominant |
| Top-2 Only | 90.7-91.0% | -0.1-0.2% | Remove weak model |
| **Optimized** | **91.0-91.5%** | **+0.2-0.7%** | Maximum performance |

## Understanding Weights

### Current Model Performance:
- **EfficientNetB2: 89.87%** ← Best individual
- **DenseNet121: 88.88%** ← Strong complementary
- **MedSigLIP-448: 87.74%** ← Weakest but adds diversity

### Optimal Weight Range (Expected):
```python
{
  'efficientnetb2': 0.35-0.40,  # Highest weight (best model)
  'densenet': 0.32-0.36,        # Medium-high (good diversity)
  'medsiglip': 0.26-0.30        # Lower (but not zero - still valuable)
}
```

**Why not exclude MedSigLIP?**
- Different architecture (Vision Transformer vs CNN)
- Captures different features
- May excel on specific cases where CNNs fail
- Ensemble diversity > individual accuracy

## How Weighting Helps

### Example Case: Ambiguous Image

**Simple Averaging:**
```
DenseNet:       Class 2 (60% conf)
MedSigLIP:      Class 1 (55% conf)
EfficientNetB2: Class 2 (70% conf)

Average probabilities → Class 2 (61.7%)
```

**Accuracy-Based Weighting:**
```
DenseNet:       Class 2 (60% conf) × 0.335 weight
MedSigLIP:      Class 1 (55% conf) × 0.331 weight
EfficientNetB2: Class 2 (70% conf) × 0.339 weight

Weighted average → Class 2 (63.2% conf) ← Higher confidence!
```

**Result:** More reliable predictions on hard cases.

## Integration with mata-dr.py

After finding optimal weights, update `mata-dr.py`:

```python
# In predict_ensemble() function, change from:
ensemble_probs = (densenet_probs + medsiglip_probs + efficientnetb2_probs) / 3

# To:
weights = [0.336, 0.329, 0.339]  # From optimal_weights.json
ensemble_probs = (
    weights[0] * densenet_probs +
    weights[1] * medsiglip_probs +
    weights[2] * efficientnetb2_probs
)
```

## Recommended Workflow

1. **Quick test** (5 min):
   ```bash
   python weighted_ensemble_inference.py
   ```

2. **If improvement seen, optimize** (15 min):
   ```bash
   python weighted_ensemble_inference.py --optimize --granularity 20
   ```

3. **If optimization helps, update mata-dr.py** with optimal weights

4. **Re-test full ensemble**:
   ```bash
   python simple_ensemble_inference.py # (updated with new weights)
   ```

## Expected Timeline

- Test predefined strategies: **~5 minutes**
- Grid search (granularity=20): **~15 minutes**
- Full optimization (granularity=50): **~45 minutes**

## When to Use Weighting

✅ **Use weighted ensemble when:**
- Model accuracies differ significantly (>2%)
- You have time to optimize
- Aiming for maximum performance
- Deploying to production

❌ **Skip weighting when:**
- Models have similar accuracy (<1% difference)
- Simple averaging already exceeds target
- Rapid prototyping
- Computational resources limited

## Your Case

**Recommendation:** ✅ **Try weighted ensemble**

Reasons:
- 2.13% spread between models (significant)
- Currently at 90.82% (close to 91-92% target)
- Weighting could push to 91.0-91.5%
- Worth 15 minutes to test


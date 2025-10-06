# 🚀 Vast.AI EfficientNetB2 Training Guide

## ✅ **Step-by-Step Training Instructions**

### **Prerequisites:**
1. ✅ Balanced dataset created: `/Volumes/Untitled/dr/dataset_eyepacs_ori_balanced_smote`
2. ✅ Upload to vast.ai server before training
3. ✅ V100 GPU recommended (training time: ~8-10 hours)

---

## **📦 Step 1: Upload Dataset to Vast.AI**

**On your local machine:**
```bash
# Compress balanced dataset for upload
cd /Volumes/Untitled/dr
tar -czf dataset_eyepacs_ori_balanced_smote.tar.gz dataset_eyepacs_ori_balanced_smote/

# Upload to vast.ai server (replace with your server details)
scp -P 6209 -i vast_ai dataset_eyepacs_ori_balanced_smote.tar.gz root@206.172.240.211:~/
```

**On vast.ai server:**
```bash
# Extract dataset
cd ~/
tar -xzf dataset_eyepacs_ori_balanced_smote.tar.gz

# Verify structure
ls -la dataset_eyepacs_ori_balanced_smote/
# Should show: train/ val/ test/ balance_report.json
```

---

## **🔧 Step 2: Setup Environment on Vast.AI**

```bash
# SSH to vast.ai server
ssh -p 6209 -i vast_ai root@206.172.240.211

# Navigate to project directory
cd /path/to/train-diabetic-retinopathy

# Activate virtual environment (or create if needed)
source venv/bin/activate

# Install dependencies (if not already installed)
pip install torch torchvision timm albumentations scikit-learn tqdm opencv-python pillow
```

---

## **🎯 Step 3: Run EfficientNetB2 Training**

**Verify dataset path exists:**
```bash
# Make sure balanced dataset is in the correct location
ls -la ./dataset_eyepacs_ori_balanced_smote/train/
# Should show 5 class folders (0, 1, 2, 3, 4) with ~8,000 images each
```

**Start training:**
```bash
# Make script executable
chmod +x train_efficientnetb2.sh

# Run training
bash train_efficientnetb2.sh
```

**Alternative: Run in tmux/screen (recommended for long training)**
```bash
# Start tmux session
tmux new -s efficientnetb2_training

# Run training
bash train_efficientnetb2.sh

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t efficientnetb2_training
```

---

## **📊 Step 4: Monitor Training Progress**

**Training will output:**
- Epoch progress (1-100)
- Training loss and accuracy
- Validation loss and accuracy
- Best model checkpoints saved every 5 epochs
- Early stopping if no improvement for 12 epochs

**Expected output example:**
```
Epoch [1/100] - Train Loss: 0.8234, Train Acc: 68.5%, Val Loss: 0.6543, Val Acc: 74.2%
Epoch [2/100] - Train Loss: 0.5612, Train Acc: 78.9%, Val Loss: 0.4821, Val Acc: 82.1%
...
Epoch [50/100] - Train Loss: 0.1234, Train Acc: 95.2%, Val Loss: 0.1567, Val Acc: 94.8%
✅ New best model saved! Val Acc: 94.8%
```

**Monitor GPU usage (in separate terminal):**
```bash
watch -n 1 nvidia-smi
```

---

## **📈 Step 5: Analyze Results After Training**

**Once training completes:**
```bash
# Analyze the best checkpoint
python model_analyzer.py --model ./efficientnetb2_eyepacs_balanced_results/models/best_efficientnetb2_multiclass.pth

# Expected output:
# - Accuracy: 96%+
# - Per-class performance
# - Medical-grade assessment
```

**Check training logs:**
```bash
# View training history
cat ./efficientnetb2_eyepacs_balanced_results/training_log.txt

# View final metrics
cat ./efficientnetb2_eyepacs_balanced_results/final_metrics.json
```

---

## **🎯 Expected Training Timeline**

| Phase | Duration | Description |
|-------|----------|-------------|
| Warmup | 30-40 min | Epochs 1-5 (stabilization) |
| Rapid Learning | 2-3 hours | Epochs 6-40 (major accuracy gains) |
| Fine-tuning | 3-4 hours | Epochs 41-80 (convergence) |
| Final Optimization | 1-2 hours | Epochs 81-100 (reaching target) |
| **Total** | **8-10 hours** | Full training on V100 GPU |

---

## **✅ Success Criteria**

**EfficientNetB2 Individual Model:**
- ✅ Overall accuracy: **≥96%**
- ✅ Class 0 (No DR): **≥96%**
- ✅ Class 1 (Mild): **≥91%**
- ✅ Class 2 (Moderate): **≥93%**
- ✅ Class 3 (Severe): **≥92%**
- ✅ Class 4 (PDR): **≥93%**

**If these targets are met:**
- 🎯 Medical-grade threshold achieved (≥90% per class)
- 🎯 Ready for ensemble combination
- 🎯 Individual model can be used standalone

---

## **📁 Output Files Location**

After training completes:
```
./efficientnetb2_eyepacs_balanced_results/
├── models/
│   ├── best_efficientnetb2_multiclass.pth        # Best model checkpoint
│   ├── checkpoint_epoch_5.pth                    # Periodic checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── ...
├── training_log.txt                              # Full training log
├── final_metrics.json                            # Final performance metrics
├── confusion_matrix.png                          # Per-class performance
└── training_curves.png                           # Loss/accuracy plots
```

---

## **🚨 Troubleshooting**

### **Issue: Out of Memory Error**
```bash
# Reduce batch size in train_efficientnetb2.sh
# Change --batch_size 32 to --batch_size 16
```

### **Issue: Dataset Not Found**
```bash
# Verify dataset path
ls -la ./dataset_eyepacs_ori_balanced_smote/

# If in different location, update train_efficientnetb2.sh line 40:
--dataset_path /correct/path/to/dataset_eyepacs_ori_balanced_smote
```

### **Issue: Training Stuck/Not Improving**
```bash
# Check if overfitting - compare train vs val accuracy
# If train >> val: Increase dropout or weight decay
# If both low: Increase learning rate or reduce regularization
```

---

## **📋 Next Steps After EfficientNetB2**

1. **Download trained model to local:**
   ```bash
   scp -P 6209 -i vast_ai root@206.172.240.211:~/train-diabetic-retinopathy/efficientnetb2_eyepacs_balanced_results/models/best_efficientnetb2_multiclass.pth ./
   ```

2. **Train ResNet50 (same balanced dataset):**
   ```bash
   bash train_resnet50.sh
   ```

3. **Train DenseNet121 (same balanced dataset):**
   ```bash
   bash train_densenet121.sh
   ```

4. **Create ensemble (after all 3 models trained):**
   - Combine EfficientNetB2 + ResNet50 + DenseNet121
   - Expected ensemble accuracy: **96.96%**
   - Exceeds medical-grade requirement!

---

## **🎯 Summary: What You're Running**

**Dataset:**
- ✅ 40,001 balanced training images (8,000 per class)
- ✅ Original validation set (3,514 images)
- ✅ Original test set (3,513 images)

**Model:**
- ✅ EfficientNetB2 (9M parameters)
- ✅ Research-validated architecture (96.27% achievable)
- ✅ Optimized for medical imaging

**Training Configuration:**
- ✅ 100 epochs with cosine annealing
- ✅ Focal loss for edge cases
- ✅ Medical augmentation (rotation, brightness, contrast)
- ✅ Early stopping to prevent overfitting
- ✅ Target: 96%+ accuracy

**Expected Result:**
- 🎯 96%+ overall accuracy
- 🎯 90%+ per-class accuracy (medical-grade ✅)
- 🎯 Ready for ensemble to reach 96.96%

---

## **📞 Support Commands**

**Check training status:**
```bash
tail -f ./efficientnetb2_eyepacs_balanced_results/training_log.txt
```

**Check GPU usage:**
```bash
nvidia-smi
```

**Estimate time remaining:**
```bash
# If at epoch 30/100 after 3 hours
# Remaining: (100-30) * (3/30) = 70 * 0.1 = 7 hours
```

Good luck with training! 🚀

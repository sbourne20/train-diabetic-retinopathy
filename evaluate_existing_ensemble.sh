#!/bin/bash
################################################################################
# Evaluate Existing OVO Ensemble
#
# PURPOSE: Evaluate pre-trained binary classifiers using weighted voting
#
# WHEN TO USE:
#   - You already have trained binary classifiers in results/models/
#   - You want to test on new data
#   - You want to use weighted voting with real accuracies
#
# PREREQUISITES:
#   - All binary classifiers trained and saved in models/ directory
#   - Checkpoint files named: best_<model>_<class_a>_<class_b>.pth
#   - Each checkpoint contains 'best_val_accuracy' or 'val_accuracy'
#
################################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  OVO Ensemble Evaluation with Weighted Voting${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Configuration
RESULTS_DIR="./efficientnetb2_5class_results"  # Change this to your results directory
DATASET_PATH="./dataset_eyepacs"  # Change this to your dataset path
NUM_CLASSES=5
BASE_MODELS=("mobilenet_v2" "inception_v3" "densenet121")  # Models used in training

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --results-dir DIR    Results directory containing models/ (default: ./efficientnetb2_5class_results)"
            echo "  --dataset PATH       Path to test dataset (default: ./dataset_eyepacs)"
            echo "  --num-classes N      Number of classes (default: 5)"
            echo "  --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --results-dir ./efficientnetb2_5class_results --dataset ./dataset_eyepacs"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}❌ Error: Results directory not found: $RESULTS_DIR${NC}"
    exit 1
fi

# Validate models directory exists
if [ ! -d "$RESULTS_DIR/models" ]; then
    echo -e "${RED}❌ Error: Models directory not found: $RESULTS_DIR/models${NC}"
    exit 1
fi

# Validate dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}❌ Error: Dataset not found: $DATASET_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Configuration:${NC}"
echo -e "   Results Directory: ${YELLOW}$RESULTS_DIR${NC}"
echo -e "   Dataset Path: ${YELLOW}$DATASET_PATH${NC}"
echo -e "   Number of Classes: ${YELLOW}$NUM_CLASSES${NC}"
echo ""

# Count available binary classifiers
echo -e "${BLUE}📊 Checking available binary classifiers...${NC}"
expected_count=$((NUM_CLASSES * (NUM_CLASSES - 1) / 2 * ${#BASE_MODELS[@]}))
actual_count=$(find "$RESULTS_DIR/models" -name "best_*.pth" -not -name "*ensemble*" | wc -l | tr -d ' ')

echo -e "   Expected: ${YELLOW}$expected_count${NC} binary classifiers"
echo -e "   Found: ${YELLOW}$actual_count${NC} binary classifiers"

if [ "$actual_count" -lt "$expected_count" ]; then
    echo -e "${YELLOW}⚠️  Warning: Some binary classifiers are missing!${NC}"
    echo -e "   Evaluation will use available classifiers only."
fi

# List models
echo ""
echo -e "${BLUE}📦 Available models:${NC}"
for model in "${BASE_MODELS[@]}"; do
    model_count=$(find "$RESULTS_DIR/models" -name "best_${model}_*.pth" | wc -l | tr -d ' ')
    echo -e "   ${model}: ${YELLOW}${model_count}${NC} classifiers"
done

# Check for OVO ensemble
echo ""
if [ -f "$RESULTS_DIR/models/ovo_ensemble_best.pth" ]; then
    echo -e "${GREEN}✅ Found existing OVO ensemble${NC}"
    echo -e "   Will reload with updated weighted voting"
else
    echo -e "${YELLOW}⚠️  No existing OVO ensemble found${NC}"
    echo -e "   Will create new ensemble from binary classifiers"
fi

# Create evaluation script
echo ""
echo -e "${BLUE}🚀 Starting evaluation...${NC}"

python3 << END_PYTHON
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

# Add current directory to path
sys.path.insert(0, '.')

# Import from ensemble_5class_trainer
from ensemble_5class_trainer import OVOEnsemble, create_dataloaders

print("\n" + "="*80)
print("EVALUATION WITH WEIGHTED VOTING")
print("="*80)

# Configuration
results_dir = Path("$RESULTS_DIR")
dataset_path = Path("$DATASET_PATH")
num_classes = $NUM_CLASSES
base_models = [${BASE_MODELS[@]/#/"'"} ${BASE_MODELS[@]/%/"'"}]
base_models = [m.strip("'") for m in base_models if m.strip("'")]

print(f"\n📁 Loading from: {results_dir}")
print(f"📊 Dataset: {dataset_path}")
print(f"🔢 Classes: {num_classes}")
print(f"🤖 Models: {base_models}")

# Create test dataloader
print("\n🔄 Loading test dataset...")
try:
    config = {
        'data': {
            'dataset_path': str(dataset_path),
            'num_classes': num_classes,
            'batch_size': 32,
            'image_size': 224,
            'num_workers': 4
        }
    }

    _, _, test_loader = create_dataloaders(config)
    print(f"✅ Loaded {len(test_loader.dataset)} test images")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Create OVO ensemble
print("\n🅾️  Creating OVO Ensemble...")
ovo_ensemble = OVOEnsemble(
    base_models=base_models,
    num_classes=num_classes,
    freeze_weights=True,
    dropout=0.5
)

# Load trained weights
print("\n🔄 Loading trained binary classifiers...")
loaded_count = 0
models_dir = results_dir / "models"

for model_name in base_models:
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            pair_name = f"pair_{i}_{j}"
            model_path = models_dir / f"best_{model_name}_{i}_{j}.pth"

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        ovo_ensemble.classifiers[model_name][pair_name].load_state_dict(
                            checkpoint['model_state_dict']
                        )
                    else:
                        ovo_ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint)

                    loaded_count += 1
                except Exception as e:
                    print(f"⚠️  Failed to load {model_path.name}: {e}")

print(f"✅ Loaded {loaded_count}/{len(base_models) * num_classes * (num_classes-1) // 2} classifiers")

# Load binary accuracies for weighted voting
print("\n⚖️  Loading binary classifier accuracies for weighted voting...")
try:
    accuracies = ovo_ensemble.load_binary_accuracies(str(results_dir))
    print("✅ Weighted voting enabled with actual model accuracies")

    # Display accuracy statistics
    all_accs = []
    for model_accs in accuracies.values():
        all_accs.extend(model_accs.values())

    if all_accs:
        print(f"\n📊 Accuracy Statistics:")
        print(f"   Mean: {np.mean(all_accs):.4f} ({np.mean(all_accs)*100:.2f}%)")
        print(f"   Min:  {np.min(all_accs):.4f} ({np.min(all_accs)*100:.2f}%)")
        print(f"   Max:  {np.max(all_accs):.4f} ({np.max(all_accs)*100:.2f}%)")

except Exception as e:
    print(f"⚠️  Failed to load accuracies: {e}")
    print("   Using default accuracy weights")

# Evaluate
print("\n🧪 Evaluating on test set...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

ovo_ensemble = ovo_ensemble.to(device)
ovo_ensemble.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        targets = targets.to(device)

        outputs = ovo_ensemble(images, return_individual=False)
        _, predictions = torch.max(outputs['logits'], 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(all_targets, all_predictions)
conf_matrix = confusion_matrix(all_targets, all_predictions)

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(all_targets, all_predictions,
                          target_names=[f"Class {i}" for i in range(num_classes)]))

print("\n🔢 Confusion Matrix:")
print(conf_matrix)

# Save results
results = {
    'accuracy': float(accuracy),
    'confusion_matrix': conf_matrix.tolist(),
    'predictions': [int(p) for p in all_predictions],
    'targets': [int(t) for t in all_targets],
    'weighted_voting': ovo_ensemble.binary_accuracies is not None
}

output_file = results_dir / 'weighted_evaluation_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")
print("\n✅ Evaluation complete!")

END_PYTHON

evaluation_status=$?

echo ""
if [ $evaluation_status -eq 0 ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ Evaluation completed successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}📊 Results saved in:${NC}"
    echo -e "   ${YELLOW}$RESULTS_DIR/weighted_evaluation_results.json${NC}"
else
    echo -e "${RED}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ Evaluation failed!${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════════${NC}"
    exit 1
fi

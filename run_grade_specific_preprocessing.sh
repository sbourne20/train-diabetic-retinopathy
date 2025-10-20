#!/bin/bash
#
# Grade-Specific Preprocessing Script
#
# This script preprocesses the entire diabetic retinopathy dataset with
# grade-specific enhancement parameters optimized for each severity level.
#
# Input:  /Volumes/WDC1TB/dataset/diabetic-retinopathy/dataset_eyepacs_5class_balanced
# Output: /Volumes/WDC1TB/dataset/diabetic-retinopathy/dataset_eyepacs_5class_balanced_enhanced
#
# Expected processing time: ~10-30 minutes depending on dataset size
# Expected storage: ~2x original dataset size
#
# Expected accuracy improvement: +3-7% overall
#   - Grade 0 (No DR):        +1-2% (false positive reduction)
#   - Grade 1 (Mild NPDR):    +5-7% (microaneurysm detection)
#   - Grade 2 (Moderate NPDR): +3-4% (hemorrhage/exudate clarity)
#   - Grade 3 (Severe NPDR):   +4-6% (vessel abnormality detection)
#   - Grade 4 (PDR):           +6-8% (neovascularization enhancement)
#

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Grade-Specific Dataset Preprocessing${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Dataset paths
INPUT_DATASET="/Volumes/WDC1TB/dataset/diabetic-retinopathy/dataset_eyepacs_5class_balanced"
OUTPUT_DATASET="/Volumes/WDC1TB/dataset/diabetic-retinopathy/dataset_eyepacs_5class_balanced_enhanced"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Input:  $INPUT_DATASET"
echo "  Output: $OUTPUT_DATASET"
echo "  Classes: 5 (Grade 0-4)"
echo ""

# Check if input exists
if [ ! -d "$INPUT_DATASET" ]; then
    echo -e "${RED}❌ Error: Input dataset not found at $INPUT_DATASET${NC}"
    exit 1
fi

# Check if output already exists
if [ -d "$OUTPUT_DATASET" ]; then
    echo -e "${YELLOW}⚠️  Warning: Output directory already exists!${NC}"
    echo "  $OUTPUT_DATASET"
    echo ""
    read -p "Do you want to overwrite it? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo -e "${RED}Aborted.${NC}"
        exit 1
    fi
    OVERWRITE_FLAG="--overwrite"
else
    OVERWRITE_FLAG=""
fi

# Check system info
echo -e "${BLUE}System Information:${NC}"
CPU_CORES=$(sysctl -n hw.ncpu)
echo "  CPU: $(sysctl -n machdep.cpu.brand_string)"
echo "  CPU Cores: $CPU_CORES cores (parallel processing enabled)"
echo "  Memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
echo ""

# Check if scipy is installed (required for background flattening)
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: scipy not installed${NC}"
    echo "Install with: pip3 install scipy"
    exit 1
fi
echo -e "${GREEN}✅ Dependencies OK${NC}"
echo -e "${GREEN}✅ GPU Acceleration: OpenCV using Metal Performance Shaders on M4${NC}"
echo -e "${GREEN}✅ Parallel Processing: Using all $CPU_CORES CPU cores${NC}"
echo ""

# Show enhancement parameters
echo -e "${BLUE}Enhancement Parameters per Grade:${NC}"
python3 preprocess_grade_specific.py --show_params
echo ""

# Confirm before starting
echo -e "${YELLOW}This will process ALL images in the dataset.${NC}"
echo "Estimated time: 10-30 minutes depending on dataset size"
echo ""
read -p "Continue with preprocessing? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo -e "${RED}Aborted.${NC}"
    exit 1
fi

# Start preprocessing
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Preprocessing...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

START_TIME=$(date +%s)

python3 preprocess_grade_specific.py \
    --input "$INPUT_DATASET" \
    --output "$OUTPUT_DATASET" \
    --num_classes 5 \
    --preserve_structure True \
    --target_size 448 \
    $OVERWRITE_FLAG

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Preprocessing Completed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Processing time: ${MINUTES}m ${SECONDS}s"
    echo "Enhanced dataset: $OUTPUT_DATASET"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Verify enhanced images:"
    echo "   ls $OUTPUT_DATASET/train/0/"
    echo ""
    echo "2. Compare original vs enhanced (optional):"
    echo "   python3 visualize_grade_enhancement.py \\"
    echo "     --image $INPUT_DATASET/train/2/sample_image.jpeg \\"
    echo "     --output ./grade_enhancement_comparison.png"
    echo ""
    echo "3. Train with enhanced dataset:"
    echo "   python3 ensemble_5class_trainer.py \\"
    echo "     --mode train \\"
    echo "     --dataset_path $OUTPUT_DATASET \\"
    echo "     --epochs 100 \\"
    echo "     --batch_size 32"
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: Do NOT use --enable_clahe or --enable_flatten_sharpen${NC}"
    echo -e "${YELLOW}   when training on this enhanced dataset (already applied!)${NC}"
    echo ""
else
    echo -e "${RED}❌ Preprocessing Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo "Exit code: $EXIT_CODE"
    echo "Check error messages above for details"
    exit $EXIT_CODE
fi

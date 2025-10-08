#!/bin/bash
# Fast dataset split using shell commands for 3-class DR dataset

SOURCE="/Volumes/Untitled/dr/dataset_eyepacs_3class/train"
TARGET="/Volumes/Untitled/dr/dataset_eyepacs_3class_balanced"

# Split ratios
TRAIN_RATIO=80
VAL_RATIO=10
TEST_RATIO=10

echo "================================================================================"
echo "3-CLASS DR DATASET SPLIT (80/10/10)"
echo "================================================================================"

for CLASS in NORMAL NPDR PDR; do
    echo ""
    echo "📁 Processing class: $CLASS"

    # Get total count
    TOTAL=$(find "$SOURCE/$CLASS" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) | wc -l | tr -d ' ')
    echo "   Total images: $TOTAL"

    # Calculate split sizes
    TRAIN_SIZE=$(( TOTAL * TRAIN_RATIO / 100 ))
    VAL_SIZE=$(( TOTAL * VAL_RATIO / 100 ))
    TEST_SIZE=$(( TOTAL - TRAIN_SIZE - VAL_SIZE ))

    echo "   Train: $TRAIN_SIZE (${TRAIN_RATIO}%)"
    echo "   Val:   $VAL_SIZE (${VAL_RATIO}%)"
    echo "   Test:  $TEST_SIZE (${TEST_RATIO}%)"

    # Create shuffled list of files
    TEMP_LIST="/tmp/dataset_split_${CLASS}_$$.txt"
    find "$SOURCE/$CLASS" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) | shuf > "$TEMP_LIST"

    # Split into train/val/test
    echo "   Copying to train..."
    head -n $TRAIN_SIZE "$TEMP_LIST" | while read file; do
        cp "$file" "$TARGET/train/$CLASS/"
    done
    echo "   ✅ Train done"

    echo "   Copying to val..."
    head -n $(( TRAIN_SIZE + VAL_SIZE )) "$TEMP_LIST" | tail -n $VAL_SIZE | while read file; do
        cp "$file" "$TARGET/val/$CLASS/"
    done
    echo "   ✅ Val done"

    echo "   Copying to test..."
    tail -n $TEST_SIZE "$TEMP_LIST" | while read file; do
        cp "$file" "$TARGET/test/$CLASS/"
    done
    echo "   ✅ Test done"

    # Cleanup
    rm "$TEMP_LIST"
done

echo ""
echo "================================================================================"
echo "📊 FINAL DATASET SUMMARY"
echo "================================================================================"

for SPLIT in train val test; do
    echo ""
    echo "${SPLIT^^}:"
    for CLASS in NORMAL NPDR PDR; do
        COUNT=$(find "$TARGET/$SPLIT/$CLASS" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) | wc -l | tr -d ' ')
        printf "  %-8s: %s\n" "$CLASS" "$COUNT"
    done
done

echo ""
echo "✅ Dataset split completed successfully!"
echo "📂 Output directory: $TARGET"

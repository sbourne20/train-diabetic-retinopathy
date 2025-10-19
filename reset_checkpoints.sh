#!/bin/bash

# Reset OVO Training Checkpoints
# This allows you to restart training from scratch or for specific pairs

echo "🗑️  OVO CHECKPOINT RESET UTILITY"
echo "=" * 70

# Show current checkpoints
echo ""
echo "Current checkpoints:"
ls -lh ./v2-model-dr/seresnext_5class_results/models/*.pth 2>/dev/null || echo "  No checkpoints found"
echo ""

# Options
echo "What would you like to do?"
echo ""
echo "1. Delete ONLY checkpoint 0_2 (retrain just that pair)"
echo "2. Delete checkpoints 0_1 AND 0_2 (fresh start)"
echo "3. Delete ALL checkpoints (complete reset)"
echo "4. Cancel (do nothing)"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🗑️  Deleting checkpoint 0_2..."
        rm -v ./v2-model-dr/seresnext_5class_results/models/best_seresnext50_32x4d_0_2.pth
        echo "✅ Done! Run ./train_5class_seresnext.sh to retrain 0_2"
        ;;
    2)
        echo ""
        echo "🗑️  Deleting checkpoints 0_1 and 0_2..."
        rm -v ./v2-model-dr/seresnext_5class_results/models/best_seresnext50_32x4d_0_1.pth
        rm -v ./v2-model-dr/seresnext_5class_results/models/best_seresnext50_32x4d_0_2.pth
        echo "✅ Done! Run ./train_5class_seresnext.sh to start fresh"
        ;;
    3)
        echo ""
        echo "⚠️  This will delete ALL checkpoints. Are you sure? (yes/no)"
        read -p "> " confirm
        if [ "$confirm" == "yes" ]; then
            echo "🗑️  Deleting all checkpoints..."
            rm -v ./v2-model-dr/seresnext_5class_results/models/best_*.pth
            echo "✅ Done! Run ./train_5class_seresnext.sh to start completely fresh"
        else
            echo "❌ Cancelled"
        fi
        ;;
    4)
        echo "❌ Cancelled - no changes made"
        ;;
    *)
        echo "❌ Invalid choice"
        ;;
esac

echo ""
echo "Remaining checkpoints:"
ls -lh ./v2-model-dr/seresnext_5class_results/models/*.pth 2>/dev/null || echo "  No checkpoints found"
echo ""

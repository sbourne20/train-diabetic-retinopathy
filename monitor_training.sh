#!/bin/bash
# Real-time Training Progress Monitor
# Run this in a separate terminal to see clear accuracy updates

echo "🔍 V100 TRAINING PROGRESS MONITOR"
echo "================================="
echo "This will show you real-time training progress with clear accuracy display"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Monitor the training log file if it exists
LOG_FILE="local_outputs/training.log"
if [ -f "$LOG_FILE" ]; then
    echo "📊 Monitoring training log: $LOG_FILE"
    tail -f "$LOG_FILE" | grep -E "(EPOCH|Training|Validation|Medical Grade)" --line-buffered
else
    echo "📊 Monitoring training output directly..."
    # Monitor the main training process
    while true; do
        clear
        echo "🎯 V100 TRAINING PROGRESS MONITOR - $(date)"
        echo "=============================================="
        
        # Look for training processes
        if pgrep -f "local_trainer.py" > /dev/null; then
            echo "✅ Training is running (PID: $(pgrep -f local_trainer.py))"
            
            # Show GPU usage
            if command -v nvidia-smi &> /dev/null; then
                echo ""
                echo "🎮 GPU STATUS:"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read gpu memory_used memory_total temp; do
                    echo "   GPU Util: ${gpu}% | Memory: ${memory_used}MB/${memory_total}MB | Temp: ${temp}°C"
                done
            fi
            
            # Show latest accuracy if checkpoint files exist
            echo ""
            echo "📈 LATEST METRICS:"
            if [ -f "local_outputs/training_history.json" ]; then
                python3 -c "
import json
import os
try:
    with open('local_outputs/training_history.json', 'r') as f:
        history = json.load(f)
    if 'val_accuracies' in history and history['val_accuracies']:
        latest_acc = history['val_accuracies'][-1]
        epoch = len(history['val_accuracies'])
        print(f'   📊 Latest Validation Accuracy: {latest_acc:.4f} (Epoch {epoch})')
    if 'best_accuracy' in history:
        print(f'   🏆 Best Accuracy So Far: {history[\"best_accuracy\"]:.4f}')
except Exception as e:
    print(f'   ⏳ Training metrics not yet available...')
"
            else
                echo "   ⏳ Training metrics file not yet created..."
            fi
            
            # Show latest checkpoint info from result folder
            echo ""
            echo "💾 MODELS IN RESULT FOLDER:"
            RESULT_DIR="local_outputs/result"
            if [ -d "$RESULT_DIR" ]; then
                echo "   📁 Result folder: $RESULT_DIR"
                
                # Check for best model
                if [ -f "$RESULT_DIR/best_model.pth" ]; then
                    best_size=$(du -h "$RESULT_DIR/best_model.pth" | cut -f1)
                    echo "   🏆 Best model: best_model.pth ($best_size)"
                fi
                
                # Check for epoch checkpoints
                if ls "$RESULT_DIR"/epoch_*.pth 1> /dev/null 2>&1; then
                    latest_checkpoint=$(ls -t "$RESULT_DIR"/epoch_*.pth | head -n1)
                    checkpoint_name=$(basename "$latest_checkpoint")
                    checkpoint_time=$(stat -c %Y "$latest_checkpoint" 2>/dev/null || stat -f %m "$latest_checkpoint")
                    current_time=$(date +%s)
                    age=$((current_time - checkpoint_time))
                    echo "   📁 Latest checkpoint: $checkpoint_name (${age}s ago)"
                    
                    # Count total checkpoints
                    checkpoint_count=$(ls "$RESULT_DIR"/epoch_*.pth | wc -l)
                    echo "   📊 Total epoch checkpoints: $checkpoint_count"
                else
                    echo "   ⏳ No epoch checkpoints created yet..."
                fi
                
                # Show all model files
                model_count=$(ls "$RESULT_DIR"/*.pth 2>/dev/null | wc -l)
                if [ "$model_count" -gt 0 ]; then
                    total_size=$(du -sh "$RESULT_DIR" | cut -f1)
                    echo "   💾 Total models: $model_count files ($total_size)"
                fi
            else
                echo "   📁 Result folder not yet created: $RESULT_DIR"
            fi
            
        else
            echo "❌ No training process detected"
            echo "   Start training with: ./medical_grade_local_training.sh"
        fi
        
        echo ""
        echo "🔄 Refreshing in 30 seconds... (Ctrl+C to stop)"
        sleep 30
    done
fi
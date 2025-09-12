#!/bin/bash
# 
# Real-time checkpoint monitoring script
# Run this in a separate terminal while training
#

echo "🔍 CHECKPOINT MONITOR"
echo "Monitoring ./results/checkpoints/ directory..."
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="

while true; do
    clear
    echo "🔍 CHECKPOINT MONITOR - $(date)"
    echo "=========================================="
    
    if [ -d "./results/checkpoints" ]; then
        echo "📁 Directory: $(pwd)/results/checkpoints"
        echo ""
        echo "📊 Checkpoint Files:"
        ls -la ./results/checkpoints/*.pth 2>/dev/null | while read line; do
            echo "   $line"
        done
        
        checkpoint_count=$(ls -1 ./results/checkpoints/*.pth 2>/dev/null | wc -l || echo "0")
        echo ""
        echo "📈 Total Checkpoints: $checkpoint_count"
        
        if [ -f "./results/checkpoints/ensemble_best.pth" ]; then
            echo "🏆 Best ensemble model: ✅ EXISTS"
        else
            echo "🏆 Best ensemble model: ⏳ WAITING"
        fi
        
        echo ""
        echo "🔄 Last updated: $(date)"
        echo "=========================================="
    else
        echo "❌ Checkpoint directory not found: ./results/checkpoints"
        echo "🔍 Current directory: $(pwd)"
        echo "📁 Available directories:"
        ls -la | grep "^d"
    fi
    
    sleep 10
done
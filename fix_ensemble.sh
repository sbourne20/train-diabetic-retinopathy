#!/bin/bash

# Fix OVO Ensemble Performance Script
echo "🔧 FIXING OVO ENSEMBLE PERFORMANCE"
echo "===================================="

echo "📝 Step 1: Create ensemble file from individual models..."
python create_ensemble.py

echo ""
echo "🎯 Step 2: Evaluate with improved weighted voting..."
python fix_ovo_ensemble.py

echo ""
echo "✅ Ensemble fix completed!"
echo "📁 Check results in: ovo_ensemble_results_v2/results/improved_ovo_results.json"
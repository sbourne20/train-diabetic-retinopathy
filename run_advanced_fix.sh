#!/bin/bash

# Advanced OVO Voting Fix
echo "🚀 ADVANCED OVO VOTING ALGORITHM FIX"
echo "===================================="

echo "🎯 Problem: Binary models are excellent (91-93%) but ensemble fails (72%)"
echo "🔧 Solution: Advanced voting algorithms for severe class imbalance"
echo ""

echo "📊 Class imbalance in test data:"
echo "   Class 0 (No DR): 43.5% - Gets vote advantage"
echo "   Classes 1,3,4: ~9% each - Votes get overwhelmed"
echo ""

echo "🛠️ Running advanced voting strategies..."
python fix_ovo_voting.py

echo ""
echo "✅ Advanced voting fix completed!"
echo "📁 Check: ovo_ensemble_results_v2/results/advanced_ovo_results.json"
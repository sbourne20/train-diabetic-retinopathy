#!/bin/bash

# Final Medical-Grade Fix
echo "🏥 FINAL MEDICAL-GRADE OVO ENSEMBLE FIX"
echo "======================================="

echo "📊 Previous Results Analysis:"
echo "   Best performance: 81.17% (pairwise voting)"
echo "   Latest attempt: 81.01% (failed to improve)"
echo "   CRITICAL ISSUE: Class 1 only 46.7% accuracy!"
echo "   Class 1 F1-score: 0.552 (far below medical threshold)"
echo ""

echo "🎯 ADAPTIVE THRESHOLD Fix Strategy:"
echo "   1. Class 1 adaptive thresholding (0.10-0.25 range)"
echo "   2. Targeted Class 0 penalty when Class 1 confident"
echo "   3. Class 1 vs Class 2 competition boost"
echo "   4. Target: Get Class 1 above 70% accuracy"
echo ""

echo "🚀 Running final medical-grade optimization..."
python final_medical_grade_fix.py

echo ""
echo "✅ Final medical-grade fix completed!"
echo "📁 Check: ovo_ensemble_results_v2/results/final_medical_grade_results.json"
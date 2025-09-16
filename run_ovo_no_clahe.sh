#!/bin/bash

# Quick Fix: OVO Training without CLAHE
# This will get you training immediately

echo "🚀 OVO TRAINING - NO CLAHE (QUICK FIX)"
echo "This fixes the CLAHE transform error and starts training immediately"
echo ""

# Clean up any previous runs
rm -rf ./ovo_no_clahe_results

# Create output directory
mkdir -p ./ovo_no_clahe_results
mkdir -p ./ovo_no_clahe_results/models
mkdir -p ./ovo_no_clahe_results/logs
mkdir -p ./ovo_no_clahe_results/results

echo "📁 Output directory created: ovo_no_clahe_results"
echo ""

# Run training without CLAHE
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset6 \
    --output_dir ./ovo_no_clahe_results \
    --epochs 25 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --freeze_weights \
    --enable_class_weights \
    --patience 15 \
    --early_stopping_patience 8 \
    --experiment_name ovo_ensemble_no_clahe_fix \
    --target_accuracy 0.9696 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ OVO training completed successfully!"
    echo "🎯 Check results in ./ovo_no_clahe_results/ directory"

    if [ -f "./ovo_no_clahe_results/results/complete_ovo_results.json" ]; then
        echo ""
        echo "📊 FINAL RESULTS:"

        # Extract accuracy
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
try:
    with open('./ovo_no_clahe_results/results/complete_ovo_results.json', 'r') as f:
        results = json.load(f)
    eval_results = results['evaluation_results']
    accuracy = eval_results['ensemble_accuracy']
    medical_pass = eval_results['medical_grade_pass']
    research_achieved = eval_results['research_target_achieved']

    print(f'🎯 Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'🏥 Medical Grade: {\"✅ PASS\" if medical_pass else \"❌ FAIL\"} (≥90% required)')
    print(f'📊 Research Target: {\"✅ ACHIEVED\" if research_achieved else \"❌ NOT ACHIEVED\"} (96.96% target)')

    print('\n📊 Individual Model Results:')
    for model, acc in eval_results['individual_accuracies'].items():
        print(f'   {model}: {acc:.4f} ({acc*100:.2f}%)')

except Exception as e:
    print(f'Results file available but could not parse: {e}')
"
        fi
    fi

else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check logs above for details"
fi

exit $EXIT_CODE
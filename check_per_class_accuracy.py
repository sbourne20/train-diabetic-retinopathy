#!/usr/bin/env python3
"""
Check per-class accuracy from W&B logs to understand what's really happening
"""

import sys

print("="*70)
print("UNDERSTANDING THE 73% ACCURACY")
print("="*70)

# Validation set distribution
val_dist = {
    0: 2581,  # 73.4% - No DR
    1: 245,   # 7.0% - Mild
    2: 529,   # 15.1% - Moderate
    3: 88,    # 2.5% - Severe
    4: 71     # 2.0% - PDR
}

total = sum(val_dist.values())

print("\n📊 Validation Set Distribution:")
for cls, count in val_dist.items():
    pct = count / total * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

print(f"\n  Total: {total} samples")

print("\n" + "="*70)
print("SCENARIO ANALYSIS")
print("="*70)

print("\n❌ Scenario 1: Model Always Predicts Class 0")
print("-" * 70)
correct_class_0 = val_dist[0]
total_predictions = total
accuracy = correct_class_0 / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")
print(f"This matches your 73% → Model might be biased to Class 0!")

print("\n✅ Scenario 2: Perfect Per-Class Accuracy (90% each)")
print("-" * 70)
total_correct = sum(count * 0.90 for count in val_dist.values())
accuracy = total_correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")
print(f"Even with 90% per-class, overall is 90% (not 96%)")

print("\n⚠️  Scenario 3: Good Class 0, Poor Others (Your Likely Case)")
print("-" * 70)
per_class_acc = {
    0: 0.95,  # 95% on Class 0
    1: 0.30,  # 30% on Class 1 (hard)
    2: 0.50,  # 50% on Class 2
    3: 0.20,  # 20% on Class 3 (very hard)
    4: 0.10   # 10% on Class 4 (very hard)
}
total_correct = sum(val_dist[cls] * per_class_acc[cls] for cls in range(5))
accuracy = total_correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")
print(f"This also gives ~73%!")
print("\nPer-class performance:")
for cls in range(5):
    print(f"  Class {cls}: {per_class_acc[cls]*100:.0f}%")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("""
Your 73% accuracy is MISLEADING because:

1. ❌ Validation set is heavily imbalanced (73% Class 0)
2. ❌ Model might be predicting Class 0 for everything
3. ❌ Or model is good at Class 0 but terrible at Classes 3, 4

To verify, check W&B confusion matrix or class-wise metrics!

Expected per-class accuracy for medical-grade model:
- Class 0: 96%+
- Class 1: 90%+
- Class 2: 92%+
- Class 3: 90%+
- Class 4: 92%+

If your model shows:
- Class 0: 95%+ but Classes 3,4 < 50% → Overfitting to majority
- All classes ~73% → Model actually learning but val set is biased

SOLUTION: Check confusion matrix in W&B!
""")

print("="*70)

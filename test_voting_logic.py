#!/usr/bin/env python3
"""
Test if the voting logic is actually the fixed version
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import OVOEnsemble

# Create a dummy ensemble
ovo = OVOEnsemble(base_models=['coatnet_0_rw_224'], num_classes=5)

# Create dummy input
dummy_input = torch.randn(2, 3, 224, 224)

# Get output
try:
    output = ovo(dummy_input)
    votes = output['logits']

    print("✅ Voting logic test:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output votes shape: {votes.shape}")
    print(f"   Sample votes: {votes[0]}")
    print(f"   Vote sum: {votes[0].sum()}")

    # Check if votes are integers (simple majority) or floats (weighted)
    if torch.all(votes == votes.long().float()):
        print("\n✅ FIXED: Using simple majority voting (integer votes)")
    else:
        print("\n❌ BROKEN: Using weighted voting (float votes)")

    # Check vote range
    max_vote = votes.max().item()
    if max_vote <= 4.1:  # Allow small float error
        print(f"✅ Vote range correct: max={max_vote:.1f} (expected ≤4)")
    else:
        print(f"❌ Vote range wrong: max={max_vote:.1f} (expected ≤4)")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

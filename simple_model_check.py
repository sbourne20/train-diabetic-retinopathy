#!/usr/bin/env python3
"""
Simple check of final_model.pth - just download and inspect the checkpoint
"""

import os
import torch
from google.cloud import storage
import tempfile

def check_final_model():
    """Download and inspect final_model.pth"""
    print("🔍 CHECKING FINAL_MODEL.PTH")
    print("=" * 40)
    
    bucket_name = "dr-data-2"
    model_gcs_path = "models/final_model.pth"
    
    # Create temporary directory for model
    with tempfile.TemporaryDirectory() as temp_dir:
        local_model_path = os.path.join(temp_dir, "final_model.pth")
        
        try:
            # Download model from GCS
            print(f"📥 Downloading from gs://{bucket_name}/{model_gcs_path}")
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(model_gcs_path)
            
            blob.download_to_filename(local_model_path)
            print(f"✅ Downloaded to {local_model_path}")
            
            # Load and inspect checkpoint
            print("🔍 Inspecting checkpoint...")
            checkpoint = torch.load(local_model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                print("📊 Checkpoint contains:")
                for key in checkpoint.keys():
                    print(f"  - {key}")
                
                # Extract key metrics if available
                epoch = checkpoint.get('epoch', 'Not found')
                val_acc = checkpoint.get('best_val_accuracy', checkpoint.get('val_accuracy', 'Not found'))
                train_acc = checkpoint.get('train_accuracy', 'Not found')
                val_loss = checkpoint.get('val_loss', 'Not found')
                
                print(f"\n📈 Key Metrics:")
                print(f"  Epoch: {epoch}")
                print(f"  Validation Accuracy: {val_acc}")
                print(f"  Training Accuracy: {train_acc}")
                print(f"  Validation Loss: {val_loss}")
                
                # Make recommendation based on validation accuracy
                if isinstance(val_acc, (int, float)):
                    val_acc_pct = val_acc * 100 if val_acc <= 1.0 else val_acc
                    
                    print(f"\n🎯 CONTINUATION RECOMMENDATION:")
                    print("=" * 40)
                    
                    if val_acc_pct >= 70:
                        print("✅ EXCELLENT: Continue with light regularization")
                        rec = "light"
                    elif val_acc_pct >= 50:
                        print("✅ GOOD: Continue with moderate regularization") 
                        rec = "moderate"
                    elif val_acc_pct >= 30:
                        print("⚠️  FAIR: Continue with extreme regularization")
                        rec = "extreme"
                    else:
                        print("❌ POOR: Consider starting fresh")
                        rec = "restart"
                    
                    return {
                        'validation_accuracy': val_acc,
                        'recommendation': rec,
                        'epoch': epoch
                    }
                else:
                    print("⚠️  Could not find validation accuracy in checkpoint")
                    return {'validation_accuracy': 'unknown', 'recommendation': 'unknown'}
                    
            else:
                print("⚠️  Checkpoint is just model state dict, no metrics available")
                return {'validation_accuracy': 'unknown', 'recommendation': 'moderate'}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

if __name__ == "__main__":
    result = check_final_model()
    if result:
        print(f"\n🏁 Result: {result}")
    else:
        print("\n❌ Check failed")
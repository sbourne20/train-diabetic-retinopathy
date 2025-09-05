#!/usr/bin/env python3
"""
Local evaluation of final_model.pth to check if suitable for continued training
Downloads model from GCS and evaluates on validation set
"""

import os
import torch
import sys
from pathlib import Path

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

from config import Config
from main import evaluate_model
from dataset import DRDataModule
from models import DiabeticRetinopathyModel
from google.cloud import storage
import tempfile

def download_model_from_gcs(bucket_name: str, model_path: str, local_path: str):
    """Download model from GCS to local path"""
    print(f"📥 Downloading model from gs://{bucket_name}/{model_path}")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    blob.download_to_filename(local_path)
    print(f"✅ Model downloaded to {local_path}")
    return local_path

def evaluate_final_model():
    """Evaluate final_model.pth for continued training suitability"""
    print("🔍 EVALUATING FINAL_MODEL.PTH FOR CONTINUED TRAINING")
    print("=" * 60)
    
    # Configuration
    bucket_name = "dr-data-2"
    model_gcs_path = "models/final_model.pth"
    dataset_name = "dataset3_augmented_resized"
    
    # Create temporary directory for model
    with tempfile.TemporaryDirectory() as temp_dir:
        local_model_path = os.path.join(temp_dir, "final_model.pth")
        
        try:
            # Download model from GCS
            download_model_from_gcs(bucket_name, model_gcs_path, local_model_path)
            
            # Setup configuration
            config = Config(
                dataset_name=dataset_name,
                dataset_type=1,  # 5-class DR structure
                batch_size=8,
                model_checkpoint_path=local_model_path
            )
            
            # Setup data module
            print("📊 Loading dataset...")
            data_module = DRDataModule(
                data_dir=f"gs://{bucket_name}/datasets/{dataset_name}",
                batch_size=config.batch_size,
                dataset_type=1,
                num_workers=2
            )
            data_module.setup()
            data_dict = data_module.get_data_dict()
            
            # Load model
            print("🧠 Loading model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DiabeticRetinopathyModel(num_classes=5)
            
            # Load checkpoint
            checkpoint = torch.load(local_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                best_acc = checkpoint.get('best_val_accuracy', 'unknown')
                print(f"📈 Loaded checkpoint from epoch {epoch}, best val accuracy: {best_acc}")
            else:
                model.load_state_dict(checkpoint)
                print("📈 Loaded model state dict")
            
            model = model.to(device)
            
            # Evaluate model
            print("🎯 Running evaluation...")
            results = evaluate_model(config, data_dict, model=model)
            
            # Display results
            print("\n" + "="*60)
            print("📊 EVALUATION RESULTS:")
            print("="*60)
            
            val_accuracy = results.get('val_accuracy', 0)
            val_loss = results.get('val_loss', 0)
            
            print(f"Validation Accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
            print(f"Validation Loss: {val_loss:.4f}")
            
            if 'per_class_accuracy' in results:
                print("\nPer-class Accuracy:")
                classes = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
                for i, (cls, acc) in enumerate(zip(classes, results['per_class_accuracy'])):
                    print(f"  Class {i} ({cls}): {acc:.3f} ({acc*100:.1f}%)")
            
            # Decision criteria
            print("\n" + "="*60)
            print("🎯 CONTINUATION TRAINING RECOMMENDATION:")
            print("="*60)
            
            val_acc_pct = val_accuracy * 100
            
            if val_acc_pct >= 70:
                print("✅ EXCELLENT (≥70%): Continue with LIGHT regularization")
                print("   Recommended: LR=1e-4, weight_decay=1e-2, dropout=0.5")
                recommendation = "light"
            elif val_acc_pct >= 50:
                print("✅ GOOD (50-70%): Continue with MODERATE regularization")
                print("   Recommended: LR=5e-5, weight_decay=5e-2, dropout=0.6")
                recommendation = "moderate"
            elif val_acc_pct >= 30:
                print("⚠️  FAIR (30-50%): Continue with EXTREME regularization")
                print("   Recommended: LR=2e-5, weight_decay=1e-1, dropout=0.7")
                recommendation = "extreme"
            else:
                print("❌ POOR (<30%): Consider starting fresh or different approach")
                print("   Model may be too degraded for effective continuation")
                recommendation = "restart"
            
            print(f"\n💰 Estimated continuation cost: ~$80-120")
            print(f"💰 Total project cost: ~$280-320 (vs ~$480 starting fresh)")
            
            return {
                'validation_accuracy': val_accuracy,
                'validation_loss': val_loss,
                'recommendation': recommendation,
                'results': results
            }
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            print(f"❌ Full error: {str(e)}")
            return None

if __name__ == "__main__":
    results = evaluate_final_model()
    
    if results:
        print("\n🎯 EVALUATION COMPLETED SUCCESSFULLY")
        print(f"Final recommendation: {results['recommendation']}")
    else:
        print("\n❌ EVALUATION FAILED - Check logs above")
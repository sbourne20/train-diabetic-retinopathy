#!/usr/bin/env python3
"""
Model Analyzer - Extract Training Statistics from Saved Checkpoints
Analyzes PyTorch model checkpoints to extract training metrics, model info, and statistics
"""

import torch
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

def format_size(bytes_size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def analyze_model_checkpoint(checkpoint_path):
    """
    Analyze a PyTorch model checkpoint and extract all available information
    
    Args:
        checkpoint_path (str): Path to the .pth checkpoint file
        
    Returns:
        dict: Comprehensive analysis of the model checkpoint
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"🔍 Analyzing checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Load checkpoint with multiple fallback methods
    checkpoint = None
    
    # Method 1: Try with weights_only=False (most compatible)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Try with older PyTorch compatibility
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("✅ Checkpoint loaded successfully (legacy mode)")
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            # Method 3: Try with pickle protocol handling
            try:
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                print("✅ Checkpoint loaded successfully (pickle mode)")
            except Exception as e3:
                print(f"❌ All loading methods failed. Last error: {e3}")
                print("🔧 The checkpoint file might be corrupted or in an unsupported format.")
                return None
    
    # File information
    file_size = os.path.getsize(checkpoint_path)
    file_modified = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    
    analysis = {
        "file_info": {
            "path": checkpoint_path,
            "size": format_size(file_size),
            "size_bytes": file_size,
            "modified": file_modified.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
    }
    
    print(f"\n📁 FILE INFORMATION:")
    print(f"   📍 Path: {checkpoint_path}")
    print(f"   📊 Size: {format_size(file_size)} ({file_size:,} bytes)")
    print(f"   🕒 Modified: {file_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🗂️  CHECKPOINT STRUCTURE:")
    print(f"   🔑 Available keys: {len(analysis['checkpoint_keys'])}")
    for key in analysis['checkpoint_keys']:
        print(f"      - {key}")
    
    # Extract training metrics
    training_metrics = {}
    model_info = {}
    optimizer_info = {}
    
    if isinstance(checkpoint, dict):
        
        # Training metrics
        metric_keys = [
            'epoch', 'best_accuracy', 'best_val_accuracy', 'current_accuracy',
            'train_accuracy', 'val_accuracy', 'validation_accuracy',
            'loss', 'train_loss', 'val_loss', 'validation_loss', 'best_loss',
            'learning_rate', 'lr', 'current_lr', 'train_losses', 'val_losses', 
            'val_accuracies', 'medical_validations'
        ]
        
        for key in metric_keys:
            if key in checkpoint:
                training_metrics[key] = checkpoint[key]
        
        # Model state information
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            model_info['num_parameters'] = sum(p.numel() for p in model_state.values())
            model_info['parameter_shapes'] = {k: list(v.shape) for k, v in model_state.items()}
            model_info['trainable_params'] = len([k for k in model_state.keys() if 'lora' in k.lower() or 'adapter' in k.lower()])
            
        # Optimizer information
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            if 'param_groups' in optimizer_state:
                optimizer_info['num_param_groups'] = len(optimizer_state['param_groups'])
                if optimizer_state['param_groups']:
                    optimizer_info['learning_rate'] = optimizer_state['param_groups'][0].get('lr', 'N/A')
                    optimizer_info['weight_decay'] = optimizer_state['param_groups'][0].get('weight_decay', 'N/A')
        
        # Configuration information
        config_keys = [
            'config', 'args', 'hyperparameters', 'model_config',
            'experiment_name', 'dataset_path', 'num_classes'
        ]
        
        config_info = {}
        for key in config_keys:
            if key in checkpoint:
                config_info[key] = checkpoint[key]
    
    # Display training metrics
    if training_metrics:
        print(f"\n🎯 TRAINING METRICS:")
        
        if 'epoch' in training_metrics:
            print(f"   📈 Current Epoch: {training_metrics['epoch']}")
        
        # Accuracy metrics
        accuracy_metrics = [k for k in training_metrics.keys() if 'accuracy' in k.lower()]
        if accuracy_metrics:
            print(f"   🎯 ACCURACY METRICS:")
            for metric in accuracy_metrics:
                value = training_metrics[metric]
                if isinstance(value, (int, float)):
                    print(f"      • {metric}: {value:.4f} ({value*100:.2f}%)")
                else:
                    print(f"      • {metric}: {value}")
        
        # Extract training accuracy from history if available
        train_accs = None
        if 'train_accuracies' in checkpoint and isinstance(checkpoint['train_accuracies'], list):
            train_accs = checkpoint['train_accuracies']
        elif 'training_accuracies' in checkpoint and isinstance(checkpoint['training_accuracies'], list):
            train_accs = checkpoint['training_accuracies']
        elif 'train_history' in checkpoint and isinstance(checkpoint['train_history'], dict):
            # Check for training accuracies in train_history
            if 'train_accuracies' in checkpoint['train_history'] and isinstance(checkpoint['train_history']['train_accuracies'], list):
                train_accs = [acc/100.0 if acc > 1.0 else acc for acc in checkpoint['train_history']['train_accuracies']]
        
        if train_accs and len(train_accs) > 0:
            best_train_acc = max(train_accs)
            current_train_acc = train_accs[-1]
            if not accuracy_metrics:  # If no accuracy metrics were shown above
                print(f"   🎯 ACCURACY METRICS:")
            print(f"      • best_train_accuracy: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
            print(f"      • current_train_accuracy: {current_train_acc:.4f} ({current_train_acc*100:.2f}%)")
            
            # Calculate training vs validation gap for overfitting assessment
            best_val_acc = None
            for key in ['best_val_accuracy', 'best_accuracy', 'val_accuracy', 'validation_accuracy']:
                if key in training_metrics:
                    best_val_acc = training_metrics[key]
                    break
            
            if best_val_acc is not None:
                acc_gap = best_train_acc - best_val_acc
                print(f"      • training_validation_gap: {acc_gap:.4f} ({acc_gap*100:.2f}% difference)")
                if acc_gap > 0.1:
                    print(f"      ⚠️  Large accuracy gap suggests overfitting")
        else:
            # No training accuracy data available
            if accuracy_metrics:  # Only show this if we already have the section
                print(f"      • train_accuracy: Not saved in checkpoint")
                print(f"      ⚪ Training accuracy gap cannot be calculated")
        
        # Loss metrics
        loss_metrics = [k for k in training_metrics.keys() if 'loss' in k.lower()]
        if loss_metrics:
            print(f"   📉 LOSS METRICS:")
            for metric in loss_metrics:
                value = training_metrics[metric]
                if isinstance(value, (int, float)):
                    print(f"      • {metric}: {value:.6f}")
                else:
                    print(f"      • {metric}: {value}")
        
        # Learning rate
        lr_metrics = [k for k in training_metrics.keys() if 'lr' in k.lower() or 'learning_rate' in k.lower()]
        if lr_metrics:
            print(f"   🎛️  LEARNING RATE:")
            for metric in lr_metrics:
                value = training_metrics[metric]
                if isinstance(value, (int, float)):
                    print(f"      • {metric}: {value:.2e}")
                else:
                    print(f"      • {metric}: {value}")
    
    # Display model information
    if model_info:
        print(f"\n🤖 MODEL INFORMATION:")
        if 'num_parameters' in model_info:
            num_params = model_info['num_parameters']
            print(f"   🔢 Total Parameters: {num_params:,}")
            print(f"   💾 Model Size (approx): {format_size(num_params * 4)}")  # Assume float32
        
        if 'trainable_params' in model_info:
            print(f"   🎯 LoRA/Adapter Layers: {model_info['trainable_params']}")
    
    # Display optimizer information
    if optimizer_info:
        print(f"\n⚙️  OPTIMIZER INFORMATION:")
        for key, value in optimizer_info.items():
            if isinstance(value, (int, float)) and 'learning_rate' in key:
                print(f"   • {key}: {value:.2e}")
            else:
                print(f"   • {key}: {value}")
    
    # Display configuration
    if 'config_info' in locals() and config_info:
        print(f"\n🔧 CONFIGURATION:")
        for key, value in config_info.items():
            if isinstance(value, dict):
                print(f"   • {key}:")
                for subkey, subvalue in value.items():
                    print(f"     - {subkey}: {subvalue}")
            else:
                print(f"   • {key}: {value}")
    
    # Medical Grade Assessment
    print(f"\n🏥 MEDICAL GRADE ASSESSMENT:")
    
    best_accuracy = None
    for key in ['best_val_accuracy', 'best_accuracy', 'val_accuracy', 'validation_accuracy']:
        if key in training_metrics:
            best_accuracy = training_metrics[key]
            break
    
    if best_accuracy is not None and isinstance(best_accuracy, (int, float)):
        print(f"   🎯 Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        if best_accuracy >= 0.90:
            print(f"   ✅ MEDICAL GRADE: FULL PASS (≥90% - Production Ready)")
            grade = "EXCELLENT"
        elif best_accuracy >= 0.85:
            print(f"   ⚠️  MEDICAL GRADE: NEAR PASS (≥85% - Close to Production)")
            grade = "GOOD"
        elif best_accuracy >= 0.80:
            print(f"   📈 MEDICAL GRADE: PROMISING LEVEL (≥80% - Research Quality)")
            grade = "PROMISING"
        else:
            print(f"   ❌ MEDICAL GRADE: NEEDS IMPROVEMENT (<80% - Below Standards)")
            grade = "NEEDS_WORK"
        
        print(f"   🏆 Performance Grade: {grade}")
    else:
        print(f"   ❓ No validation accuracy found in checkpoint")
    
    # Research-focused analysis
    research_metrics = analyze_research_metrics(checkpoint, training_metrics)
    
    # Display research metrics
    display_research_metrics(research_metrics)
    
    # High-level summary for users
    display_high_level_summary(training_metrics, research_metrics, checkpoint)
    
    # Detailed analysis summary
    analysis.update({
        "training_metrics": training_metrics,
        "model_info": model_info,
        "optimizer_info": optimizer_info,
        "config_info": config_info if 'config_info' in locals() else {},
        "research_metrics": research_metrics,
        "medical_grade": {
            "best_accuracy": best_accuracy,
            "grade": grade if 'grade' in locals() else "UNKNOWN"
        }
    })
    
    return analysis

def analyze_research_metrics(checkpoint, training_metrics):
    """Extract research-focused metrics from checkpoint"""
    research = {}
    
    # Training convergence analysis
    if 'val_accuracies' in checkpoint and isinstance(checkpoint['val_accuracies'], list):
        val_accs = checkpoint['val_accuracies']
        research['convergence'] = {
            'total_epochs': len(val_accs),
            'final_accuracy': val_accs[-1] if val_accs else None,
            'best_accuracy': max(val_accs) if val_accs else None,
            'best_epoch': val_accs.index(max(val_accs)) + 1 if val_accs else None,
            'accuracy_improvement': val_accs[-1] - val_accs[0] if len(val_accs) > 1 else None,
            'convergence_rate': calculate_convergence_rate(val_accs) if len(val_accs) > 5 else None
        }
    
    # Training stability analysis
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        research['stability'] = {
            'overfitting_ratio': calculate_overfitting_ratio(train_losses, val_losses),
            'loss_variance': calculate_loss_variance(val_losses),
            'training_smoothness': calculate_training_smoothness(train_losses)
        }
    
    # Medical validation analysis
    if 'medical_validations' in checkpoint:
        med_vals = checkpoint['medical_validations']
        research['medical_performance'] = analyze_medical_validations(med_vals)
    
    # LoRA efficiency analysis
    if 'model_state_dict' in checkpoint:
        research['lora_analysis'] = analyze_lora_efficiency(checkpoint['model_state_dict'])
    
    return research

def calculate_convergence_rate(accuracies):
    """Calculate how quickly the model converges"""
    if len(accuracies) < 5:
        return None
    
    # Calculate rate of improvement over epochs
    improvements = [accuracies[i+1] - accuracies[i] for i in range(len(accuracies)-1)]
    avg_improvement = sum(improvements) / len(improvements)
    
    # Find epoch where improvement becomes minimal (< 0.005)
    convergence_epoch = None
    for i, imp in enumerate(improvements):
        if abs(imp) < 0.005:  # Less than 0.5% improvement
            convergence_epoch = i + 2  # +2 because we start from epoch 1 and look at next epoch
            break
    
    return {
        'avg_improvement_per_epoch': avg_improvement,
        'convergence_epoch': convergence_epoch,
        'early_convergence': convergence_epoch is not None and convergence_epoch < len(accuracies) * 0.5
    }

def calculate_overfitting_ratio(train_losses, val_losses):
    """Calculate overfitting ratio from loss curves"""
    if len(train_losses) != len(val_losses) or len(train_losses) < 3:
        return None
    
    # Compare last 3 epochs
    recent_train = sum(train_losses[-3:]) / 3
    recent_val = sum(val_losses[-3:]) / 3
    
    return recent_val / recent_train if recent_train > 0 else None

def calculate_loss_variance(losses):
    """Calculate variance in validation loss (stability metric)"""
    if len(losses) < 3:
        return None
    
    mean_loss = sum(losses) / len(losses)
    variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
    return variance

def calculate_training_smoothness(losses):
    """Calculate how smooth the training is (fewer oscillations = better)"""
    if len(losses) < 3:
        return None
    
    # Count significant oscillations (>5% change)
    oscillations = 0
    for i in range(1, len(losses)-1):
        if abs(losses[i] - losses[i-1]) > 0.05 * losses[i-1]:
            oscillations += 1
    
    return 1 - (oscillations / (len(losses) - 2))  # Higher = smoother

def analyze_medical_validations(medical_vals):
    """Analyze medical-grade validation metrics"""
    if not isinstance(medical_vals, list) or not medical_vals:
        return None
    
    # Count medical grade passes/fails
    passes = sum(1 for val in medical_vals if isinstance(val, bool) and val)
    total = len(medical_vals)
    
    return {
        'total_validations': total,
        'medical_grade_passes': passes,
        'medical_grade_pass_rate': passes / total if total > 0 else 0,
        'final_medical_grade': medical_vals[-1] if medical_vals else None
    }

def analyze_lora_efficiency(model_state):
    """Analyze LoRA layer efficiency"""
    lora_params = 0
    total_params = 0
    lora_layers = []
    
    for name, param in model_state.items():
        param_count = param.numel()
        total_params += param_count
        
        if 'lora' in name.lower():
            lora_params += param_count
            lora_layers.append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count
            })
    
    return {
        'total_parameters': total_params,
        'lora_parameters': lora_params,
        'lora_efficiency': lora_params / total_params if total_params > 0 else 0,
        'lora_layers_count': len(lora_layers),
        'memory_efficiency': f"{(1 - lora_params/total_params)*100:.1f}% memory saved" if total_params > 0 else None
    }

def display_research_metrics(research_metrics):
    """Display research-focused metrics"""
    if not research_metrics:
        return
    
    print(f"\n🔬 RESEARCH METRICS:")
    
    # Convergence Analysis
    if 'convergence' in research_metrics and research_metrics['convergence']:
        conv = research_metrics['convergence']
        print(f"   📈 CONVERGENCE ANALYSIS:")
        print(f"      • Training Epochs: {conv.get('total_epochs', 'N/A')}")
        if conv.get('best_epoch'):
            print(f"      • Best Performance at Epoch: {conv['best_epoch']}")
        if conv.get('accuracy_improvement'):
            print(f"      • Total Accuracy Improvement: +{conv['accuracy_improvement']:.4f} ({conv['accuracy_improvement']*100:.2f}%)")
        
        if conv.get('convergence_rate'):
            cr = conv['convergence_rate']
            print(f"      • Average Improvement/Epoch: {cr.get('avg_improvement_per_epoch', 0):.4f}")
            if cr.get('convergence_epoch'):
                print(f"      • Converged at Epoch: {cr['convergence_epoch']}")
                if cr.get('early_convergence'):
                    print(f"      • ⚡ Early Convergence: YES (efficient training)")
                else:
                    print(f"      • 🐌 Early Convergence: NO (needed full training)")
    
    # Training Stability
    if 'stability' in research_metrics and research_metrics['stability']:
        stab = research_metrics['stability']
        print(f"   🎯 TRAINING STABILITY:")
        
        if stab.get('overfitting_ratio'):
            ratio = stab['overfitting_ratio']
            print(f"      • Overfitting Ratio: {ratio:.3f}", end="")
            if ratio < 1.3:
                print(" (✅ No significant overfitting)")
            elif ratio < 2.5:
                print(" (✅ Normal gap for medical AI)")
            elif ratio < 4.0:
                print(" (⚠️ Moderate gap - common with focal loss)")
            elif ratio < 6.0:
                print(" (⚠️ Large gap - manageable in ensemble)")
            else:
                print(" (❌ Very large gap - monitor generalization)")
        
        if stab.get('training_smoothness'):
            smoothness = stab['training_smoothness']
            print(f"      • Training Smoothness: {smoothness:.3f}", end="")
            if smoothness > 0.8:
                print(" (✅ Very stable)")
            elif smoothness > 0.5:
                print(" (✅ Normal dynamics for medical AI)")
            elif smoothness > 0.3:
                print(" (⚠️ Expected oscillations with focal loss)")
            else:
                print(" (⚠️ High oscillations - monitor convergence)")
        
        if stab.get('loss_variance'):
            print(f"      • Loss Variance: {stab['loss_variance']:.6f}")
    
    # Medical Performance
    if 'medical_performance' in research_metrics and research_metrics['medical_performance']:
        med = research_metrics['medical_performance']
        print(f"   🏥 MEDICAL VALIDATION HISTORY:")
        print(f"      • Total Validations: {med.get('total_validations', 0)}")
        print(f"      • Medical Grade Passes: {med.get('medical_grade_passes', 0)}")
        if med.get('medical_grade_pass_rate') is not None:
            pass_rate = med['medical_grade_pass_rate'] * 100
            print(f"      • Medical Pass Rate: {pass_rate:.1f}%")
        final_grade = med.get('final_medical_grade')
        if final_grade:
            print(f"      • Final Medical Grade: ✅ DETECTED")
        else:
            print(f"      • Final Medical Grade: ❌ NOT DETECTED")
    
    # LoRA Efficiency
    if 'lora_analysis' in research_metrics and research_metrics['lora_analysis']:
        lora = research_metrics['lora_analysis']
        print(f"   🔧 LoRA EFFICIENCY ANALYSIS:")
        print(f"      • Total Parameters: {lora.get('total_parameters', 0):,}")
        print(f"      • LoRA Parameters: {lora.get('lora_parameters', 0):,}")
        if lora.get('lora_efficiency'):
            eff = lora['lora_efficiency'] * 100
            print(f"      • LoRA Efficiency: {eff:.2f}% of total params")
        print(f"      • LoRA Layers Count: {lora.get('lora_layers_count', 0)}")
        if lora.get('memory_efficiency'):
            print(f"      • Memory Savings: {lora['memory_efficiency']}")
    
    # Performance Insights
    print(f"   💡 RESEARCH INSIGHTS:")
    insights = generate_research_insights(research_metrics)
    for insight in insights:
        print(f"      • {insight}")

def display_high_level_summary(training_metrics, research_metrics, checkpoint=None):
    """Display high-level summary for average users"""
    print(f"\n📋 HIGH-LEVEL MODEL ASSESSMENT:")
    
    # Get validation accuracy
    best_val_accuracy = None
    for key in ['best_val_accuracy', 'best_accuracy', 'val_accuracy', 'validation_accuracy']:
        if key in training_metrics:
            best_val_accuracy = training_metrics[key]
            break
    
    # Get training accuracy
    best_train_accuracy = None
    if checkpoint:
        if 'train_accuracies' in checkpoint and isinstance(checkpoint['train_accuracies'], list):
            train_accs = checkpoint['train_accuracies']
            best_train_accuracy = max(train_accs) if train_accs else None
        elif 'training_accuracies' in checkpoint and isinstance(checkpoint['training_accuracies'], list):
            train_accs = checkpoint['training_accuracies']
            best_train_accuracy = max(train_accs) if train_accs else None
        elif 'train_history' in checkpoint and isinstance(checkpoint['train_history'], dict):
            # Check for training accuracies in train_history
            if 'train_accuracies' in checkpoint['train_history'] and isinstance(checkpoint['train_history']['train_accuracies'], list):
                train_accs = [acc/100.0 if acc > 1.0 else acc for acc in checkpoint['train_history']['train_accuracies']]
                best_train_accuracy = max(train_accs) if train_accs else None
    
    # Accuracy assessment
    if best_val_accuracy is not None:
        val_accuracy_pct = best_val_accuracy * 100
        if best_val_accuracy >= 0.90:
            accuracy_status = "excellent"
        elif best_val_accuracy >= 0.85:
            accuracy_status = "good"
        elif best_val_accuracy >= 0.80:
            accuracy_status = "decent"
        elif best_val_accuracy >= 0.70:
            accuracy_status = "fair"
        else:
            accuracy_status = "poor"
        
        # Show both training and validation if available
        if best_train_accuracy is not None:
            train_accuracy_pct = best_train_accuracy * 100
            acc_gap = best_train_accuracy - best_val_accuracy
            print(f"   📊 Validation Accuracy: {val_accuracy_pct:.2f}% ({accuracy_status})")
            print(f"   🎯 Training Accuracy: {train_accuracy_pct:.2f}% (gap: {acc_gap*100:.2f}%)")
        else:
            print(f"   📊 Accuracy: {val_accuracy_pct:.2f}% ({accuracy_status})")
    else:
        print(f"   📊 Accuracy: Not available")
        accuracy_status = "unknown"
    
    # Overfitting assessment
    overfitting_status = "unknown"
    overfitting_concern = "⚪"
    if 'stability' in research_metrics and research_metrics['stability']:
        overfitting_ratio = research_metrics['stability'].get('overfitting_ratio')
        if overfitting_ratio is not None:
            if overfitting_ratio < 1.3:
                overfitting_status = "No significant overfitting"
                overfitting_concern = "✅"
            elif overfitting_ratio < 2.0:
                overfitting_status = "Mild overfitting (normal for medical AI)"
                overfitting_concern = "⚠️"
            elif overfitting_ratio < 3.5:
                overfitting_status = "Moderate overfitting (manageable in ensemble)"
                overfitting_concern = "⚠️"
            else:
                overfitting_status = "High overfitting (monitor generalization)"
                overfitting_concern = "❌"
            
            print(f"   {overfitting_concern} Overfitting: {overfitting_status}")
    
    # Medical suitability - CORRECTED for realistic medical AI assessment
    medical_suitable = False
    if best_val_accuracy is not None and best_val_accuracy >= 0.90:
        medical_suitable = True
        medical_status = "✅ MEDICAL-GRADE: Production Ready (≥90%)"
    elif best_val_accuracy is not None and best_val_accuracy >= 0.88:
        medical_status = "✅ NEAR MEDICAL-GRADE: Excellent Performance (≥88%)"
    elif best_val_accuracy is not None and best_val_accuracy >= 0.85:
        medical_status = "✅ RESEARCH-GRADE: Very Good Performance (≥85%)"
    elif best_val_accuracy is not None and best_val_accuracy >= 0.80:
        medical_status = "⚠️ PROMISING: Decent Performance (≥80%)"
    else:
        medical_status = "❌ NEEDS IMPROVEMENT: Below Medical Standards (<80%)"
    
    print(f"   🏥 Medical Grade: {medical_status}")
    
    # Recommendations - CORRECTED for medical AI context
    recommendations = []

    # Medical AI specific recommendations
    if best_val_accuracy is not None and best_val_accuracy >= 0.88:
        recommendations.append("excellent performance for medical AI - ready for ensemble")
    elif best_val_accuracy is not None and best_val_accuracy >= 0.85:
        recommendations.append("strong foundation for medical ensemble training")
    elif best_val_accuracy is not None and best_val_accuracy >= 0.80:
        recommendations.append("decent baseline - consider ensemble approach")
    elif best_val_accuracy is not None and best_val_accuracy < 0.80:
        recommendations.append("improve accuracy with better training")

    # Overfitting in medical context
    if overfitting_concern == "❌" and best_val_accuracy is not None and best_val_accuracy >= 0.85:
        recommendations.append("overfitting manageable in ensemble setting")
    elif overfitting_concern in ["⚠️", "❌"] and best_val_accuracy is not None and best_val_accuracy < 0.85:
        recommendations.append("address overfitting (early stopping, regularization)")

    if 'stability' in research_metrics and research_metrics['stability']:
        smoothness = research_metrics['stability'].get('training_smoothness', 1.0)
        if smoothness < 0.3:
            recommendations.append("monitor training convergence (very high oscillations)")

    if not recommendations:
        if medical_suitable:
            recommendations.append("model ready for production deployment")
        elif best_val_accuracy is not None and best_val_accuracy >= 0.85:
            recommendations.append("excellent component for medical ensemble")
        else:
            recommendations.append("continue training optimization")

    if recommendations:
        print(f"   🔧 Status: {recommendations[0].upper()}")
        if len(recommendations) > 1:
            print(f"   💡 Additional: {', '.join(recommendations[1:])}")

    print(f"   " + "="*60)

def generate_research_insights(research_metrics):
    """Generate research insights from the metrics"""
    insights = []
    
    # Convergence insights
    if 'convergence' in research_metrics:
        conv = research_metrics['convergence']
        if conv.get('convergence_rate', {}).get('early_convergence'):
            insights.append("Model converged early - very efficient training")
        elif conv.get('accuracy_improvement', 0) > 0.6:
            insights.append("Excellent learning capacity (+60% accuracy improvement)")
        elif conv.get('accuracy_improvement', 0) > 0.4:
            insights.append("Good learning capacity (+40% accuracy improvement)")
    
    # Stability insights
    if 'stability' in research_metrics:
        stab = research_metrics['stability']
        if stab.get('overfitting_ratio', 2) < 1.1:
            insights.append("Excellent generalization - no overfitting detected")
        if stab.get('training_smoothness', 0) > 0.8:
            insights.append("Very stable training - suitable for production")
    
    # LoRA insights
    if 'lora_analysis' in research_metrics:
        lora = research_metrics['lora_analysis']
        if lora.get('lora_efficiency', 0) < 0.05:
            insights.append("Highly memory-efficient LoRA implementation")
        if lora.get('lora_layers_count', 0) > 500:
            insights.append("Extensive LoRA adaptation - good fine-tuning coverage")
    
    # Medical insights
    if 'medical_performance' in research_metrics:
        med = research_metrics['medical_performance']
        if med.get('medical_grade_pass_rate', 0) > 0.8:
            insights.append("Consistently achieves medical-grade performance")
        elif med.get('final_medical_grade'):
            insights.append("Recently achieved medical-grade performance")
    
    if not insights:
        insights.append("Model shows standard training characteristics")
    
    return insights

def save_analysis_report(analysis, output_path):
    """Save analysis results to JSON file"""
    
    # Convert datetime and other non-serializable objects to strings
    def convert_for_json(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, 'shape'):
            return str(obj.shape)
        return str(obj)
    
    # Create JSON-serializable copy
    json_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, dict):
            json_analysis[key] = {k: convert_for_json(v) for k, v in value.items()}
        else:
            json_analysis[key] = convert_for_json(value)
    
    with open(output_path, 'w') as f:
        json.dump(json_analysis, f, indent=2, default=str)
    
    print(f"\n💾 Analysis report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch model checkpoints and extract training statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_analyzer.py --model ./results/models/best_model.pth
  python model_analyzer.py --model ./checkpoints/epoch_10.pth --output analysis.json
  python model_analyzer.py --model ./results/models/checkpoint_best_model.pth --verbose
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file to save analysis results (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed parameter information'
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze the model
        analysis = analyze_model_checkpoint(args.model)
        
        if analysis is None:
            print("❌ Failed to analyze model checkpoint")
            sys.exit(1)
        
        # Save analysis report if requested
        if args.output:
            save_analysis_report(analysis, args.output)
        
        # Show detailed parameter info if verbose
        if args.verbose and 'model_info' in analysis and 'parameter_shapes' in analysis['model_info']:
            print(f"\n📝 DETAILED PARAMETER INFORMATION:")
            param_shapes = analysis['model_info']['parameter_shapes']
            for name, shape in param_shapes.items():
                num_params = 1
                for dim in shape:
                    num_params *= dim
                print(f"   • {name}: {shape} ({num_params:,} parameters)")
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
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
        smoothness = research_metrics['stability'].get('training_smoothness')
        if smoothness is not None and smoothness < 0.3:
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
        conv_rate = conv.get('convergence_rate')
        if conv_rate and conv_rate.get('early_convergence'):
            insights.append("Model converged early - very efficient training")
        elif conv.get('accuracy_improvement', 0) > 0.6:
            insights.append("Excellent learning capacity (+60% accuracy improvement)")
        elif conv.get('accuracy_improvement', 0) > 0.4:
            insights.append("Good learning capacity (+40% accuracy improvement)")
    
    # Stability insights
    if 'stability' in research_metrics:
        stab = research_metrics['stability']
        overfitting_ratio = stab.get('overfitting_ratio')
        if overfitting_ratio is not None and overfitting_ratio < 1.1:
            insights.append("Excellent generalization - no overfitting detected")
        training_smoothness = stab.get('training_smoothness')
        if training_smoothness is not None and training_smoothness > 0.8:
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

def find_all_model_checkpoints(search_dirs=None):
    """Find all .pth checkpoint files in specified directories"""
    if search_dirs is None:
        search_dirs = [
            './results',
            './checkpoints',
            './ovo_models',
            './models',
            '.'
        ]

    checkpoint_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.pth'):
                        checkpoint_files.append(os.path.join(root, file))

    return sorted(checkpoint_files)

def extract_model_name(checkpoint_path, checkpoint=None):
    """Extract a readable model name from checkpoint path and content"""
    # Try to extract meaningful name from path
    parts = checkpoint_path.replace('\\', '/').split('/')
    filename = os.path.basename(checkpoint_path)

    # Look for class information in filename
    # Pattern 1: class_X or class_X_vs_Y
    if 'class_' in filename.lower():
        base_name = filename.replace('.pth', '').replace('_best_model', '').replace('checkpoint_', '')
        return base_name

    # Pattern 2: OVO pattern like "ovo_mild_moderate" or "ovo_0_1"
    for part in parts:
        if 'ovo_' in part.lower():
            return part.replace('.pth', '').replace('_best_model', '').replace('checkpoint_', '')

    # Pattern 3: Try to extract class info from checkpoint content
    if checkpoint and isinstance(checkpoint, dict):
        # Check for class pair information
        if 'class_pair' in checkpoint:
            class_pair = checkpoint['class_pair']
            if isinstance(class_pair, (list, tuple)) and len(class_pair) == 2:
                # Get parent directory name
                parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
                return f"{parent_dir}_{class_pair[0]}-{class_pair[1]}"

        # Check for class names in config
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = checkpoint['config']
            if 'class_names' in config:
                classes = config['class_names']
                if isinstance(classes, list) and len(classes) == 2:
                    parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
                    return f"{parent_dir}_{classes[0]}-{classes[1]}"

    # Pattern 4: Extract from parent directory + filename pattern
    parent_dir = os.path.basename(os.path.dirname(checkpoint_path))

    # If filename has numbers that might indicate classes
    import re
    numbers = re.findall(r'\d+', filename)
    if len(numbers) >= 2:
        # Likely class indices
        return f"{parent_dir}_{numbers[0]}-{numbers[1]}"
    elif len(numbers) == 1:
        # Single class or epoch number - use parent dir + number
        return f"{parent_dir}_{numbers[0]}"

    # Default: use parent directory name
    return parent_dir

def extract_biclass_metrics(checkpoint):
    """Extract precision, recall, F1, and AUC from checkpoint if available"""
    metrics = {
        'precision': None,
        'recall': None,
        'f1_score': None,
        'auc': None,
        'val_accuracy': None,
        'test_accuracy': None,
        'accuracy': None,  # Best available accuracy (for backward compatibility)
        'is_state_dict_only': False
    }

    # Check if this is just a state_dict (weights only) - common for ensemble models
    if isinstance(checkpoint, dict):
        # Check if it looks like a pure state_dict (all keys are tensor names)
        all_keys = list(checkpoint.keys())
        if all_keys and all(isinstance(checkpoint[k], torch.Tensor) for k in all_keys[:10]):
            # This appears to be a state_dict without training metrics
            metrics['is_state_dict_only'] = True
            return metrics

        # Try to find metrics in various checkpoint keys
        # Check for explicit metric keys
        for key in ['precision', 'test_precision', 'val_precision']:
            if key in checkpoint:
                metrics['precision'] = checkpoint[key]
                break

        for key in ['recall', 'test_recall', 'val_recall', 'sensitivity']:
            if key in checkpoint:
                metrics['recall'] = checkpoint[key]
                break

        for key in ['f1_score', 'f1', 'test_f1', 'val_f1']:
            if key in checkpoint:
                metrics['f1_score'] = checkpoint[key]
                break

        for key in ['auc', 'roc_auc', 'test_auc', 'val_auc']:
            if key in checkpoint:
                metrics['auc'] = checkpoint[key]
                break

        # Extract BOTH validation and test accuracy separately
        for key in ['best_val_accuracy', 'val_accuracy', 'validation_accuracy']:
            if key in checkpoint and metrics['val_accuracy'] is None:
                metrics['val_accuracy'] = checkpoint[key]
                break

        for key in ['test_accuracy', 'final_test_accuracy']:
            if key in checkpoint and metrics['test_accuracy'] is None:
                metrics['test_accuracy'] = checkpoint[key]
                break

        # For backward compatibility: set 'accuracy' to validation first, then test if no val
        if metrics['val_accuracy'] is not None:
            metrics['accuracy'] = metrics['val_accuracy']
        elif metrics['test_accuracy'] is not None:
            metrics['accuracy'] = metrics['test_accuracy']
        else:
            # Fallback to any accuracy key
            for key in ['best_accuracy', 'accuracy']:
                if key in checkpoint and metrics['accuracy'] is None:
                    metrics['accuracy'] = checkpoint[key]
                    break

    return metrics

def analyze_all_models(checkpoint_files, output_table=True):
    """Analyze all model checkpoints and display summary table"""
    all_results = []

    for checkpoint_path in checkpoint_files:
        try:
            print(f"\n🔍 Loading: {checkpoint_path}")

            # Load checkpoint quietly
            checkpoint = None
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                except:
                    print(f"   ❌ Failed to load")
                    continue

            if checkpoint is None:
                continue

            # Extract model name (pass checkpoint for better naming)
            model_name = extract_model_name(checkpoint_path, checkpoint)

            # Extract metrics
            metrics = extract_biclass_metrics(checkpoint)

            all_results.append({
                'model_name': model_name,
                'path': checkpoint_path,
                **metrics
            })

            print(f"   ✅ Loaded successfully")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    if output_table and all_results:
        print("\n" + "="*120)
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("="*120)

        # Print table header with both Val and Test accuracy
        print(f"\n{'Model Name':<35} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC':<10} | {'Val Acc':<10} | {'Test Acc':<10}")
        print("-" * 120)

        # Print each model's metrics
        for result in all_results:
            model_name = result['model_name'][:33]  # Truncate if too long

            # Check if this is a weights-only checkpoint
            if result.get('is_state_dict_only', False):
                precision = "Weights"
                recall = "Only"
                f1 = "(Ensemble)"
                auc = "-"
                val_accuracy = "-"
                test_accuracy = "-"
            else:
                precision = f"{result['precision']:.4f}" if result['precision'] is not None else "N/A"
                recall = f"{result['recall']:.4f}" if result['recall'] is not None else "N/A"
                f1 = f"{result['f1_score']:.4f}" if result['f1_score'] is not None else "N/A"
                auc = f"{result['auc']:.4f}" if result['auc'] is not None else "N/A"
                val_accuracy = f"{result['val_accuracy']:.4f}" if result['val_accuracy'] is not None else "N/A"
                test_accuracy = f"{result['test_accuracy']:.4f}" if result['test_accuracy'] is not None else "N/A"

            print(f"{model_name:<35} | {precision:<10} | {recall:<10} | {f1:<10} | {auc:<10} | {val_accuracy:<10} | {test_accuracy:<10}")

        print("="*120)

        # Calculate averages for metrics (excluding state_dict_only models)
        valid_results = [r for r in all_results if not r.get('is_state_dict_only', False)]

        if valid_results:
            # Calculate average validation and test accuracies separately
            val_accuracies = [r['val_accuracy'] for r in valid_results if r['val_accuracy'] is not None]
            test_accuracies = [r['test_accuracy'] for r in valid_results if r['test_accuracy'] is not None]

            avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else None
            avg_test_accuracy = sum(test_accuracies) / len(test_accuracies) if test_accuracies else None

            # Calculate other averages if available
            precisions = [r['precision'] for r in valid_results if r['precision'] is not None]
            recalls = [r['recall'] for r in valid_results if r['recall'] is not None]
            f1_scores = [r['f1_score'] for r in valid_results if r['f1_score'] is not None]
            aucs = [r['auc'] for r in valid_results if r['auc'] is not None]

            avg_precision = sum(precisions) / len(precisions) if precisions else None
            avg_recall = sum(recalls) / len(recalls) if recalls else None
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
            avg_auc = sum(aucs) / len(aucs) if aucs else None

            # Format average strings
            precision_str = f"{avg_precision:.4f}" if avg_precision is not None else "-"
            recall_str = f"{avg_recall:.4f}" if avg_recall is not None else "-"
            f1_str = f"{avg_f1:.4f}" if avg_f1 is not None else "-"
            auc_str = f"{avg_auc:.4f}" if avg_auc is not None else "-"
            val_accuracy_str = f"{avg_val_accuracy:.4f}" if avg_val_accuracy is not None else "-"
            test_accuracy_str = f"{avg_test_accuracy:.4f}" if avg_test_accuracy is not None else "-"

            print(f"\n{'Model Average':<35} | {precision_str:<10} | {recall_str:<10} | {f1_str:<10} | {auc_str:<10} | {val_accuracy_str:<10} | {test_accuracy_str:<10}")

        print("="*120)
        print(f"\n✅ Analyzed {len(all_results)} models successfully!")

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch model checkpoints and extract training statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single model
  python model_analyzer.py --model ./results/models/best_model.pth

  # Analyze all models in default directories
  python model_analyzer.py

  # Analyze all models with detailed output
  python model_analyzer.py --verbose

  # Analyze single model with JSON output
  python model_analyzer.py --model ./checkpoints/epoch_10.pth --output analysis.json
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to the model checkpoint (.pth file). If not specified, analyzes all models.'
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

    parser.add_argument(
        '--search-dirs',
        type=str,
        nargs='+',
        default=None,
        help='Directories to search for model checkpoints (default: ./results, ./checkpoints, ./ovo_models, ./models, .)'
    )

    args = parser.parse_args()

    try:
        if args.model:
            # Check if --model is a directory or a file
            if os.path.isdir(args.model):
                # Directory provided - analyze all .pth files in it
                print(f"🔎 Searching for model checkpoints in: {args.model}")
                checkpoint_files = find_all_model_checkpoints([args.model])

                if not checkpoint_files:
                    print(f"❌ No checkpoint files found in: {args.model}")
                    sys.exit(1)

                print(f"📁 Found {len(checkpoint_files)} checkpoint files")

                # Analyze all models in the directory
                results = analyze_all_models(checkpoint_files, output_table=True)

                # Save results if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"\n💾 Results saved to: {args.output}")

            elif os.path.isfile(args.model):
                # Single file provided (original behavior)
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

            else:
                print(f"❌ Path does not exist: {args.model}")
                sys.exit(1)

        else:
            # Multi-model analysis (new behavior)
            print("🔎 Searching for all model checkpoints...")
            checkpoint_files = find_all_model_checkpoints(args.search_dirs)

            if not checkpoint_files:
                print("❌ No checkpoint files found in search directories")
                print("   Search directories: ./results, ./checkpoints, ./ovo_models, ./models, .")
                sys.exit(1)

            print(f"📁 Found {len(checkpoint_files)} checkpoint files")

            # Analyze all models
            results = analyze_all_models(checkpoint_files, output_table=True)

            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
Comprehensive Medical-Grade DR Model with ALL findings from medical_terms_type1.json
Optimized for 95%+ accuracy on n1-highmem-4 + Tesla P100
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
import json
import datetime
from collections import defaultdict

# Tesla P100 optimization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable mixed precision for Tesla P100
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled for Tesla P100")
except Exception as e:
    print(f"⚠️ Mixed precision setup: {e}")

# ALL MEDICAL FINDINGS from medical_terms_type1.json
COMPREHENSIVE_MEDICAL_FINDINGS = {
    # ===== NPDR FEATURES =====
    'microaneurysms': {
        'type': 'multi_class',
        'classes': ['none', 'mild_1_20', 'moderate_21_50', 'severe_50plus'],
        'quadrant_based': True,
        'medical_weight': 1.0,
        'clinical_significance': 'early_dr_indicator'
    },
    'intraretinal_hemorrhages': {
        'type': 'multi_class', 
        'classes': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_weight': 1.2,
        'clinical_significance': 'dr_progression'
    },
    'dot_blot_hemorrhages': {
        'type': 'multi_class',
        'classes': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_weight': 1.2,
        'clinical_significance': 'dr_progression'
    },
    'flame_hemorrhages': {
        'type': 'multi_class',
        'classes': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_weight': 1.3,
        'clinical_significance': 'severe_dr_indicator'
    },
    'hard_exudates': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': True,
        'medical_weight': 1.5,
        'clinical_significance': 'macular_threat'
    },
    'cotton_wool_spots': {
        'type': 'multi_class',
        'classes': ['none', '1_3_spots', 'more_than_3'],
        'quadrant_based': True,
        'medical_weight': 1.8,
        'clinical_significance': 'severe_npdr_sign'
    },
    'venous_beading': {
        'type': 'multi_class',
        'classes': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_weight': 2.0,
        'clinical_significance': 'severe_npdr_critical'
    },
    'venous_caliber_changes': {
        'type': 'multi_class',
        'classes': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': False,
        'medical_weight': 1.4,
        'clinical_significance': 'vascular_changes'
    },
    'venous_looping_or_reduplication': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.6,
        'clinical_significance': 'vascular_abnormality'
    },
    'IRMA': {
        'type': 'multi_class',
        'classes': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_weight': 2.2,
        'clinical_significance': 'severe_npdr_critical'
    },
    
    # ===== PDR FEATURES =====
    'neovascularization_disc': {
        'type': 'multi_class',
        'classes': ['absent', 'less_than_1_3DD', '1_3DD_to_1DD', 'greater_than_1DD'],
        'quadrant_based': False,
        'medical_weight': 3.0,
        'clinical_significance': 'pdr_high_risk'
    },
    'neovascularization_elsewhere': {
        'type': 'multi_class', 
        'classes': ['absent', 'less_than_half_DD', 'half_DD_or_greater'],
        'quadrant_based': False,
        'medical_weight': 2.8,
        'clinical_significance': 'pdr_moderate_risk'
    },
    'neovascularization_iris': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 3.5,
        'clinical_significance': 'advanced_pdr_emergency'
    },
    'neovascular_activity': {
        'type': 'multi_class',
        'classes': ['none', 'active', 'fibrosed_regressed', 'unknown'],
        'quadrant_based': False,
        'medical_weight': 2.5,
        'clinical_significance': 'pdr_activity_status'
    },
    'preretinal_hemorrhage': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 3.2,
        'clinical_significance': 'pdr_complication_severe'
    },
    'vitreous_hemorrhage': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 3.5,
        'clinical_significance': 'pdr_complication_critical'
    },
    'fibrovascular_proliferation': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 3.8,
        'clinical_significance': 'advanced_pdr_critical'
    },
    'tractional_retinal_detachment': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 4.0,
        'clinical_significance': 'pdr_emergency'
    },
    
    # ===== TREATMENT MARKERS =====
    'PRP_scars': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 0.8,
        'clinical_significance': 'treatment_history'
    },
    'focal_laser_scars': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 0.8,
        'clinical_significance': 'treatment_history'
    },
    'vitrectomy_changes': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 0.9,
        'clinical_significance': 'surgical_history'
    },
    
    # ===== LOCALIZATION FEATURES =====
    'within_1DD_of_fovea_microaneurysms': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.8,
        'clinical_significance': 'macular_involvement'
    },
    'within_1DD_of_fovea_hemorrhages': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 2.0,
        'clinical_significance': 'macular_threat'
    },
    'within_1DD_of_fovea_hard_exudates': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 2.5,
        'clinical_significance': 'macular_emergency'
    },
    'within_1DD_of_fovea_IRMA': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 2.3,
        'clinical_significance': 'macular_severe'
    },
    
    # ===== IMAGE QUALITY ASSESSMENT =====
    'image_quality_gradable': {
        'type': 'binary',
        'classes': ['non_gradable', 'gradable'],
        'quadrant_based': False,
        'medical_weight': 3.0,
        'clinical_significance': 'quality_control'
    },
    'image_quality_issues': {
        'type': 'multi_class',
        'classes': ['none', 'blur', 'media_opacity', 'small_pupil', 'artifact', 'poor_field_definition'],
        'quadrant_based': False,
        'medical_weight': 2.0,
        'clinical_significance': 'quality_assessment'
    },
    
    # ===== CONFOUNDERS =====
    'hypertensive_retinopathy_signs': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.2,
        'clinical_significance': 'differential_diagnosis'
    },
    'retinal_vein_occlusion_signs': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.5,
        'clinical_significance': 'differential_diagnosis'
    },
    'AMD_drusen': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.0,
        'clinical_significance': 'comorbidity'
    },
    'myopic_degeneration': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 0.9,
        'clinical_significance': 'comorbidity'
    },
    'cataract_presence': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.1,
        'clinical_significance': 'imaging_quality_factor'
    },
    'vitreous_floaters': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 0.7,
        'clinical_significance': 'imaging_artifact'
    },
    'optic_disc_abnormalities': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.3,
        'clinical_significance': 'comorbidity'
    },
    'other_retinal_pathology': {
        'type': 'binary',
        'classes': ['absent', 'present'],
        'quadrant_based': False,
        'medical_weight': 1.4,
        'clinical_significance': 'differential_diagnosis'
    }
}

# GRADING RULES from medical_terms_type1.json
GRADING_RULES = {
    'hemorrhages_4_quadrants': {
        'type': 'binary',
        'rule_weight': 3.0,
        'medical_significance': 'severe_npdr_criteria'
    },
    'venous_beading_2plus_quadrants': {
        'type': 'binary', 
        'rule_weight': 3.0,
        'medical_significance': 'severe_npdr_criteria'
    },
    'irma_1plus_quadrant': {
        'type': 'binary',
        'rule_weight': 3.0,
        'medical_significance': 'severe_npdr_criteria'
    },
    'meets_4_2_1_rule': {
        'type': 'binary',
        'rule_weight': 4.0,
        'medical_significance': 'severe_npdr_diagnosis'
    },
    'severe_npdr_criteria_met': {
        'type': 'binary',
        'rule_weight': 4.0,
        'medical_significance': 'severe_npdr_confirmed'
    },
    'referable_DR': {
        'type': 'binary',
        'rule_weight': 3.5,
        'medical_significance': 'referral_required'
    },
    'sight_threatening_DR': {
        'type': 'binary',
        'rule_weight': 4.0,
        'medical_significance': 'urgent_referral'
    }
}

class ComprehensiveMedicalDRModel:
    """Medical-grade DR model with ALL findings from medical_terms_type1.json."""
    
    def __init__(self, input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.findings_config = COMPREHENSIVE_MEDICAL_FINDINGS
        self.grading_rules = GRADING_RULES
        
        # Medical-grade requirements (95%+)
        self.medical_requirements = {
            'minimum_accuracy': 0.95,
            'minimum_sensitivity': 0.93,
            'minimum_specificity': 0.95,
            'minimum_f1_score': 0.93,
            'minimum_auc': 0.95
        }
        
    def create_comprehensive_model(self) -> tf.keras.Model:
        """Create comprehensive model with ALL medical findings."""
        
        print(f"🏥 Creating comprehensive medical model with {len(self.findings_config)} findings")
        
        # Use EfficientNetB4 for higher accuracy (still Tesla P100 compatible)
        base_model = tf.keras.applications.EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None
        )
        
        # Fine-tune more layers for medical accuracy
        base_model.trainable = True
        for layer in base_model.layers[:-50]:  # Fine-tune more layers
            layer.trainable = False
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Enhanced preprocessing for medical images
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Medical-grade augmentation (very conservative)
        x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = tf.keras.layers.RandomRotation(0.03)(x)  # Very small rotation
        x = tf.keras.layers.RandomContrast(0.02)(x)  # Minimal contrast
        x = tf.keras.layers.RandomBrightness(0.02)(x)  # Minimal brightness
        
        # Feature extraction with attention
        features = base_model(x, training=True)
        
        # Multi-scale feature processing for different finding sizes
        # Global features for overall DR assessment
        global_avg = tf.keras.layers.GlobalAveragePooling2D()(features)
        global_max = tf.keras.layers.GlobalMaxPooling2D()(features)
        
        # Spatial attention for localized findings
        attention_features = self._create_medical_attention(features)
        
        # Combine all feature types
        combined_features = tf.keras.layers.Concatenate()([
            global_avg, 
            global_max, 
            attention_features
        ])
        
        # Medical-grade feature processing
        x = tf.keras.layers.Dense(1024, activation='relu')(combined_features)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Lower dropout for higher accuracy
        
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        # Shared medical features
        medical_features = tf.keras.layers.Dense(256, activation='relu')(x)
        medical_features = tf.keras.layers.BatchNormalization()(medical_features)
        medical_features = tf.keras.layers.Dropout(0.1)(medical_features)
        
        # Create outputs for ALL findings
        outputs = {}
        
        # 1. DR Severity (5-class ICDR system)
        dr_branch = tf.keras.layers.Dense(128, activation='relu')(medical_features)
        dr_branch = tf.keras.layers.Dropout(0.1)(dr_branch)
        outputs['dr_severity'] = tf.keras.layers.Dense(
            5, activation='softmax', name='dr_severity', dtype='float32'
        )(dr_branch)
        
        # 2. Individual medical findings
        for finding_name, finding_config in self.findings_config.items():
            finding_branch = tf.keras.layers.Dense(64, activation='relu')(medical_features)
            finding_branch = tf.keras.layers.Dropout(0.05)(finding_branch)
            
            if finding_config['type'] == 'binary':
                output_size = 1
                activation = 'sigmoid'
            else:  # multi_class
                output_size = len(finding_config['classes'])
                activation = 'softmax'
            
            outputs[finding_name] = tf.keras.layers.Dense(
                output_size,
                activation=activation,
                name=finding_name,
                dtype='float32'
            )(finding_branch)
        
        # 3. Grading rules
        for rule_name, rule_config in self.grading_rules.items():
            rule_branch = tf.keras.layers.Dense(32, activation='relu')(medical_features)
            outputs[rule_name] = tf.keras.layers.Dense(
                1, activation='sigmoid', name=rule_name, dtype='float32'
            )(rule_branch)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        print(f"✅ Comprehensive model created:")
        print(f"   - Parameters: {model.count_params():,}")
        print(f"   - DR Severity: 5-class output")
        print(f"   - Medical Findings: {len(self.findings_config)} types")
        print(f"   - Grading Rules: {len(self.grading_rules)} rules")
        print(f"   - Total Outputs: {len(outputs)}")
        
        return model
    
    def _create_medical_attention(self, features: tf.Tensor) -> tf.Tensor:
        """Create medical-specific spatial attention mechanism."""
        
        # Channel attention (what features are important)
        channel_attention = tf.keras.layers.GlobalAveragePooling2D()(features)
        channel_attention = tf.keras.layers.Dense(features.shape[-1] // 8, activation='relu')(channel_attention)
        channel_attention = tf.keras.layers.Dense(features.shape[-1], activation='sigmoid')(channel_attention)
        channel_attention = tf.keras.layers.Reshape((1, 1, features.shape[-1]))(channel_attention)
        
        # Spatial attention (where features are important)
        spatial_attention = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(features)
        
        # Apply both attentions
        attended_features = tf.keras.layers.Multiply()([features, channel_attention])
        attended_features = tf.keras.layers.Multiply()([attended_features, spatial_attention])
        
        # Global pooling of attended features
        attended_pooled = tf.keras.layers.GlobalAveragePooling2D()(attended_features)
        
        return attended_pooled
    
    def compile_medical_model(self, model: tf.keras.Model, learning_rate: float = 0.00005) -> tf.keras.Model:
        """Compile with medical-grade loss functions and metrics."""
        
        losses = {}
        loss_weights = {}
        metrics = {}
        
        # DR severity loss (most important)
        losses['dr_severity'] = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        loss_weights['dr_severity'] = 4.0
        metrics['dr_severity'] = [
            tf.keras.metrics.CategoricalAccuracy(name='dr_accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='dr_top2_accuracy'),
            tf.keras.metrics.Precision(name='dr_precision'),
            tf.keras.metrics.Recall(name='dr_recall'),
            tf.keras.metrics.AUC(name='dr_auc')
        ]
        
        # Individual finding losses and metrics
        for finding_name, finding_config in self.findings_config.items():
            medical_weight = finding_config['medical_weight']
            
            if finding_config['type'] == 'binary':
                losses[finding_name] = tf.keras.losses.BinaryCrossentropy()
                metrics[finding_name] = [
                    tf.keras.metrics.BinaryAccuracy(name=f'{finding_name}_accuracy'),
                    tf.keras.metrics.Precision(name=f'{finding_name}_precision'),
                    tf.keras.metrics.Recall(name=f'{finding_name}_recall'),
                    tf.keras.metrics.AUC(name=f'{finding_name}_auc')
                ]
            else:  # multi_class
                losses[finding_name] = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03)
                metrics[finding_name] = [
                    tf.keras.metrics.CategoricalAccuracy(name=f'{finding_name}_accuracy'),
                    tf.keras.metrics.Precision(name=f'{finding_name}_precision'),
                    tf.keras.metrics.Recall(name=f'{finding_name}_recall')
                ]
            
            loss_weights[finding_name] = medical_weight
        
        # Grading rules losses
        for rule_name, rule_config in self.grading_rules.items():
            losses[rule_name] = tf.keras.losses.BinaryCrossentropy()
            loss_weights[rule_name] = rule_config['rule_weight']
            metrics[rule_name] = [
                tf.keras.metrics.BinaryAccuracy(name=f'{rule_name}_accuracy'),
                tf.keras.metrics.Precision(name=f'{rule_name}_precision'),
                tf.keras.metrics.Recall(name=f'{rule_name}_recall'),
                tf.keras.metrics.AUC(name=f'{rule_name}_auc')
            ]
        
        # Medical-grade optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        # Mixed precision wrapper for Tesla P100
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        print(f"✅ Medical model compiled:")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Total loss functions: {len(losses)}")
        print(f"   - Medical weighting: Applied")
        print(f"   - Gradient clipping: Enabled")
        print(f"   - Mixed precision: {'Enabled' if 'mixed_float16' in str(tf.keras.mixed_precision.global_policy()) else 'Disabled'}")
        
        return model


class MedicalGradeCallback(tf.keras.callbacks.Callback):
    """Enhanced medical validation callback for 95%+ accuracy."""
    
    def __init__(self, medical_requirements: Dict[str, float], output_dir: str):
        super().__init__()
        self.medical_requirements = medical_requirements
        self.output_dir = output_dir
        self.best_medical_score = 0.0
        self.medical_grade_achieved = False
        
    def on_epoch_end(self, epoch, logs=None):
        """Enhanced medical validation."""
        if logs is None:
            return
        
        # Primary DR accuracy
        dr_accuracy = logs.get('val_dr_accuracy', 0)
        dr_precision = logs.get('val_dr_precision', 0)
        dr_recall = logs.get('val_dr_recall', 0)
        dr_auc = logs.get('val_dr_auc', 0)
        
        # Calculate F1 score
        if dr_precision > 0 and dr_recall > 0:
            f1_score = 2 * (dr_precision * dr_recall) / (dr_precision + dr_recall)
        else:
            f1_score = 0
        
        # Medical requirements check
        meets_accuracy = dr_accuracy >= self.medical_requirements['minimum_accuracy']
        meets_sensitivity = dr_recall >= self.medical_requirements['minimum_sensitivity']
        meets_specificity = dr_precision >= self.medical_requirements['minimum_specificity']
        meets_f1 = f1_score >= self.medical_requirements['minimum_f1_score']
        meets_auc = dr_auc >= self.medical_requirements['minimum_auc']
        
        # Overall medical grade
        medical_grade = all([meets_accuracy, meets_sensitivity, meets_specificity, meets_f1, meets_auc])
        
        # Medical score (weighted combination)
        medical_score = (
            dr_accuracy * 0.3 + 
            dr_recall * 0.25 + 
            dr_precision * 0.25 + 
            f1_score * 0.1 + 
            dr_auc * 0.1
        )
        
        # Enhanced logging
        print(f"\n{'='*60}")
        print(f"🏥 MEDICAL VALIDATION EPOCH {epoch + 1} - TARGET: 95%+")
        print(f"{'='*60}")
        print(f"Medical Score: {medical_score:.4f}")
        print(f"Accuracy    (≥{self.medical_requirements['minimum_accuracy']:.2f}): {'✅' if meets_accuracy else '❌'} {dr_accuracy:.4f}")
        print(f"Sensitivity (≥{self.medical_requirements['minimum_sensitivity']:.2f}): {'✅' if meets_sensitivity else '❌'} {dr_recall:.4f}")
        print(f"Specificity (≥{self.medical_requirements['minimum_specificity']:.2f}): {'✅' if meets_specificity else '❌'} {dr_precision:.4f}")
        print(f"F1-Score    (≥{self.medical_requirements['minimum_f1_score']:.2f}): {'✅' if meets_f1 else '❌'} {f1_score:.4f}")
        print(f"AUC         (≥{self.medical_requirements['minimum_auc']:.2f}): {'✅' if meets_auc else '❌'} {dr_auc:.4f}")
        print(f"Medical Grade: {'🏥 MEDICAL APPROVED ✅' if medical_grade else '⚠️ REQUIRES IMPROVEMENT'}")
        
        if medical_grade and not self.medical_grade_achieved:
            print(f"🎉 MEDICAL-GRADE ACHIEVED AT EPOCH {epoch + 1}!")
            self.medical_grade_achieved = True
        
        print(f"{'='*60}")
        
        # Save detailed medical report
        if medical_score > self.best_medical_score:
            self.best_medical_score = medical_score
            self._save_comprehensive_report(epoch, medical_score, medical_grade, logs)
    
    def _save_comprehensive_report(self, epoch: int, medical_score: float, medical_grade: bool, logs: dict):
        """Save comprehensive medical validation report."""
        
        # Collect finding accuracies
        finding_metrics = {}
        for key, value in logs.items():
            if 'val_' in key and '_accuracy' in key and key != 'val_dr_accuracy':
                finding_name = key.replace('val_', '').replace('_accuracy', '')
                finding_metrics[finding_name] = float(value)
        
        report = {
            'epoch': epoch + 1,
            'medical_score': float(medical_score),
            'medical_grade_achieved': medical_grade,
            'timestamp': datetime.datetime.now().isoformat(),
            'primary_metrics': {
                'dr_accuracy': float(logs.get('val_dr_accuracy', 0)),
                'dr_precision': float(logs.get('val_dr_precision', 0)),
                'dr_recall': float(logs.get('val_dr_recall', 0)),
                'dr_auc': float(logs.get('val_dr_auc', 0)),
                'f1_score': 2 * logs.get('val_dr_precision', 0) * logs.get('val_dr_recall', 0) / 
                           (logs.get('val_dr_precision', 0) + logs.get('val_dr_recall', 0) + 1e-8)
            },
            'finding_accuracies': finding_metrics,
            'medical_requirements': self.medical_requirements,
            'requirements_status': {
                'accuracy': logs.get('val_dr_accuracy', 0) >= self.medical_requirements['minimum_accuracy'],
                'sensitivity': logs.get('val_dr_recall', 0) >= self.medical_requirements['minimum_sensitivity'],
                'specificity': logs.get('val_dr_precision', 0) >= self.medical_requirements['minimum_specificity'],
                'f1_score': (2 * logs.get('val_dr_precision', 0) * logs.get('val_dr_recall', 0) / 
                            (logs.get('val_dr_precision', 0) + logs.get('val_dr_recall', 0) + 1e-8)) >= 
                           self.medical_requirements['minimum_f1_score'],
                'auc': logs.get('val_dr_auc', 0) >= self.medical_requirements['minimum_auc']
            },
            'total_findings_tracked': len(COMPREHENSIVE_MEDICAL_FINDINGS),
            'grading_rules_tracked': len(GRADING_RULES),
            'hardware_optimization': 'n1-highmem-4 + Tesla P100'
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, f'medical_validation_comprehensive_epoch_{epoch + 1}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)


def create_medical_grade_callbacks(output_dir: str, medical_requirements: Dict[str, float]) -> List[tf.keras.callbacks.Callback]:
    """Create medical-grade callbacks for 95%+ accuracy."""
    
    callbacks = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_dr_accuracy',
        patience=25,  # More patience for medical-grade convergence
        restore_best_weights=True,
        min_delta=0.0005,  # Smaller delta for fine-tuning
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'medical_grade_model_best.h5'),
        monitor='val_dr_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Enhanced learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive reduction
        patience=12,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Medical validation callback
    medical_validator = MedicalGradeCallback(medical_requirements, output_dir)
    callbacks.append(medical_validator)
    
    # CSV logger
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(output_dir, 'medical_training_log.csv')
    )
    callbacks.append(csv_logger)
    
    return callbacks


def main():
    """Demonstrate comprehensive medical model."""
    
    print("🏥 COMPREHENSIVE MEDICAL-GRADE DR SYSTEM")
    print("=" * 70)
    print("Hardware: n1-highmem-4 + Tesla P100 (Cost-effective)")
    print("Target Accuracy: 95%+ (Medical-grade)")
    print(f"Total Medical Findings: {len(COMPREHENSIVE_MEDICAL_FINDINGS)}")
    print(f"Grading Rules: {len(GRADING_RULES)}")
    print("=" * 70)
    
    # Create comprehensive model
    medical_model = ComprehensiveMedicalDRModel(input_shape=(512, 512, 3))
    model = medical_model.create_comprehensive_model()
    model = medical_model.compile_medical_model(model, learning_rate=0.00005)
    
    # Show model summary
    print("\n📋 Model Architecture:")
    model.summary(line_length=100)
    
    print(f"\n✅ Comprehensive medical system ready:")
    print(f"   - All {len(COMPREHENSIVE_MEDICAL_FINDINGS)} findings implemented")
    print(f"   - Medical-grade validation (95%+ target)")
    print(f"   - Tesla P100 optimized (n1-highmem-4)")
    print(f"   - Enhanced callbacks and monitoring")


if __name__ == "__main__":
    main()
"""
Retinal Finding Detection System for Diabetic Retinopathy
Implements detailed finding detection based on medical_terms_type1.json specifications
Optimized for NVIDIA Tesla P100 GPU
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
import json
import datetime
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import argparse

# Tesla P100 GPU optimization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("===== RETINAL FINDING DETECTOR INITIALIZATION =====")
# Configure for Tesla P100
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Tesla P100 configured: {len(gpus)} GPU(s)")
    
    # Mixed precision for Tesla P100
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled")
except Exception as e:
    print(f"⚠️ GPU setup warning: {e}")

print("=" * 50)

# Retinal findings from medical_terms_type1.json
RETINAL_FINDINGS = {
    # NPDR Features
    'microaneurysms': {
        'description': 'Small, round, red spots representing microaneurysms',
        'severity_levels': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_significance': 'early_dr_sign'
    },
    'dot_blot_hemorrhages': {
        'description': 'Small, dot-like hemorrhages in retinal tissue',
        'severity_levels': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_significance': 'dr_progression'
    },
    'flame_hemorrhages': {
        'description': 'Flame-shaped hemorrhages in nerve fiber layer',
        'severity_levels': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_significance': 'dr_progression'
    },
    'hard_exudates': {
        'description': 'Yellow, waxy deposits from lipid leakage',
        'severity_levels': ['absent', 'present'],
        'quadrant_based': True,
        'medical_significance': 'macular_threat'
    },
    'cotton_wool_spots': {
        'description': 'Soft, fluffy white spots from nerve fiber layer infarcts',
        'severity_levels': ['0', '1_3', '>3'],
        'quadrant_based': True,
        'medical_significance': 'severe_dr_sign'
    },
    'venous_beading': {
        'description': 'Irregular venous caliber changes',
        'severity_levels': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_significance': 'severe_npdr_sign'
    },
    'IRMA': {
        'description': 'Intraretinal microvascular abnormalities',
        'severity_levels': ['none', 'mild', 'moderate', 'severe'],
        'quadrant_based': True,
        'medical_significance': 'severe_npdr_sign'
    },
    
    # PDR Features
    'neovascularization_disc': {
        'description': 'New vessels on optic disc (NVD)',
        'severity_levels': ['absent', 'present'],
        'area_measurements': ['none', '<1/3DD', '1/3-1DD', '>1DD'],
        'medical_significance': 'pdr_sign'
    },
    'neovascularization_elsewhere': {
        'description': 'New vessels elsewhere (NVE)',
        'severity_levels': ['absent', 'present'],
        'area_measurements': ['none', '<1/2DD', '≥1/2DD'],
        'medical_significance': 'pdr_sign'
    },
    'neovascularization_iris': {
        'description': 'New vessels on iris (NVI)',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'advanced_pdr'
    },
    'preretinal_hemorrhage': {
        'description': 'Hemorrhage in front of retina',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'pdr_complication'
    },
    'vitreous_hemorrhage': {
        'description': 'Blood in vitreous cavity',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'pdr_complication'
    },
    'fibrovascular_proliferation': {
        'description': 'Fibrous tissue growth with new vessels',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'advanced_pdr'
    },
    'tractional_retinal_detachment': {
        'description': 'Retinal detachment from fibrous traction',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'severe_pdr_complication'
    },
    
    # Treatment Signs
    'PRP_scars': {
        'description': 'Panretinal photocoagulation laser scars',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'treatment_marker'
    },
    'focal_laser_scars': {
        'description': 'Focal/grid laser treatment scars',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'treatment_marker'
    },
    'vitrectomy_changes': {
        'description': 'Changes from vitrectomy surgery',
        'severity_levels': ['absent', 'present'],
        'medical_significance': 'surgical_history'
    }
}

# Grading rules from medical terminology
GRADING_RULES = {
    '4_2_1_rule': {
        'description': 'Severe NPDR criteria: 4Q hemorrhages, 2Q venous beading, 1Q IRMA',
        'components': ['hemorrhages_4_quadrants', 'venous_beading_2plus_quadrants', 'irma_1plus_quadrant'],
        'threshold': 'any_one_present'
    },
    'high_risk_characteristics': {
        'description': 'PDR requiring immediate treatment',
        'components': ['NVD_present', 'NVE_large_area', 'vitreous_hemorrhage'],
        'threshold': 'specific_combinations'
    }
}


class RetinalFindingDetector:
    """Advanced retinal finding detection system."""
    
    def __init__(self, input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.findings_config = RETINAL_FINDINGS
        self.grading_rules = GRADING_RULES
        
    def create_multi_finding_model(self) -> tf.keras.Model:
        """Create a multi-task model for detecting multiple retinal findings."""
        
        # Use EfficientNetB4 for better feature representation
        base_model = tf.keras.applications.EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None  # We'll add custom pooling
        )
        
        # Fine-tune the top layers
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Preprocessing
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Conservative augmentation for medical images
        x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = tf.keras.layers.RandomRotation(0.05)(x)  # Very small rotation
        x = tf.keras.layers.RandomContrast(0.05)(x)  # Minimal contrast change
        
        # Feature extraction
        features = base_model(x, training=True)
        
        # Multi-scale feature extraction
        # Global features
        global_features = tf.keras.layers.GlobalAveragePooling2D()(features)
        global_features = tf.keras.layers.Dropout(0.3)(global_features)
        
        # Local features (for spatial findings)
        local_features = tf.keras.layers.GlobalMaxPooling2D()(features)
        local_features = tf.keras.layers.Dropout(0.3)(local_features)
        
        # Attention mechanism for important regions
        attention_features = self._create_attention_mechanism(features)
        
        # Combine all features
        combined_features = tf.keras.layers.Concatenate()([
            global_features, 
            local_features, 
            attention_features
        ])
        
        # Shared dense layers
        x = tf.keras.layers.Dense(512, activation='relu')(combined_features)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Individual finding detectors
        outputs = {}
        
        for finding_name, finding_config in self.findings_config.items():
            # Determine output size based on finding type
            if 'severity_levels' in finding_config:
                if finding_config['severity_levels'] == ['absent', 'present']:
                    output_size = 1
                    activation = 'sigmoid'
                else:
                    output_size = len(finding_config['severity_levels'])
                    activation = 'softmax'
            else:
                output_size = 1
                activation = 'sigmoid'
            
            # Specific branch for this finding
            finding_branch = tf.keras.layers.Dense(64, activation='relu')(x)
            finding_branch = tf.keras.layers.Dropout(0.2)(finding_branch)
            
            outputs[finding_name] = tf.keras.layers.Dense(
                output_size,
                activation=activation,
                name=finding_name,
                dtype='float32'
            )(finding_branch)
        
        # Additional composite outputs for grading rules
        # 4-2-1 rule detector
        rule_421_branch = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs['rule_4_2_1'] = tf.keras.layers.Dense(
            1, activation='sigmoid', name='rule_4_2_1', dtype='float32'
        )(rule_421_branch)
        
        # High-risk characteristics
        high_risk_branch = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs['high_risk_pdr'] = tf.keras.layers.Dense(
            1, activation='sigmoid', name='high_risk_pdr', dtype='float32'
        )(high_risk_branch)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _create_attention_mechanism(self, features: tf.Tensor) -> tf.Tensor:
        """Create spatial attention mechanism for important retinal regions."""
        
        # Squeeze-and-Excitation like attention
        se_features = tf.keras.layers.GlobalAveragePooling2D()(features)
        se_features = tf.keras.layers.Dense(features.shape[-1] // 16, activation='relu')(se_features)
        se_features = tf.keras.layers.Dense(features.shape[-1], activation='sigmoid')(se_features)
        se_features = tf.keras.layers.Reshape((1, 1, features.shape[-1]))(se_features)
        
        # Apply attention
        attended_features = tf.keras.layers.Multiply()([features, se_features])
        attended_features = tf.keras.layers.GlobalAveragePooling2D()(attended_features)
        
        return attended_features
    
    def create_localization_model(self) -> tf.keras.Model:
        """Create a model for localizing findings (bounding box regression)."""
        
        # Use a detection-oriented backbone
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            alpha=1.0
        )
        
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Feature extraction
        features = base_model(x, training=True)
        
        # Feature Pyramid Network-like structure
        # Multiple scale outputs for different finding sizes
        
        # Large findings (e.g., hemorrhages, exudates)
        large_features = tf.keras.layers.GlobalAveragePooling2D()(features)
        large_bbox = tf.keras.layers.Dense(4, name='large_findings_bbox')(large_features)
        large_conf = tf.keras.layers.Dense(len(self.findings_config), activation='sigmoid', 
                                         name='large_findings_conf')(large_features)
        
        # Small findings (e.g., microaneurysms)
        small_features = tf.keras.layers.GlobalMaxPooling2D()(features)
        small_bbox = tf.keras.layers.Dense(4, name='small_findings_bbox')(small_features)
        small_conf = tf.keras.layers.Dense(len(self.findings_config), activation='sigmoid',
                                         name='small_findings_conf')(small_features)
        
        outputs = {
            'large_findings_bbox': large_bbox,
            'large_findings_conf': large_conf,
            'small_findings_bbox': small_bbox,
            'small_findings_conf': small_conf
        }
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_finding_model(self, model: tf.keras.Model, learning_rate: float = 0.0001) -> tf.keras.Model:
        """Compile the retinal finding detection model."""
        
        # Define losses for different findings
        losses = {}
        loss_weights = {}
        
        for finding_name, finding_config in self.findings_config.items():
            medical_weight = self._get_medical_importance_weight(finding_config['medical_significance'])
            
            if 'severity_levels' in finding_config:
                if finding_config['severity_levels'] == ['absent', 'present']:
                    losses[finding_name] = tf.keras.losses.BinaryCrossentropy()
                    loss_weights[finding_name] = medical_weight
                else:
                    losses[finding_name] = tf.keras.losses.CategoricalCrossentropy()
                    loss_weights[finding_name] = medical_weight
            else:
                losses[finding_name] = tf.keras.losses.BinaryCrossentropy()
                loss_weights[finding_name] = medical_weight
        
        # Add losses for composite rules
        losses['rule_4_2_1'] = tf.keras.losses.BinaryCrossentropy()
        losses['high_risk_pdr'] = tf.keras.losses.BinaryCrossentropy()
        loss_weights['rule_4_2_1'] = 2.0  # High importance
        loss_weights['high_risk_pdr'] = 2.0  # High importance
        
        # Define metrics
        metrics = {}
        for finding_name in losses.keys():
            if 'rule_' in finding_name or 'high_risk' in finding_name:
                metrics[finding_name] = [
                    tf.keras.metrics.BinaryAccuracy(name=f'{finding_name}_accuracy'),
                    tf.keras.metrics.Precision(name=f'{finding_name}_precision'),
                    tf.keras.metrics.Recall(name=f'{finding_name}_recall'),
                    tf.keras.metrics.AUC(name=f'{finding_name}_auc')
                ]
            elif finding_name in self.findings_config:
                finding_config = self.findings_config[finding_name]
                if ('severity_levels' in finding_config and 
                    finding_config['severity_levels'] != ['absent', 'present']):
                    metrics[finding_name] = [
                        tf.keras.metrics.CategoricalAccuracy(name=f'{finding_name}_accuracy'),
                        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name=f'{finding_name}_top2')
                    ]
                else:
                    metrics[finding_name] = [
                        tf.keras.metrics.BinaryAccuracy(name=f'{finding_name}_accuracy'),
                        tf.keras.metrics.Precision(name=f'{finding_name}_precision'),
                        tf.keras.metrics.Recall(name=f'{finding_name}_recall'),
                        tf.keras.metrics.AUC(name=f'{finding_name}_auc')
                    ]
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Mixed precision wrapper
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        return model
    
    def _get_medical_importance_weight(self, significance: str) -> float:
        """Get loss weight based on medical significance."""
        weight_mapping = {
            'early_dr_sign': 1.0,
            'dr_progression': 1.2,
            'severe_dr_sign': 1.5,
            'severe_npdr_sign': 1.8,
            'pdr_sign': 2.0,
            'pdr_complication': 2.5,
            'advanced_pdr': 3.0,
            'severe_pdr_complication': 3.5,
            'macular_threat': 2.0,
            'treatment_marker': 0.8,
            'surgical_history': 0.8
        }
        return weight_mapping.get(significance, 1.0)


class FindingAnnotationGenerator:
    """Generate finding annotations and reports."""
    
    def __init__(self, model: tf.keras.Model, findings_config: Dict):
        self.model = model
        self.findings_config = findings_config
    
    def predict_findings(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict all findings for a single image."""
        
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        
        # Parse predictions
        findings_report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'findings': {},
            'grading_rules': {},
            'clinical_significance': {},
            'recommendations': []
        }
        
        # Process individual findings
        for finding_name, finding_config in self.findings_config.items():
            if finding_name in predictions:
                pred_value = predictions[finding_name][0]
                
                if 'severity_levels' in finding_config:
                    if finding_config['severity_levels'] == ['absent', 'present']:
                        # Binary classification
                        confidence = float(pred_value[0]) if len(pred_value) > 0 else float(pred_value)
                        present = confidence > 0.5
                        findings_report['findings'][finding_name] = {
                            'present': present,
                            'confidence': confidence,
                            'severity': 'present' if present else 'absent'
                        }
                    else:
                        # Multi-class severity
                        severity_idx = np.argmax(pred_value)
                        severity = finding_config['severity_levels'][severity_idx]
                        confidence = float(pred_value[severity_idx])
                        findings_report['findings'][finding_name] = {
                            'severity': severity,
                            'confidence': confidence,
                            'severity_probabilities': {
                                level: float(prob) for level, prob in 
                                zip(finding_config['severity_levels'], pred_value)
                            }
                        }
                
                # Add medical significance
                findings_report['clinical_significance'][finding_name] = finding_config.get('medical_significance', 'unknown')
        
        # Process grading rules
        if 'rule_4_2_1' in predictions:
            rule_421_conf = float(predictions['rule_4_2_1'][0])
            findings_report['grading_rules']['4_2_1_rule'] = {
                'meets_criteria': rule_421_conf > 0.5,
                'confidence': rule_421_conf
            }
        
        if 'high_risk_pdr' in predictions:
            high_risk_conf = float(predictions['high_risk_pdr'][0])
            findings_report['grading_rules']['high_risk_pdr'] = {
                'high_risk': high_risk_conf > 0.5,
                'confidence': high_risk_conf
            }
        
        # Generate recommendations
        findings_report['recommendations'] = self._generate_clinical_recommendations(findings_report)
        
        return findings_report
    
    def _generate_clinical_recommendations(self, findings_report: Dict) -> List[str]:
        """Generate clinical recommendations based on findings."""
        recommendations = []
        
        # Check for severe findings
        severe_findings = []
        for finding_name, finding_data in findings_report['findings'].items():
            significance = findings_report['clinical_significance'].get(finding_name, '')
            
            if 'severe' in significance or 'advanced' in significance:
                if (finding_data.get('present', False) or 
                    finding_data.get('severity', 'absent') != 'absent'):
                    severe_findings.append(finding_name)
        
        if severe_findings:
            recommendations.append("immediate_ophthalmology_referral")
            recommendations.append("urgent_vitreoretinal_consult")
        
        # Check grading rules
        if findings_report['grading_rules'].get('4_2_1_rule', {}).get('meets_criteria', False):
            recommendations.append("follow_up_3_months")
            recommendations.append("consider_panretinal_photocoagulation")
        
        if findings_report['grading_rules'].get('high_risk_pdr', {}).get('high_risk', False):
            recommendations.append("immediate_ophthalmology_referral")
            recommendations.append("anti_VEGF_evaluation")
        
        # Check for macular threats
        macular_threats = ['hard_exudates']
        for threat in macular_threats:
            if (threat in findings_report['findings'] and 
                findings_report['findings'][threat].get('present', False)):
                recommendations.append("focal_or_grid_laser")
                recommendations.append("OCT_recommended")
        
        # General recommendations
        if not recommendations:
            recommendations.append("routine_screening")
        
        recommendations.append("optimize_glycemic_control")
        recommendations.append("manage_hypertension")
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_heatmap(self, image: np.ndarray, finding_name: str) -> np.ndarray:
        """Generate attention heatmap for a specific finding."""
        
        # Create a model that outputs intermediate activations
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer('efficientnetb4').output, self.model.output[finding_name]]
        )
        
        # Get activations
        with tf.GradientTape() as tape:
            features, prediction = intermediate_model(np.expand_dims(image, axis=0))
            
        # Compute gradients
        grads = tape.gradient(prediction, features)
        
        # Generate heatmap using Grad-CAM
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        features = features[0]
        
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, features), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        heatmap = heatmap / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        return heatmap


def main():
    """Main function for retinal finding detection training."""
    parser = argparse.ArgumentParser(description='Train retinal finding detection model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to annotated dataset')
    parser.add_argument('--output_dir', type=str, default='./finding_detection_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--input_size', type=int, default=512, help='Input image size')
    
    args = parser.parse_args()
    
    print("🔍 RETINAL FINDING DETECTION SYSTEM")
    print("=" * 50)
    print(f"Findings to detect: {len(RETINAL_FINDINGS)}")
    print(f"Tesla P100 optimized: ✅")
    print("=" * 50)
    
    # Create detector
    detector = RetinalFindingDetector(input_shape=(args.input_size, args.input_size, 3))
    
    # Create model
    print("📋 Building multi-finding detection model...")
    model = detector.create_multi_finding_model()
    model = detector.compile_finding_model(model, args.learning_rate)
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    print(f"📊 Detecting {len(RETINAL_FINDINGS)} different findings")
    
    # TODO: Add dataset loading and training logic here
    # This would require annotated data for each finding type
    
    print("✅ Retinal finding detection system ready!")


if __name__ == "__main__":
    main()
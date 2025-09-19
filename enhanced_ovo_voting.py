"""
Enhanced OVO Voting Mechanism - Replacement for ensemble_local_trainer.py

This contains the fixed voting mechanism that needs to be integrated
into the training pipeline to achieve medical-grade performance.
"""

import torch
import torch.nn.functional as F
import numpy as np

class EnhancedOVOEnsemble:
    """Enhanced OVO Ensemble with medical-grade voting fixes"""

    def __init__(self, base_models, num_classes=5):
        self.base_models = base_models
        self.num_classes = num_classes

        # REAL accuracies - will be populated during training or loaded from diagnostics
        self.binary_accuracies = {
            'mobilenet_v2': {
                'pair_0_1': 0.8745, 'pair_0_2': 0.8228, 'pair_0_3': 0.9827, 'pair_0_4': 0.9845,
                'pair_1_2': 0.8072, 'pair_1_3': 0.8667, 'pair_1_4': 0.9251, 'pair_2_3': 0.8567,
                'pair_2_4': 0.8836, 'pair_3_4': 0.7205
            },
            'inception_v3': {
                'pair_0_1': 0.8340, 'pair_0_2': 0.7850, 'pair_0_3': 0.9286, 'pair_0_4': 0.9732,
                'pair_1_2': 0.7972, 'pair_1_3': 0.8267, 'pair_1_4': 0.8719, 'pair_2_3': 0.8206,
                'pair_2_4': 0.8501, 'pair_3_4': 0.7654
            },
            'densenet121': {
                'pair_0_1': 0.8870, 'pair_0_2': 0.8791, 'pair_0_3': 0.9827, 'pair_0_4': 0.9881,
                'pair_1_2': 0.8483, 'pair_1_3': 0.8767, 'pair_1_4': 0.8968, 'pair_2_3': 0.8927,
                'pair_2_4': 0.8819, 'pair_3_4': 0.7937
            }
        }

        # Weak classifiers (below 80% accuracy)
        self.weak_classifiers = {
            ('inception_v3', 'pair_0_2'), ('inception_v3', 'pair_1_2'),
            ('inception_v3', 'pair_3_4'), ('mobilenet_v2', 'pair_3_4')
        }

        # Medical-grade class weights (massive Class 1 boost)
        self.class_weights = torch.tensor([1.0, 8.0, 2.0, 4.0, 5.0])
        self.class1_pairs = ['pair_0_1', 'pair_1_2', 'pair_1_3', 'pair_1_4']

    def enhanced_forward(self, x, classifiers, return_individual=False):
        """FIXED forward pass with medical-grade voting"""
        batch_size = x.size(0)
        device = x.device

        # Initialize enhanced vote accumulation
        class_scores = torch.zeros(batch_size, self.num_classes, device=device)
        total_weights = torch.zeros(batch_size, self.num_classes, device=device)

        individual_predictions = {} if return_individual else None
        class_weights = self.class_weights.to(device)

        for model_name, model_classifiers in classifiers.items():
            if return_individual:
                individual_predictions[model_name] = torch.zeros(batch_size, self.num_classes, device=device)

            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction
                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # FIXED: Real accuracy-based weighting
                base_accuracy = self.binary_accuracies[model_name][classifier_name]

                # Handle weak classifiers with penalty
                if (model_name, classifier_name) in self.weak_classifiers:
                    accuracy_weight = (base_accuracy ** 4) * 0.5  # Heavy penalty
                else:
                    accuracy_weight = base_accuracy ** 2

                # Boost excellent classifiers
                if base_accuracy > 0.95:
                    accuracy_weight *= 1.5

                # FIXED: True confidence weighting
                confidence = torch.abs(binary_output - 0.5) * 2
                weighted_confidence = confidence * accuracy_weight

                # FIXED: Medical-grade class weighting
                class_a_weight = class_weights[class_a]
                class_b_weight = class_weights[class_b]

                # EMERGENCY Class 1 boost
                if classifier_name in self.class1_pairs:
                    if class_a == 1:
                        class_a_weight *= 3.0  # 3x emergency boost
                    if class_b == 1:
                        class_b_weight *= 3.0  # 3x emergency boost

                # FIXED: Probability-based voting (not binary)
                prob_class_a = (1.0 - binary_output) * class_a_weight * weighted_confidence
                prob_class_b = binary_output * class_b_weight * weighted_confidence

                # Accumulate weighted votes
                class_scores[:, class_a] += prob_class_a
                class_scores[:, class_b] += prob_class_b

                total_weights[:, class_a] += class_a_weight * weighted_confidence
                total_weights[:, class_b] += class_b_weight * weighted_confidence

                if return_individual:
                    individual_predictions[model_name][:, class_a] += prob_class_a
                    individual_predictions[model_name][:, class_b] += prob_class_b

        # FIXED: Proper normalization
        normalized_scores = class_scores / (total_weights + 1e-8)
        final_predictions = F.softmax(normalized_scores, dim=1)

        result = {
            'logits': final_predictions,
            'raw_scores': normalized_scores,
            'votes': class_scores
        }

        if return_individual:
            for model_name in individual_predictions:
                model_weights = total_weights / len(self.base_models)
                individual_predictions[model_name] = individual_predictions[model_name] / (model_weights + 1e-8)
            result['individual_predictions'] = individual_predictions

        return result

# Replacement code for ensemble_local_trainer.py lines 447-495
FIXED_FORWARD_METHOD = '''
def forward(self, x, return_individual=False):
    """FIXED forward pass with medical-grade voting mechanism"""
    batch_size = x.size(0)
    device = x.device

    # Enhanced vote accumulation
    class_scores = torch.zeros(batch_size, self.num_classes, device=device)
    total_weights = torch.zeros(batch_size, self.num_classes, device=device)

    individual_predictions = {} if return_individual else None

    # Medical-grade class weights (Class 1 emergency boost)
    class_weights = torch.tensor([1.0, 8.0, 2.0, 4.0, 5.0], device=device)
    class1_pairs = ['pair_0_1', 'pair_1_2', 'pair_1_3', 'pair_1_4']

    # Binary accuracy weights (load from diagnostic analysis)
    binary_accuracies = {
        'mobilenet_v2': {
            'pair_0_1': 0.8745, 'pair_0_2': 0.8228, 'pair_0_3': 0.9827, 'pair_0_4': 0.9845,
            'pair_1_2': 0.8072, 'pair_1_3': 0.8667, 'pair_1_4': 0.9251, 'pair_2_3': 0.8567,
            'pair_2_4': 0.8836, 'pair_3_4': 0.7205
        },
        'inception_v3': {
            'pair_0_1': 0.8340, 'pair_0_2': 0.7850, 'pair_0_3': 0.9286, 'pair_0_4': 0.9732,
            'pair_1_2': 0.7972, 'pair_1_3': 0.8267, 'pair_1_4': 0.8719, 'pair_2_3': 0.8206,
            'pair_2_4': 0.8501, 'pair_3_4': 0.7654
        },
        'densenet121': {
            'pair_0_1': 0.8870, 'pair_0_2': 0.8791, 'pair_0_3': 0.9827, 'pair_0_4': 0.9881,
            'pair_1_2': 0.8483, 'pair_1_3': 0.8767, 'pair_1_4': 0.8968, 'pair_2_3': 0.8927,
            'pair_2_4': 0.8819, 'pair_3_4': 0.7937
        }
    }

    weak_classifiers = {
        ('inception_v3', 'pair_0_2'), ('inception_v3', 'pair_1_2'),
        ('inception_v3', 'pair_3_4'), ('mobilenet_v2', 'pair_3_4')
    }

    for model_name, model_classifiers in self.classifiers.items():
        if return_individual:
            individual_predictions[model_name] = torch.zeros(batch_size, self.num_classes, device=device)

        for classifier_name, classifier in model_classifiers.items():
            class_a, class_b = map(int, classifier_name.split('_')[1:])

            # Get binary prediction
            binary_output = classifier(x).squeeze()
            if binary_output.dim() == 0:
                binary_output = binary_output.unsqueeze(0)

            # FIXED: Accuracy-based weighting
            base_accuracy = binary_accuracies[model_name][classifier_name]

            if (model_name, classifier_name) in weak_classifiers:
                accuracy_weight = (base_accuracy ** 4) * 0.5  # Penalty
            else:
                accuracy_weight = base_accuracy ** 2

            if base_accuracy > 0.95:
                accuracy_weight *= 1.5  # Boost excellent classifiers

            # FIXED: Confidence-based weighting
            confidence = torch.abs(binary_output - 0.5) * 2
            weighted_confidence = confidence * accuracy_weight

            # FIXED: Medical-grade class weighting
            class_a_weight = class_weights[class_a]
            class_b_weight = class_weights[class_b]

            # Emergency Class 1 boost
            if classifier_name in class1_pairs:
                if class_a == 1:
                    class_a_weight *= 3.0
                if class_b == 1:
                    class_b_weight *= 3.0

            # FIXED: Probability-based voting
            prob_class_a = (1.0 - binary_output) * class_a_weight * weighted_confidence
            prob_class_b = binary_output * class_b_weight * weighted_confidence

            class_scores[:, class_a] += prob_class_a
            class_scores[:, class_b] += prob_class_b

            total_weights[:, class_a] += class_a_weight * weighted_confidence
            total_weights[:, class_b] += class_b_weight * weighted_confidence

            if return_individual:
                individual_predictions[model_name][:, class_a] += prob_class_a
                individual_predictions[model_name][:, class_b] += prob_class_b

    # FIXED: Proper normalization and softmax
    normalized_scores = class_scores / (total_weights + 1e-8)
    final_predictions = F.softmax(normalized_scores, dim=1)

    result = {
        'logits': final_predictions,
        'votes': class_scores
    }

    if return_individual:
        for model_name in individual_predictions:
            model_weights = total_weights / len(self.base_models)
            individual_predictions[model_name] = individual_predictions[model_name] / (model_weights + 1e-8)
        result['individual_predictions'] = individual_predictions

    return result
'''
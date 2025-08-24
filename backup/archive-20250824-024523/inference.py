import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple
import json
import argparse

from models import DiabeticRetinopathyModel
from config import get_config
from utils import set_seed

class DRInferenceEngine:
    """Inference engine for diabetic retinopathy classification with medical reasoning."""
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = get_config()  # Get default config
            # Update with loaded config (simplified)
        else:
            self.config = get_config()
        
        # Initialize model
        self.model = DiabeticRetinopathyModel(self.config)
        self.model.to(self.device)
        
        # Load trained weights
        self.load_model(model_path)
        
        # Setup transforms
        self.transform = self._get_inference_transforms()
        
        # Class names
        self.rg_class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR']
        self.me_class_names = ['No Risk', 'Low Risk', 'High Risk']
        
        # Risk level mappings
        self.risk_levels = {
            'rg': {0: 'Low', 1: 'Low', 2: 'Moderate', 3: 'High'},
            'me': {0: 'Low', 1: 'Moderate', 2: 'High'}
        }
        
    def load_model(self, model_path: str):
        """Load trained model weights."""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (for DataParallel models)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                cleaned_state_dict[k] = v
            
            self.model.load_state_dict(cleaned_state_dict, strict=False)
            self.model.eval()
            
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def _get_inference_transforms(self) -> A.Compose:
        """Get transforms for inference."""
        
        return A.Compose([
            A.Resize(self.config.model.img_size, self.config.model.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference."""
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # Assume it's already a numpy array
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict_single(self, image_path: str) -> Dict:
        """Make prediction for a single image."""
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            # Get probabilities
            rg_probs = F.softmax(outputs['rg_logits'], dim=1)
            me_probs = F.softmax(outputs['me_logits'], dim=1)
            
            # Get predictions
            rg_pred = torch.argmax(rg_probs, dim=1)
            me_pred = torch.argmax(me_probs, dim=1)
            
            # Get confidences
            rg_confidence = torch.max(rg_probs, dim=1)[0]
            me_confidence = torch.max(me_probs, dim=1)[0]
            
            # Generate medical report
            medical_reports = self.model.generate_medical_report(
                image_tensor, rg_pred, me_pred
            )
        
        # Prepare results
        result = {
            'image_path': image_path,
            'retinopathy_grade': {
                'grade': int(rg_pred.item()),
                'name': self.rg_class_names[rg_pred.item()],
                'confidence': float(rg_confidence.item()),
                'risk_level': self.risk_levels['rg'][rg_pred.item()],
                'probabilities': {
                    self.rg_class_names[i]: float(rg_probs[0, i].item()) 
                    for i in range(len(self.rg_class_names))
                }
            },
            'macular_edema': {
                'grade': int(me_pred.item()),
                'name': self.me_class_names[me_pred.item()],
                'confidence': float(me_confidence.item()),
                'risk_level': self.risk_levels['me'][me_pred.item()],
                'probabilities': {
                    self.me_class_names[i]: float(me_probs[0, i].item()) 
                    for i in range(len(self.me_class_names))
                }
            },
            'medical_report': medical_reports[0] if medical_reports else "Unable to generate medical report",
            'overall_risk_assessment': self._assess_overall_risk(rg_pred.item(), me_pred.item()),
            'recommendations': self._generate_recommendations(rg_pred.item(), me_pred.item())
        }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Make predictions for multiple images."""
        
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                error_result = {
                    'image_path': image_path,
                    'error': str(e),
                    'retinopathy_grade': None,
                    'macular_edema': None,
                    'medical_report': f"Error processing image: {e}"
                }
                results.append(error_result)
        
        return results
    
    def _assess_overall_risk(self, rg_grade: int, me_grade: int) -> Dict:
        """Assess overall risk based on both RG and ME grades."""
        
        # Risk matrix
        risk_matrix = {
            (0, 0): 'Low', (0, 1): 'Low', (0, 2): 'Moderate',
            (1, 0): 'Low', (1, 1): 'Moderate', (1, 2): 'High',
            (2, 0): 'Moderate', (2, 1): 'High', (2, 2): 'High',
            (3, 0): 'High', (3, 1): 'High', (3, 2): 'High'
        }
        
        overall_risk = risk_matrix.get((rg_grade, me_grade), 'High')
        
        # Determine urgency
        if rg_grade >= 3 or me_grade >= 2:
            urgency = 'Urgent - Immediate ophthalmological consultation required'
        elif rg_grade >= 2 or me_grade >= 1:
            urgency = 'Moderate - Ophthalmological follow-up within 3-6 months'
        else:
            urgency = 'Low - Routine annual screening'
        
        return {
            'risk_level': overall_risk,
            'urgency': urgency,
            'composite_score': (rg_grade * 0.6 + me_grade * 0.4)  # Weighted score
        }
    
    def _generate_recommendations(self, rg_grade: int, me_grade: int) -> List[str]:
        """Generate clinical recommendations based on grades."""
        
        recommendations = []
        
        # RG-based recommendations
        if rg_grade == 0:
            recommendations.append("Continue annual diabetic eye screening")
        elif rg_grade == 1:
            recommendations.append("Increase screening frequency to 6-12 months")
            recommendations.append("Optimize diabetes control")
        elif rg_grade == 2:
            recommendations.append("Refer to ophthalmologist within 3-6 months")
            recommendations.append("Consider laser photocoagulation if progressing")
            recommendations.append("Intensive diabetes management")
        elif rg_grade == 3:
            recommendations.append("Urgent ophthalmological referral")
            recommendations.append("Consider anti-VEGF therapy")
            recommendations.append("Monitor for proliferative changes")
        
        # ME-based recommendations
        if me_grade >= 1:
            recommendations.append("Monitor for visual changes")
            recommendations.append("Consider OCT imaging for detailed macular assessment")
        if me_grade == 2:
            recommendations.append("Evaluate for anti-VEGF or steroid treatment")
            recommendations.append("Assess cardiovascular risk factors")
        
        # General recommendations
        recommendations.extend([
            "Maintain HbA1c < 7% (53 mmol/mol)",
            "Control blood pressure < 140/90 mmHg",
            "Regular cardiovascular risk assessment"
        ])
        
        return recommendations
    
    def export_result(self, result: Dict, output_path: str, format: str = 'json'):
        """Export prediction result to file."""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        
        elif format.lower() == 'txt':
            with open(output_path, 'w') as f:
                f.write(self._format_text_report(result))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_text_report(self, result: Dict) -> str:
        """Format result as text report."""
        
        if 'error' in result:
            return f"Error processing image: {result['error']}"
        
        rg = result['retinopathy_grade']
        me = result['macular_edema']
        overall = result['overall_risk_assessment']
        
        report = f"""
DIABETIC RETINOPATHY ANALYSIS REPORT
=====================================

Image: {os.path.basename(result['image_path'])}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

FINDINGS:
---------
Retinopathy Grade: {rg['grade']} - {rg['name']}
Confidence: {rg['confidence']:.3f}
Risk Level: {rg['risk_level']}

Macular Edema Risk: {me['grade']} - {me['name']}
Confidence: {me['confidence']:.3f}
Risk Level: {me['risk_level']}

OVERALL ASSESSMENT:
------------------
Risk Level: {overall['risk_level']}
Urgency: {overall['urgency']}
Composite Score: {overall['composite_score']:.2f}

DETAILED PROBABILITIES:
----------------------
Retinopathy Grades:
{chr(10).join([f"  {name}: {prob:.3f}" for name, prob in rg['probabilities'].items()])}

Macular Edema Risks:
{chr(10).join([f"  {name}: {prob:.3f}" for name, prob in me['probabilities'].items()])}

MEDICAL REASONING:
-----------------
{result['medical_report']}

RECOMMENDATIONS:
---------------
{chr(10).join([f"• {rec}" for rec in result['recommendations']])}

DISCLAIMER:
----------
This AI-generated analysis is for screening purposes only and should not replace
professional medical diagnosis. Please consult an ophthalmologist for definitive
diagnosis and treatment planning.
        """.strip()
        
        return report

def main():
    """Main function for command-line inference."""
    
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--image_path', help='Path to single image')
    parser.add_argument('--image_dir', help='Directory containing images')
    parser.add_argument('--output_dir', default='inference_results', help='Output directory')
    parser.add_argument('--format', choices=['json', 'txt'], default='json', help='Output format')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = DRInferenceEngine(
        model_path=args.model_path,
        device=args.device
    )
    
    # Prepare images
    if args.image_path:
        image_paths = [args.image_path]
    elif args.image_dir:
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext.upper()}")))
    else:
        raise ValueError("Either --image_path or --image_dir must be provided")
    
    # Make predictions
    print(f"Processing {len(image_paths)} image(s)...")
    results = engine.predict_batch(image_paths)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        if args.image_path:
            output_filename = f"result.{args.format}"
        else:
            image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
            output_filename = f"{image_name}_result.{args.format}"
        
        output_path = os.path.join(args.output_dir, output_filename)
        engine.export_result(result, output_path, args.format)
        
        # Print summary
        if 'error' not in result:
            print(f"\n{result['image_path']}:")
            print(f"  RG: {result['retinopathy_grade']['name']} (confidence: {result['retinopathy_grade']['confidence']:.3f})")
            print(f"  ME: {result['macular_edema']['name']} (confidence: {result['macular_edema']['confidence']:.3f})")
            print(f"  Overall Risk: {result['overall_risk_assessment']['risk_level']}")
        else:
            print(f"\nError processing {result['image_path']}: {result['error']}")
    
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    import glob
    import pandas as pd
    main()
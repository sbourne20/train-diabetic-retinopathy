#!/usr/bin/env python3
"""
Deep Dataset Analysis for Medical-Grade Diabetic Retinopathy Training
Comprehensive assessment of dataset quality, sufficiency, and medical compliance
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import argparse
from datetime import datetime

class MedicalDatasetAnalyzer:
    """Comprehensive medical dataset analysis"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.analysis_results = {}
        self.medical_standards = {
            'min_samples_per_class': {
                'research': 100,
                'clinical_validation': 500,
                'medical_production': 1000
            },
            'class_balance_ratio': {
                'excellent': 3.0,
                'good': 5.0,
                'acceptable': 10.0,
                'poor': float('inf')
            },
            'image_quality': {
                'min_resolution': (224, 224),
                'preferred_resolution': (512, 512),
                'medical_resolution': (1024, 1024)
            },
            'total_samples': {
                'minimum': 5000,
                'good': 10000,
                'excellent': 20000
            }
        }

    def analyze_class_distribution(self):
        """Analyze class distribution across splits"""
        print("\n" + "="*70)
        print("📊 CLASS DISTRIBUTION ANALYSIS")
        print("="*70)

        distribution_data = {}

        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                print(f"⚠️  Warning: {split} directory not found")
                continue

            split_data = {}
            split_total = 0

            for class_id in range(5):  # DR classes 0-4
                class_path = split_path / str(class_id)
                if class_path.exists():
                    count = len([f for f in class_path.iterdir()
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    split_data[class_id] = count
                    split_total += count
                else:
                    split_data[class_id] = 0

            distribution_data[split] = split_data
            distribution_data[f'{split}_total'] = split_total

            print(f"\n📁 {split.upper()} SET ({split_total:,} images):")
            for class_id in range(5):
                count = split_data[class_id]
                percentage = (count / split_total * 100) if split_total > 0 else 0
                print(f"   Class {class_id}: {count:5,} images ({percentage:5.1f}%)")

        # Overall statistics
        total_images = sum(distribution_data[f'{split}_total'] for split in ['train', 'val', 'test']
                          if f'{split}_total' in distribution_data)

        print(f"\n📈 OVERALL DATASET:")
        print(f"   Total Images: {total_images:,}")

        overall_class_counts = defaultdict(int)
        for split in ['train', 'val', 'test']:
            if split in distribution_data:
                for class_id, count in distribution_data[split].items():
                    overall_class_counts[class_id] += count

        for class_id in range(5):
            count = overall_class_counts[class_id]
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"   Class {class_id}: {count:5,} images ({percentage:5.1f}%)")

        self.analysis_results['distribution'] = distribution_data
        self.analysis_results['overall_counts'] = dict(overall_class_counts)
        self.analysis_results['total_images'] = total_images

        return distribution_data, overall_class_counts, total_images

    def assess_class_balance(self, class_counts):
        """Assess class balance quality"""
        print("\n" + "="*70)
        print("⚖️  CLASS BALANCE ASSESSMENT")
        print("="*70)

        if not class_counts:
            print("❌ No class data available for assessment")
            return

        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)

        if min_count == 0:
            print("❌ CRITICAL: One or more classes have 0 samples!")
            return

        balance_ratio = max_count / min_count

        print(f"📊 Class Balance Statistics:")
        print(f"   Minimum class size: {min_count:,} samples")
        print(f"   Maximum class size: {max_count:,} samples")
        print(f"   Balance ratio: {balance_ratio:.1f}:1")

        # Assess balance quality
        if balance_ratio <= self.medical_standards['class_balance_ratio']['excellent']:
            balance_grade = "✅ EXCELLENT"
            balance_status = "Well-balanced dataset"
        elif balance_ratio <= self.medical_standards['class_balance_ratio']['good']:
            balance_grade = "🟡 GOOD"
            balance_status = "Acceptable balance, minor adjustments recommended"
        elif balance_ratio <= self.medical_standards['class_balance_ratio']['acceptable']:
            balance_grade = "⚠️ ACCEPTABLE"
            balance_status = "Significant imbalance, class weights strongly recommended"
        else:
            balance_grade = "❌ POOR"
            balance_status = "Severe imbalance, SMOTE and class weights required"

        print(f"\n🏆 Balance Grade: {balance_grade}")
        print(f"📝 Status: {balance_status}")

        # Medical sufficiency assessment per class
        print(f"\n🏥 MEDICAL SUFFICIENCY PER CLASS:")
        standards = self.medical_standards['min_samples_per_class']

        for class_id, count in class_counts.items():
            if count >= standards['medical_production']:
                grade = "✅ MEDICAL PRODUCTION READY"
            elif count >= standards['clinical_validation']:
                grade = "🟡 CLINICAL VALIDATION READY"
            elif count >= standards['research']:
                grade = "📈 RESEARCH LEVEL"
            else:
                grade = "❌ INSUFFICIENT"

            print(f"   Class {class_id}: {count:,} samples - {grade}")

        self.analysis_results['balance'] = {
            'ratio': balance_ratio,
            'grade': balance_grade,
            'status': balance_status,
            'min_count': min_count,
            'max_count': max_count
        }

        return balance_ratio, balance_grade

    def analyze_image_quality(self, sample_size=100):
        """Analyze image quality metrics"""
        print("\n" + "="*70)
        print("🔍 IMAGE QUALITY ANALYSIS")
        print("="*70)

        quality_metrics = {
            'resolutions': [],
            'file_sizes': [],
            'formats': defaultdict(int),
            'aspect_ratios': [],
            'mean_brightness': [],
            'contrast_scores': [],
            'blur_scores': []
        }

        all_images = []
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue

            for class_id in range(5):
                class_path = split_path / str(class_id)
                if class_path.exists():
                    for img_path in class_path.iterdir():
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            all_images.append(img_path)

        if not all_images:
            print("❌ No images found for analysis")
            return

        # Sample images for detailed analysis
        sample_images = np.random.choice(all_images,
                                       min(sample_size, len(all_images)),
                                       replace=False)

        print(f"📊 Analyzing {len(sample_images)} sample images...")

        for img_path in sample_images:
            try:
                # Basic file info
                file_size = img_path.stat().st_size
                quality_metrics['file_sizes'].append(file_size)
                quality_metrics['formats'][img_path.suffix.lower()] += 1

                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                quality_metrics['resolutions'].append((w, h))
                quality_metrics['aspect_ratios'].append(w / h)

                # Quality metrics
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Brightness
                brightness = np.mean(gray)
                quality_metrics['mean_brightness'].append(brightness)

                # Contrast (RMS contrast)
                contrast = np.std(gray)
                quality_metrics['contrast_scores'].append(contrast)

                # Blur detection (Laplacian variance)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality_metrics['blur_scores'].append(blur_score)

            except Exception as e:
                print(f"   ⚠️ Error analyzing {img_path.name}: {e}")
                continue

        # Analyze results
        if quality_metrics['resolutions']:
            resolutions = quality_metrics['resolutions']
            widths, heights = zip(*resolutions)

            print(f"\n📐 RESOLUTION ANALYSIS:")
            print(f"   Images analyzed: {len(resolutions):,}")
            print(f"   Width range: {min(widths)}-{max(widths)} pixels")
            print(f"   Height range: {min(heights)}-{max(heights)} pixels")
            print(f"   Most common resolution: {max(set(resolutions), key=resolutions.count)}")

            # Resolution sufficiency
            min_res = self.medical_standards['image_quality']['min_resolution']
            pref_res = self.medical_standards['image_quality']['preferred_resolution']
            med_res = self.medical_standards['image_quality']['medical_resolution']

            sufficient_count = sum(1 for w, h in resolutions if w >= min_res[0] and h >= min_res[1])
            preferred_count = sum(1 for w, h in resolutions if w >= pref_res[0] and h >= pref_res[1])
            medical_count = sum(1 for w, h in resolutions if w >= med_res[0] and h >= med_res[1])

            print(f"\n📊 RESOLUTION SUFFICIENCY:")
            print(f"   Minimum ({min_res[0]}x{min_res[1]}+): {sufficient_count}/{len(resolutions)} ({sufficient_count/len(resolutions)*100:.1f}%)")
            print(f"   Preferred ({pref_res[0]}x{pref_res[1]}+): {preferred_count}/{len(resolutions)} ({preferred_count/len(resolutions)*100:.1f}%)")
            print(f"   Medical ({med_res[0]}x{med_res[1]}+): {medical_count}/{len(resolutions)} ({medical_count/len(resolutions)*100:.1f}%)")

        if quality_metrics['file_sizes']:
            sizes = quality_metrics['file_sizes']
            print(f"\n💾 FILE SIZE ANALYSIS:")
            print(f"   Average size: {np.mean(sizes)/1024/1024:.1f} MB")
            print(f"   Size range: {min(sizes)/1024:.0f} KB - {max(sizes)/1024/1024:.1f} MB")

        if quality_metrics['formats']:
            print(f"\n📁 FORMAT DISTRIBUTION:")
            for fmt, count in quality_metrics['formats'].items():
                print(f"   {fmt}: {count} files ({count/len(sample_images)*100:.1f}%)")

        if quality_metrics['blur_scores']:
            blur_scores = quality_metrics['blur_scores']
            blur_threshold = 100  # Typical threshold for blur detection
            sharp_count = sum(1 for score in blur_scores if score > blur_threshold)

            print(f"\n🔍 IMAGE SHARPNESS:")
            print(f"   Sharp images: {sharp_count}/{len(blur_scores)} ({sharp_count/len(blur_scores)*100:.1f}%)")
            print(f"   Average blur score: {np.mean(blur_scores):.1f}")

        self.analysis_results['quality'] = quality_metrics
        return quality_metrics

    def assess_medical_sufficiency(self):
        """Overall medical sufficiency assessment"""
        print("\n" + "="*70)
        print("🏥 MEDICAL SUFFICIENCY ASSESSMENT")
        print("="*70)

        total_images = self.analysis_results.get('total_images', 0)
        class_counts = self.analysis_results.get('overall_counts', {})
        balance_info = self.analysis_results.get('balance', {})

        # Total dataset size assessment
        standards = self.medical_standards['total_samples']
        if total_images >= standards['excellent']:
            size_grade = "✅ EXCELLENT"
            size_status = "Outstanding dataset size for medical research"
        elif total_images >= standards['good']:
            size_grade = "🟡 GOOD"
            size_status = "Good dataset size, suitable for clinical validation"
        elif total_images >= standards['minimum']:
            size_grade = "⚠️ ACCEPTABLE"
            size_status = "Minimum acceptable size, consider expanding"
        else:
            size_grade = "❌ INSUFFICIENT"
            size_status = "Dataset too small for reliable medical AI training"

        print(f"📊 DATASET SIZE: {total_images:,} images")
        print(f"🏆 Size Grade: {size_grade}")
        print(f"📝 Status: {size_status}")

        # Per-class sufficiency
        min_class_samples = min(class_counts.values()) if class_counts else 0
        class_standards = self.medical_standards['min_samples_per_class']

        if min_class_samples >= class_standards['medical_production']:
            class_grade = "✅ MEDICAL PRODUCTION"
            class_status = "All classes meet medical production standards"
        elif min_class_samples >= class_standards['clinical_validation']:
            class_grade = "🟡 CLINICAL VALIDATION"
            class_status = "Suitable for clinical validation studies"
        elif min_class_samples >= class_standards['research']:
            class_grade = "📈 RESEARCH LEVEL"
            class_status = "Adequate for research and development"
        else:
            class_grade = "❌ INSUFFICIENT"
            class_status = "Some classes below minimum requirements"

        print(f"\n📋 PER-CLASS SUFFICIENCY:")
        print(f"   Minimum class size: {min_class_samples:,}")
        print(f"🏆 Class Grade: {class_grade}")
        print(f"📝 Status: {class_status}")

        # Overall recommendation
        balance_ratio = balance_info.get('ratio', float('inf'))

        print(f"\n" + "="*70)
        print("🎯 OVERALL RECOMMENDATION")
        print("="*70)

        # Determine if dataset is sufficient for target accuracy (96.96%)
        if (total_images >= standards['good'] and
            min_class_samples >= class_standards['clinical_validation'] and
            balance_ratio <= 10.0):

            recommendation = "✅ DATASET SUFFICIENT for medical-grade ensemble training"
            confidence = "HIGH CONFIDENCE of achieving 96.96% target accuracy"
            next_steps = [
                "Proceed with ensemble training",
                "Use class weights and SMOTE for imbalance",
                "Apply medical-grade preprocessing (CLAHE)",
                "Implement proper validation protocols"
            ]

        elif (total_images >= standards['minimum'] and
              min_class_samples >= class_standards['research']):

            recommendation = "⚠️ DATASET PARTIALLY SUFFICIENT"
            confidence = "MODERATE CONFIDENCE - may need optimization"
            next_steps = [
                "Proceed with caution",
                "Implement heavy data augmentation",
                "Use advanced techniques (SMOTE, focal loss)",
                "Consider additional data collection",
                "Monitor training carefully for overfitting"
            ]

        else:
            recommendation = "❌ DATASET INSUFFICIENT for medical-grade training"
            confidence = "LOW CONFIDENCE of achieving target accuracy"
            next_steps = [
                "Collect more data before training",
                "Focus on underrepresented classes",
                "Consider synthetic data generation",
                "Implement transfer learning with medical datasets",
                "Re-evaluate targets and expectations"
            ]

        print(f"🏆 {recommendation}")
        print(f"📊 {confidence}")
        print(f"\n📝 RECOMMENDED NEXT STEPS:")
        for i, step in enumerate(next_steps, 1):
            print(f"   {i}. {step}")

        # Research comparison
        print(f"\n📚 RESEARCH COMPARISON:")
        print(f"   Target: EfficientNetB2 (96.27%), ResNet50 (94.95%), DenseNet121 (91.21%)")
        print(f"   Ensemble Target: 96.96% accuracy")
        print(f"   Dataset Quality: {'Meets' if 'SUFFICIENT' in recommendation else 'Below'} research standards")

        self.analysis_results['medical_assessment'] = {
            'size_grade': size_grade,
            'class_grade': class_grade,
            'recommendation': recommendation,
            'confidence': confidence,
            'next_steps': next_steps
        }

        return recommendation, confidence

    def generate_report(self, output_file=None):
        """Generate comprehensive analysis report"""
        if output_file is None:
            output_file = f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'medical_standards': self.medical_standards,
            'results': self.analysis_results
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed analysis report saved: {output_file}")
        return output_file

    def run_full_analysis(self):
        """Run complete dataset analysis"""
        print("🔬 COMPREHENSIVE MEDICAL DATASET ANALYSIS")
        print("="*70)
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"⏱️  Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Class distribution
            distribution, class_counts, total = self.analyze_class_distribution()

            # Step 2: Class balance assessment
            balance_ratio, balance_grade = self.assess_class_balance(class_counts)

            # Step 3: Image quality analysis
            quality_metrics = self.analyze_image_quality(sample_size=200)

            # Step 4: Medical sufficiency assessment
            recommendation, confidence = self.assess_medical_sufficiency()

            # Step 5: Generate report
            report_file = self.generate_report()

            print(f"\n" + "="*70)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"📊 Total Images Analyzed: {total:,}")
            print(f"📄 Report Generated: {report_file}")

            return self.analysis_results

        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Deep Medical Dataset Analysis')
    parser.add_argument('--dataset_path', default='./dataset6',
                       help='Path to dataset directory')
    parser.add_argument('--sample_size', type=int, default=200,
                       help='Number of images to sample for quality analysis')
    parser.add_argument('--output_report',
                       help='Output file for analysis report (default: auto-generated)')

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path '{args.dataset_path}' does not exist!")
        return 1

    analyzer = MedicalDatasetAnalyzer(args.dataset_path)
    results = analyzer.run_full_analysis()

    if results:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
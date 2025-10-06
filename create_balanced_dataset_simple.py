#!/usr/bin/env python3
"""
Simplified Dataset Balancing using Heavy Augmentation
Fast approach for achieving 90%+ per-class accuracy

Strategy:
- Class 0: Undersample to 10,000
- Class 1-4: Heavy augmentation to reach 8,000 samples
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
import random
import argparse
import json
from datetime import datetime

class SimpleDatasetBalancer:
    """Fast dataset balancing using heavy augmentation only"""

    def __init__(self, input_dir: str, output_dir: str, target_samples: int = 8000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_samples = target_samples
        self.stats = {'original': {}, 'balanced': {}}

        random.seed(42)
        np.random.seed(42)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    def augment_image(self, img: np.ndarray, severity: str = 'moderate') -> np.ndarray:
        """Apply random augmentation"""
        aug_img = img.copy()

        # Rotation
        if random.random() < 0.7:
            angle = random.uniform(-20 if severity == 'heavy' else -15, 20 if severity == 'heavy' else 15)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))

        # Flip
        if random.random() < 0.5:
            aug_img = cv2.flip(aug_img, 1)

        # Brightness/contrast
        if random.random() < 0.7:
            alpha = random.uniform(0.85, 1.15)  # Contrast
            beta = random.randint(-20, 20)  # Brightness
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        # Zoom
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            h, w = aug_img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            aug_img = cv2.resize(aug_img, (new_w, new_h))
            if scale > 1.0:
                # Crop center
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                aug_img = aug_img[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                aug_img = cv2.copyMakeBorder(aug_img, pad_y, h-new_h-pad_y,
                                             pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)

        return aug_img

    def balance_class(self, class_id: int):
        """Balance specific class"""
        print(f"\n{'='*70}")
        print(f"Processing Class {class_id}")
        print(f"{'='*70}")

        class_path = self.input_dir / 'train' / str(class_id)
        output_path = self.output_dir / 'train' / str(class_id)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all images
        img_files = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        original_count = len(img_files)
        self.stats['original'][class_id] = original_count

        print(f"Original: {original_count:,} images")
        print(f"Target: {self.target_samples:,} images")

        # Determine strategy
        if class_id == 0:
            # Undersample Class 0
            img_files = random.sample(img_files, min(10000, len(img_files)))
            multiplier = 1
            use_clahe = False
            print(f"Strategy: Undersample to {len(img_files):,}")
        elif class_id in [1, 2]:
            # Moderate augmentation
            multiplier = max(1, self.target_samples // original_count + 1)
            use_clahe = True
            print(f"Strategy: Moderate augmentation (x{multiplier}) + CLAHE")
        else:
            # Heavy augmentation for Class 3, 4
            multiplier = max(1, self.target_samples // original_count + 2)
            use_clahe = True
            print(f"Strategy: Heavy augmentation (x{multiplier}) + CLAHE")

        #  Process images
        saved_count = 0
        batch_size = 100

        for batch_idx in range(0, len(img_files), batch_size):
            batch_files = img_files[batch_idx:batch_idx+batch_size]

            for img_path in batch_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))

                    # Apply CLAHE if needed
                    if use_clahe:
                        img = self.apply_clahe(img)

                    # Save original
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_file = output_path / f"class{class_id}_{saved_count:06d}.jpg"
                    cv2.imwrite(str(out_file), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1

                    if saved_count >= self.target_samples:
                        break

                    # Generate augmented versions
                    severity = 'heavy' if class_id in [3, 4] else 'moderate'
                    for aug_idx in range(multiplier - 1):
                        if saved_count >= self.target_samples:
                            break

                        aug_img = self.augment_image(img, severity)
                        aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                        out_file = output_path / f"class{class_id}_{saved_count:06d}.jpg"
                        cv2.imwrite(str(out_file), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved_count += 1

                except Exception as e:
                    print(f"Warning: Failed {img_path}: {e}")
                    continue

            if saved_count >= self.target_samples:
                break

            if (batch_idx + batch_size) % 500 == 0:
                print(f"  Progress: {saved_count:,} / {self.target_samples:,}")

        self.stats['balanced'][class_id] = saved_count
        print(f"✅ Class {class_id} complete: {saved_count:,} images")

    def copy_val_test(self):
        """Copy validation and test sets"""
        print(f"\n{'='*70}")
        print("Copying validation and test sets")
        print(f"{'='*70}")

        for split in ['val', 'test']:
            src = self.input_dir / split
            dst = self.output_dir / split

            if not src.exists():
                continue

            print(f"\nCopying {split} set...")
            for class_id in range(5):
                src_class = src / str(class_id)
                dst_class = dst / str(class_id)

                if not src_class.exists():
                    continue

                dst_class.mkdir(parents=True, exist_ok=True)
                img_files = list(src_class.glob('*.jpeg')) + list(src_class.glob('*.jpg')) + list(src_class.glob('*.png'))

                for img_file in img_files:
                    shutil.copy2(img_file, dst_class / img_file.name)

                print(f"  Class {class_id}: {len(img_files):,} images")

    def generate_report(self):
        """Generate balance report"""
        print(f"\n{'='*70}")
        print("BALANCE REPORT")
        print(f"{'='*70}")

        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output: {self.output_dir}")

        print(f"\n{'-'*70}")
        print("BEFORE → AFTER")
        print(f"{'-'*70}")

        total_before = sum(self.stats['original'].values())
        total_after = sum(self.stats['balanced'].values())

        for class_id in range(5):
            before = self.stats['original'][class_id]
            after = self.stats['balanced'][class_id]
            change = after - before
            pct = (change / before * 100) if before > 0 else 0
            print(f"Class {class_id}: {before:6,} → {after:6,} ({change:+6,}, {pct:+6.1f}%)")

        print(f"{'-'*70}")
        print(f"Total:   {total_before:6,} → {total_after:6,} ({total_after - total_before:+6,})")

        # Save JSON report
        report_file = self.output_dir / 'balance_report.json'
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'target_samples': self.target_samples,
                'original': self.stats['original'],
                'balanced': self.stats['balanced']
            }, f, indent=2)

        print(f"\n✅ Report saved: {report_file}")

    def run(self):
        """Execute balancing"""
        print(f"{'='*70}")
        print("FAST DATASET BALANCING")
        print(f"={'='*70}\n")

        for class_id in range(5):
            self.balance_class(class_id)

        self.copy_val_test()
        self.generate_report()

        print(f"\n{'='*70}")
        print("✅ BALANCING COMPLETE")
        print(f"={'='*70}")
        print(f"\nBalanced dataset: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Fast dataset balancing')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_samples', type=int, default=8000)
    args = parser.parse_args()

    balancer = SimpleDatasetBalancer(args.input_dir, args.output_dir, args.target_samples)
    balancer.run()

if __name__ == '__main__':
    main()

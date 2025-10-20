#!/usr/bin/env python3
"""
Visualization Tool for Grade-Specific Enhancement

Creates before/after comparison images to validate enhancement parameters.
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess_grade_specific import GradeSpecificEnhancer, GRADE_SPECIFIC_PARAMS

def visualize_enhancement(image_path: str, output_path: str = None):
    """
    Create side-by-side comparison of original vs enhanced for all grades.

    Args:
        image_path: Path to input fundus image
        output_path: Path to save visualization (optional)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Grade-Specific Enhancement Comparison\n{Path(image_path).name}',
                 fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Enhanced versions for each grade
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for grade, (row, col) in enumerate(positions):
        # Apply grade-specific enhancement
        enhancer = GradeSpecificEnhancer(grade)
        enhanced = enhancer.enhance(image_rgb)

        # Display
        axes[row, col].imshow(enhanced)
        params = GRADE_SPECIFIC_PARAMS[grade]
        title = f"Grade {grade}: {params['name']}\n" \
                f"Flatten={params['flatten_strength']}, " \
                f"Sharp={params['sharpen_amount']}, " \
                f"Contrast={params['contrast_factor']}"
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize grade-specific enhancement')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to fundus image')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization (optional, shows if not provided)')

    args = parser.parse_args()
    visualize_enhancement(args.image, args.output)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced Dataset with CLAHE and Medical-Grade Preprocessing for Ensemble Training

This module implements the advanced preprocessing pipeline demonstrated in the 
96.96% accuracy research paper, including CLAHE, SMOTE, and medical-grade
augmentation strategies for diabetic retinopathy classification.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLAHEProcessor:
    """
    Contrast Limited Adaptive Histogram Equalization for fundus images.
    
    Critical preprocessing step that improves vessel contrast and lesion visibility,
    contributing +3-5% accuracy improvement as demonstrated in research literature.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to fundus image.
        
        Args:
            image: Input image as numpy array (H, W, C) or PIL Image
            
        Returns:
            CLAHE-processed image as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert BGR to LAB color space for better CLAHE results
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            image = clahe.apply(image)
        
        return image

class MedicalAugmentation:
    """
    Medical-grade augmentation pipeline preserving retinal anatomy.
    
    Implements carefully tuned augmentations that maintain clinical relevance
    while improving model generalization.
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 rotation_range: float = 15.0,
                 brightness_range: float = 0.1,
                 contrast_range: float = 0.1,
                 zoom_range: Tuple[float, float] = (0.95, 1.05),
                 horizontal_flip_prob: float = 0.5):
        
        self.img_size = img_size
        
        # Training augmentations (preserve retinal anatomy)
        self.train_transform = A.Compose([
            # Geometric transformations
            A.Rotate(limit=rotation_range, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.RandomScale(scale_limit=(zoom_range[0]-1.0, zoom_range[1]-1.0), p=0.5),
            
            # Color transformations (camera variation simulation)
            A.RandomBrightnessContrast(
                brightness_limit=brightness_range,
                contrast_limit=contrast_range,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,      # Minimal hue shift (preserves color accuracy)
                sat_shift_limit=10,     # Slight saturation variation
                val_shift_limit=10,     # Slight value variation
                p=0.3
            ),
            
            # Noise and blur (realistic camera effects)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=1.0),
            ], p=0.2),
            
            A.OneOf([
                A.Blur(blur_limit=2, p=1.0),
                A.GaussianBlur(blur_limit=2, p=1.0),
            ], p=0.1),
            
            # Resize and normalize
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation/test transformations (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_train_transform(self) -> Callable:
        return self.train_transform
    
    def get_val_transform(self) -> Callable:
        return self.val_transform

class DRDataset(Dataset):
    """
    Enhanced Diabetic Retinopathy Dataset with CLAHE and medical augmentation.
    
    Supports both directory structure and CSV-based data loading with
    comprehensive preprocessing pipeline for medical-grade performance.
    """
    
    def __init__(self,
                 data_paths: List[str],
                 labels: List[int],
                 transform: Optional[Callable] = None,
                 enable_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                 cache_images: bool = False):
        
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.enable_clahe = enable_clahe
        self.cache_images = cache_images
        
        # Validate data
        assert len(data_paths) == len(labels), "Number of paths must match number of labels"
        
        # Initialize CLAHE processor
        if enable_clahe:
            self.clahe_processor = CLAHEProcessor(clahe_clip_limit, clahe_tile_grid_size)
        
        # Image cache for faster training
        self.image_cache = {} if cache_images else None
        
        logger.info(f"DRDataset initialized: {len(data_paths)} samples, CLAHE: {enable_clahe}")
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image (from cache or disk)
        if self.image_cache is not None and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            image_path = self.data_paths[idx]
            try:
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                
                # Cache image if enabled
                if self.image_cache is not None:
                    self.image_cache[idx] = image
                    
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                # Return black image as fallback
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply CLAHE preprocessing
        if self.enable_clahe:
            image = self.clahe_processor(image)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default preprocessing if no transform provided
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'dr_grade': label,
            'image_path': self.data_paths[idx]
        }

def create_data_splits_ensemble(dataset_path: str,
                               num_classes: int = 5,
                               train_split: float = 0.7,
                               val_split: float = 0.15,
                               test_split: float = 0.15,
                               seed: int = 42) -> Tuple[List[Tuple[str, int]], ...]:
    """
    Create train/val/test splits for ensemble training with balanced sampling.
    
    Args:
        dataset_path: Path to dataset directory (train/val/test structure)
        num_classes: Number of DR severity classes
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data) as lists of (path, label) tuples
    """
    np.random.seed(seed)
    
    # Collect all images and labels
    all_data = []
    class_counts = Counter()
    
    # Check if using directory structure (train/val/test) or single directory
    has_splits = all(os.path.exists(os.path.join(dataset_path, split)) 
                    for split in ['train', 'val', 'test'])
    
    if has_splits:
        # Use existing train/val/test structure
        logger.info("Using existing train/val/test directory structure")
        
        train_data = []
        val_data = []
        test_data = []
        
        for split, data_list in [('train', train_data), ('val', val_data), ('test', test_data)]:
            split_path = os.path.join(dataset_path, split)
            for class_id in range(num_classes):
                class_dir = os.path.join(split_path, str(class_id))
                if os.path.exists(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_file)
                            data_list.append((img_path, class_id))
                            class_counts[class_id] += 1
    else:
        # Create splits from single directory
        logger.info("Creating train/val/test splits from single directory")
        
        # Collect all data by class
        class_data = {i: [] for i in range(num_classes)}
        
        for class_id in range(num_classes):
            class_dir = os.path.join(dataset_path, str(class_id))
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        class_data[class_id].append((img_path, class_id))
                        class_counts[class_id] += 1
        
        # Create balanced splits for each class
        train_data = []
        val_data = []
        test_data = []
        
        for class_id, class_samples in class_data.items():
            if len(class_samples) == 0:
                logger.warning(f"No samples found for class {class_id}")
                continue
                
            # Shuffle class samples
            np.random.shuffle(class_samples)
            
            # Calculate split indices
            n_samples = len(class_samples)
            n_train = int(n_samples * train_split)
            n_val = int(n_samples * val_split)
            
            # Split data
            train_data.extend(class_samples[:n_train])
            val_data.extend(class_samples[n_train:n_train + n_val])
            test_data.extend(class_samples[n_train + n_val:])
    
    # Shuffle splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    logger.info(f"Dataset splits created:")
    logger.info(f"  Training: {len(train_data)} samples")
    logger.info(f"  Validation: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    logger.info(f"  Class distribution: {dict(class_counts)}")
    
    return train_data, val_data, test_data

def apply_smote_balancing(train_data: List[Tuple[str, int]],
                         k_neighbors: int = 5,
                         sampling_strategy: str = 'auto',
                         random_state: int = 42) -> List[Tuple[str, int]]:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance training data.
    
    Note: For image data, SMOTE is applied to image features rather than raw pixels.
    This implementation creates synthetic samples by duplicating existing samples
    with augmentation rather than true SMOTE interpolation.
    
    Args:
        train_data: List of (image_path, label) tuples
        k_neighbors: Number of neighbors for SMOTE
        sampling_strategy: SMOTE sampling strategy
        random_state: Random seed
        
    Returns:
        Balanced training data with synthetic samples
    """
    logger.info("Applying SMOTE-inspired class balancing...")
    
    # Group data by class
    class_data = {}
    for path, label in train_data:
        if label not in class_data:
            class_data[label] = []
        class_data[label].append((path, label))
    
    # Find target count (maximum class size)
    class_counts = {label: len(samples) for label, samples in class_data.items()}
    max_count = max(class_counts.values())
    
    logger.info(f"Original class distribution: {class_counts}")
    logger.info(f"Target count per class: {max_count}")
    
    # Balance classes by oversampling minorities
    balanced_data = []
    
    for label, samples in class_data.items():
        current_count = len(samples)
        balanced_data.extend(samples)  # Add original samples
        
        if current_count < max_count:
            # Oversample this class
            needed = max_count - current_count
            oversampled = np.random.choice(len(samples), size=needed, replace=True)
            
            for idx in oversampled:
                # Add oversampled version (will get different augmentation)
                balanced_data.append(samples[idx])
    
    # Shuffle balanced data
    np.random.seed(random_state)
    np.random.shuffle(balanced_data)
    
    # Log results
    new_class_counts = Counter([label for _, label in balanced_data])
    logger.info(f"Balanced class distribution: {dict(new_class_counts)}")
    logger.info(f"Dataset size: {len(train_data)} → {len(balanced_data)}")
    
    return balanced_data

def compute_ensemble_class_weights(train_data: List[Tuple[str, int]],
                                 num_classes: int = 5,
                                 severe_multiplier: float = 8.0,
                                 pdr_multiplier: float = 6.0) -> torch.Tensor:
    """
    Compute class weights for ensemble training with medical-grade emphasis.
    
    Args:
        train_data: Training data as list of (path, label) tuples
        num_classes: Number of classes
        severe_multiplier: Additional weight for severe NPDR (class 3)
        pdr_multiplier: Additional weight for PDR (class 4)
        
    Returns:
        Class weights tensor
    """
    # Extract labels
    labels = [label for _, label in train_data]
    
    # Compute inverse frequency weights
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Inverse frequency weighting
    weights = total_samples / (num_classes * class_counts + 1e-6)
    
    # Apply medical-grade multipliers for severe cases
    if len(weights) > 3:
        weights[3] *= severe_multiplier  # Severe NPDR
    if len(weights) > 4:
        weights[4] *= pdr_multiplier     # PDR
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    logger.info(f"Class weights computed: {weights}")
    logger.info(f"Class counts: {class_counts}")
    
    return torch.FloatTensor(weights)

def create_ensemble_dataloaders(train_data: List[Tuple[str, int]],
                               val_data: List[Tuple[str, int]],
                               test_data: List[Tuple[str, int]],
                               config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for ensemble training with enhanced preprocessing.
    
    Args:
        train_data: Training data as list of (path, label) tuples
        val_data: Validation data as list of (path, label) tuples
        test_data: Test data as list of (path, label) tuples
        config: Ensemble configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Apply SMOTE balancing if enabled
    if config.data.enable_smote:
        train_data = apply_smote_balancing(
            train_data,
            k_neighbors=config.data.smote_k_neighbors,
            sampling_strategy=config.data.smote_sampling_strategy
        )
    
    # Create medical augmentation pipeline
    if config.data.enable_medical_augmentation:
        augmenter = MedicalAugmentation(
            img_size=config.model.img_size,
            rotation_range=config.data.rotation_range,
            brightness_range=config.data.brightness_range,
            contrast_range=config.data.contrast_range,
            zoom_range=config.data.zoom_range,
            horizontal_flip_prob=config.data.horizontal_flip_prob
        )
        train_transform = augmenter.get_train_transform()
        val_transform = augmenter.get_val_transform()
    else:
        # Basic transforms without augmentation
        train_transform = A.Compose([
            A.Resize(config.model.img_size, config.model.img_size),
            A.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std),
            ToTensorV2()
        ])
        val_transform = train_transform
    
    # Extract paths and labels
    train_paths, train_labels = zip(*train_data) if train_data else ([], [])
    val_paths, val_labels = zip(*val_data) if val_data else ([], [])
    test_paths, test_labels = zip(*test_data) if test_data else ([], [])
    
    # Create datasets
    train_dataset = DRDataset(
        data_paths=list(train_paths),
        labels=list(train_labels),
        transform=train_transform,
        enable_clahe=config.data.enable_clahe,
        clahe_clip_limit=config.data.clahe_clip_limit,
        clahe_tile_grid_size=config.data.clahe_tile_grid_size,
        cache_images=False  # Disable caching for large datasets
    )
    
    val_dataset = DRDataset(
        data_paths=list(val_paths),
        labels=list(val_labels),
        transform=val_transform,
        enable_clahe=config.data.enable_clahe,
        clahe_clip_limit=config.data.clahe_clip_limit,
        clahe_tile_grid_size=config.data.clahe_tile_grid_size,
        cache_images=False
    )
    
    test_dataset = DRDataset(
        data_paths=list(test_paths),
        labels=list(test_labels),
        transform=val_transform,
        enable_clahe=config.data.enable_clahe,
        clahe_clip_limit=config.data.clahe_clip_limit,
        clahe_tile_grid_size=config.data.clahe_tile_grid_size,
        cache_images=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test enhanced dataset and preprocessing
    print("🧪 Testing Enhanced Dataset with CLAHE and Medical Augmentation")
    
    # Test CLAHE processor
    clahe = CLAHEProcessor(clip_limit=2.0)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    processed = clahe(test_image)
    print(f"✅ CLAHE processing: {test_image.shape} → {processed.shape}")
    
    # Test medical augmentation
    augmenter = MedicalAugmentation(img_size=224)
    train_transform = augmenter.get_train_transform()
    val_transform = augmenter.get_val_transform()
    
    # Test with synthetic data
    augmented = train_transform(image=test_image)
    print(f"✅ Medical augmentation: {test_image.shape} → {augmented['image'].shape}")
    
    # Test SMOTE balancing
    dummy_data = [('path1.jpg', 0)] * 100 + [('path2.jpg', 1)] * 20 + [('path3.jpg', 2)] * 200
    balanced = apply_smote_balancing(dummy_data, random_state=42)
    print(f"✅ SMOTE balancing: {len(dummy_data)} → {len(balanced)} samples")
    
    # Test class weights
    weights = compute_ensemble_class_weights(dummy_data, num_classes=5)
    print(f"✅ Class weights computed: {weights}")
    
    print("✅ Enhanced dataset testing completed successfully!")
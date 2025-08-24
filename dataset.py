import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import json
from sklearn.model_selection import train_test_split
from utils import compute_class_weights

class DiabeticRetinopathyDataset(Dataset):
    """Dataset class for diabetic retinopathy classification."""
    
    def __init__(self, 
                 data_info: List[Dict], 
                 transform: Optional[A.Compose] = None,
                 img_size: int = 224):
        """
        Args:
            data_info: List of dictionaries containing image info
            transform: Albumentations transform pipeline
            img_size: Target image size
        """
        self.data_info = data_info
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self) -> int:
        return len(self.data_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_info[idx]
        
        # Load image (handle both local and GCS paths)
        image_path = item['image_path']
        
        if image_path.startswith('gs://'):
            # Download from GCS and read into memory
            from google.cloud import storage
            import tempfile
            import os
            
            # Parse GCS path
            path_parts = image_path.replace('gs://', '').split('/')
            bucket_name = path_parts[0]
            blob_name = '/'.join(path_parts[1:])
            
            # Download to temporary file
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image_path)[1], delete=False) as tmp_file:
                blob.download_to_filename(tmp_file.name)
                image = cv2.imread(tmp_file.name)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
        else:
            # Local file
            image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Basic resize and normalization
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Base sample
        sample = {
            'image': image,
            'image_path': image_path,
            'filename': item.get('filename', os.path.basename(image_path))
        }
        
        # Handle different dataset types
        if 'dr_grade' in item:
            # Dataset type 1: single DR grade
            sample['dr_grade'] = torch.tensor(item['dr_grade'], dtype=torch.long)
            # Generate synthetic enhanced labels for DR grade
            enhanced_labels = self._generate_synthetic_labels_dr(item['dr_grade'])
            sample.update(enhanced_labels)
        else:
            # Dataset type 0: RG and ME grades
            sample['rg_grade'] = torch.tensor(item['rg_grade'], dtype=torch.long)
            sample['me_grade'] = torch.tensor(item['me_grade'], dtype=torch.long)
            # Add synthetic enhanced labels for training
            enhanced_labels = self._generate_synthetic_labels(item['rg_grade'], item['me_grade'])
            sample.update(enhanced_labels)
        
        # Add any existing enhanced labels from item
        for key in ['referable_dr', 'sight_threatening', 'pdr_activity', 
                   'hemorrhages_4q', 'venous_beading_2q', 'irma_1q', 'meets_421',
                   'nvd_area', 'nve_area', 'nv_activity', 'image_quality', 
                   'confidence_target', 'lesion_bbox']:
            if key in item:
                if key == 'lesion_bbox':
                    sample[key] = torch.tensor(item[key], dtype=torch.float32)
                elif key in ['image_quality', 'confidence_target']:
                    sample[key] = torch.tensor(item[key], dtype=torch.float32)
                elif key in ['referable_dr', 'sight_threatening', 'hemorrhages_4q', 
                           'venous_beading_2q', 'irma_1q', 'meets_421']:
                    sample[key] = torch.tensor(item[key], dtype=torch.long)
                else:
                    sample[key] = torch.tensor(item[key], dtype=torch.long)
        
        return sample
    
    def _generate_synthetic_labels(self, rg_grade: int, me_grade: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic enhanced labels for training."""
        
        enhanced = {}
        
        # Clinical workflow labels
        enhanced['referable_dr'] = torch.tensor(int(rg_grade >= 2 or me_grade >= 1), dtype=torch.long)
        enhanced['sight_threatening'] = torch.tensor(int(rg_grade >= 4 or me_grade == 2), dtype=torch.long)
        
        # PDR activity (synthetic based on RG grade)
        if rg_grade == 4:
            pdr_activity = np.random.choice([1, 2], p=[0.3, 0.7])  # suspected or present
        elif rg_grade == 3:
            pdr_activity = np.random.choice([0, 1], p=[0.7, 0.3])  # absent or suspected
        else:
            pdr_activity = 0  # absent
        enhanced['pdr_activity'] = torch.tensor(pdr_activity, dtype=torch.long)
        
        # ETDRS 4-2-1 rule components (synthetic)
        if rg_grade >= 3:
            # Higher chance of positive findings in severe NPDR
            enhanced['hemorrhages_4q'] = torch.tensor(np.random.choice([0, 1], p=[0.3, 0.7]), dtype=torch.long)
            enhanced['venous_beading_2q'] = torch.tensor(np.random.choice([0, 1], p=[0.4, 0.6]), dtype=torch.long)
            enhanced['irma_1q'] = torch.tensor(np.random.choice([0, 1], p=[0.2, 0.8]), dtype=torch.long)
        elif rg_grade >= 2:
            # Moderate findings in moderate NPDR
            enhanced['hemorrhages_4q'] = torch.tensor(np.random.choice([0, 1], p=[0.6, 0.4]), dtype=torch.long)
            enhanced['venous_beading_2q'] = torch.tensor(np.random.choice([0, 1], p=[0.8, 0.2]), dtype=torch.long)
            enhanced['irma_1q'] = torch.tensor(np.random.choice([0, 1], p=[0.7, 0.3]), dtype=torch.long)
        else:
            # Minimal findings in mild/no DR
            enhanced['hemorrhages_4q'] = torch.tensor(0, dtype=torch.long)
            enhanced['venous_beading_2q'] = torch.tensor(0, dtype=torch.long)
            enhanced['irma_1q'] = torch.tensor(0, dtype=torch.long)
        
        # Meets 4-2-1 rule (combination of above)
        meets_421 = int(enhanced['hemorrhages_4q'].item() and 
                       enhanced['venous_beading_2q'].item() and 
                       enhanced['irma_1q'].item())
        enhanced['meets_421'] = torch.tensor(meets_421, dtype=torch.long)
        
        # Image quality (synthetic)
        base_quality = np.random.normal(3.0, 0.5)  # Mean quality around 3/4
        if rg_grade >= 3:
            base_quality += np.random.normal(0, 0.3)  # More variation in severe cases
        enhanced['image_quality'] = torch.tensor(np.clip(base_quality, 0, 4), dtype=torch.float32)
        
        # Confidence target (synthetic - higher confidence for clear cases)
        if rg_grade == 0 or rg_grade == 4:
            confidence = np.random.beta(8, 2)  # High confidence for clear cases
        elif rg_grade in [1, 2]:
            confidence = np.random.beta(4, 3)  # Moderate confidence  
        else:
            confidence = np.random.beta(3, 4)  # Lower confidence for borderline cases
        enhanced['confidence_target'] = torch.tensor(confidence, dtype=torch.float32)
        
        return enhanced
    
    def _generate_synthetic_labels_dr(self, dr_grade: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic enhanced labels for DR classification (dataset type 1)."""
        
        enhanced = {}
        
        # Map DR grade (0-4) to clinical workflow labels
        # DR grades: 0=No DR, 1=Mild NPDR, 2=Moderate NPDR, 3=Severe NPDR, 4=PDR
        enhanced['referable_dr'] = torch.tensor(int(dr_grade >= 2), dtype=torch.long)
        enhanced['sight_threatening'] = torch.tensor(int(dr_grade >= 3), dtype=torch.long)
        
        # PDR activity (only for grade 4 - PDR)
        if dr_grade == 4:
            enhanced['pdr_activity'] = torch.tensor(np.random.choice([0, 1]), dtype=torch.long)
        else:
            enhanced['pdr_activity'] = torch.tensor(0, dtype=torch.long)
        
        # Synthetic microaneurysm features
        if dr_grade >= 1:
            enhanced['hemorrhages_4q'] = torch.tensor(int(dr_grade >= 3), dtype=torch.long)
            enhanced['venous_beading_2q'] = torch.tensor(int(dr_grade >= 3), dtype=torch.long)
            enhanced['irma_1q'] = torch.tensor(int(dr_grade >= 3), dtype=torch.long)
        else:
            enhanced['hemorrhages_4q'] = torch.tensor(0, dtype=torch.long)
            enhanced['venous_beading_2q'] = torch.tensor(0, dtype=torch.long)
            enhanced['irma_1q'] = torch.tensor(0, dtype=torch.long)
        
        # 4-2-1 rule compliance (severe NPDR criteria)
        enhanced['meets_421'] = torch.tensor(int(dr_grade == 3), dtype=torch.long)
        
        # Synthetic confidence score
        if dr_grade in [0, 4]:  # Clear cases
            confidence = np.random.beta(8, 2)  # Higher confidence
        else:  # Borderline cases
            confidence = np.random.beta(3, 4)  # Lower confidence
        enhanced['confidence_target'] = torch.tensor(confidence, dtype=torch.float32)
        
        return enhanced

def get_transforms(img_size: int = 224, is_training: bool = True) -> A.Compose:
    """Get augmentation transforms for training/validation."""
    
    if is_training:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.7
            ),
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.CLAHE(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform

def create_data_splits(rg_path: str, me_path: str, 
                      train_split: float = 0.7, 
                      val_split: float = 0.15,
                      test_split: float = 0.15,
                      seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create train/validation/test splits from dataset directories."""
    
    data_info = []
    
    # Helper function to list files in GCS or local directories
    def list_files_in_directory(path: str) -> List[str]:
        if path.startswith('gs://'):
            from google.cloud import storage
            # Parse GCS path
            path_parts = path.replace('gs://', '').split('/')
            bucket_name = path_parts[0]
            prefix = '/'.join(path_parts[1:]) + '/' if len(path_parts) > 1 else ''
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            files = []
            for blob in blobs:
                # Only get direct files in this directory, not subdirectories
                relative_path = blob.name[len(prefix):]
                if '/' not in relative_path and relative_path.endswith(('.tif', '.jpg', '.png')):
                    files.append(relative_path)
            return files
        else:
            # Local filesystem
            if os.path.exists(path):
                return [f for f in os.listdir(path) if f.endswith(('.tif', '.jpg', '.png'))]
            return []
    
    def path_join(base_path: str, *paths: str) -> str:
        if base_path.startswith('gs://'):
            return base_path + '/' + '/'.join(paths)
        else:
            return os.path.join(base_path, *paths)
    
    # Get all RG images and their grades (now supports 0-4 including PDR)
    rg_images = {}
    for grade in range(5):  # RG grades 0-4 (including PDR)
        grade_dir = path_join(rg_path, str(grade))
        img_files = list_files_in_directory(grade_dir)
        
        for img_file in img_files:
            img_path = path_join(grade_dir, img_file)
            # Use filename as key to match with ME images
            rg_images[img_file] = {
                'rg_grade': grade,
                'rg_path': img_path
            }
    
    # Get all ME images and their grades
    me_images = {}
    for grade in range(4):  # ME grades 0-3 (though 3 might be empty)
        grade_dir = path_join(me_path, str(grade))
        img_files = list_files_in_directory(grade_dir)
        
        for img_file in img_files:
            img_path = path_join(grade_dir, img_file)
            me_images[img_file] = {
                'me_grade': grade,
                'me_path': img_path
            }
    
    # Match RG and ME images by filename
    for img_file, rg_info in rg_images.items():
        if img_file in me_images:
            me_info = me_images[img_file]
            data_info.append({
                'image_path': rg_info['rg_path'],  # Use RG path as primary
                'rg_grade': rg_info['rg_grade'],
                'me_grade': me_info['me_grade'],
                'filename': img_file
            })
    
    print(f"Total matched images: {len(data_info)}")
    
    # Print class distribution
    rg_dist = pd.Series([item['rg_grade'] for item in data_info]).value_counts().sort_index()
    me_dist = pd.Series([item['me_grade'] for item in data_info]).value_counts().sort_index()
    
    print("RG Grade distribution:")
    print(rg_dist)
    print("\nME Grade distribution:")
    print(me_dist)
    
    # Create stratified splits based on RG grades (primary task)
    rg_grades = [item['rg_grade'] for item in data_info]
    
    # First split: train + val vs test
    train_val_data, test_data = train_test_split(
        data_info, 
        test_size=test_split,
        random_state=seed,
        stratify=rg_grades
    )
    
    # Second split: train vs val
    train_val_rg = [item['rg_grade'] for item in train_val_data]
    val_size = val_split / (train_split + val_split)
    
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_rg
    )
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    
    return train_data, val_data, test_data

def create_dataloaders(train_data: List[Dict], 
                      val_data: List[Dict], 
                      test_data: List[Dict],
                      config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test sets."""
    
    # Auto-adjust num_workers based on system
    import os
    max_workers = min(config.data.num_workers, os.cpu_count() or 4)
    print(f"Using {max_workers} workers for DataLoader (requested: {config.data.num_workers})")
    
    # Get transforms
    train_transform = get_transforms(config.model.img_size, is_training=True)
    val_transform = get_transforms(config.model.img_size, is_training=False)
    
    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(
        train_data, 
        transform=train_transform,
        img_size=config.model.img_size
    )
    
    val_dataset = DiabeticRetinopathyDataset(
        val_data, 
        transform=val_transform,
        img_size=config.model.img_size
    )
    
    test_dataset = DiabeticRetinopathyDataset(
        test_data, 
        transform=val_transform,
        img_size=config.model.img_size
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=max_workers,
        pin_memory=config.data.pin_memory,
        drop_last=getattr(config.data, 'drop_last', True),
        prefetch_factor=getattr(config.data, 'prefetch_factor', 2),
        persistent_workers=getattr(config.data, 'persistent_workers', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=max_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=getattr(config.data, 'prefetch_factor', 2),
        persistent_workers=getattr(config.data, 'persistent_workers', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=max_workers,
        pin_memory=config.data.pin_memory
    )
    
    return train_loader, val_loader, test_loader

def compute_dataset_class_weights(data_info: List[Dict], num_classes_rg: int = 5, num_classes_me: int = 3):
    """Compute class weights for handling imbalanced classes."""
    
    # Extract labels
    rg_labels = np.array([item['rg_grade'] for item in data_info])
    me_labels = np.array([item['me_grade'] for item in data_info])
    
    # Compute class weights
    rg_weights = compute_class_weights(rg_labels, num_classes_rg)
    me_weights = compute_class_weights(me_labels, num_classes_me)
    
    print("RG Class weights:", rg_weights)
    print("ME Class weights:", me_weights)
    
    return rg_weights, me_weights

def compute_dr_class_weights(data_info: List[Dict], num_classes: int) -> np.ndarray:
    """Compute class weights for DR classification (dataset type 1)."""
    
    dr_labels = np.array([item['dr_grade'] for item in data_info])
    dr_weights = compute_class_weights(dr_labels, num_classes)
    
    print("DR Class weights:", dr_weights)
    
    return dr_weights

def create_data_splits_type1(dataset_path: str, num_classes: int = 5, 
                            seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create data splits from type 1 dataset structure (train/val/test/class directories)."""
    
    print(f"Loading dataset type 1 from: {dataset_path}")
    
    # Helper function to list files in GCS or local directories
    def list_files_in_directory(path: str) -> List[str]:
        if path.startswith('gs://'):
            from google.cloud import storage
            client = storage.Client()
            path_parts = path.replace('gs://', '').split('/')
            bucket_name = path_parts[0]
            prefix = '/'.join(path_parts[1:]) + '/'
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            return [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.lower().endswith(('.tif', '.jpg', '.jpeg', '.png'))]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                for filename in filenames:
                    if filename.lower().endswith(('.tif', '.jpg', '.jpeg', '.png')):
                        files.append(os.path.join(root, filename))
            return files
    
    train_data = []
    val_data = []
    test_data = []
    
    for split in ['train', 'val', 'test']:
        split_path = f"{dataset_path}/{split}"
        
        for class_id in range(num_classes):
            class_path = f"{split_path}/{class_id}"
            files = list_files_in_directory(class_path)
            
            for file_path in files:
                # For dataset type 1, only use DR grade
                # Don't create synthetic ME labels as they would be inaccurate
                data_item = {
                    'image_path': file_path,
                    'dr_grade': class_id,  # Primary: 5-class DR classification
                    'split': split,
                    'dataset_type': 1  # Mark as dataset type 1
                }
                
                if split == 'train':
                    train_data.append(data_item)
                elif split == 'val':
                    val_data.append(data_item)
                elif split == 'test':
                    test_data.append(data_item)
    
    print(f"Dataset loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data

def save_data_splits(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], 
                    output_dir: str = "data"):
    """Save data splits to JSON files for reproducibility."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train_split.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
        
    with open(os.path.join(output_dir, "val_split.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
        
    with open(os.path.join(output_dir, "test_split.json"), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Data splits saved to {output_dir}/")

def load_data_splits(data_dir: str = "data") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load previously saved data splits."""
    
    with open(os.path.join(data_dir, "train_split.json"), 'r') as f:
        train_data = json.load(f)
        
    with open(os.path.join(data_dir, "val_split.json"), 'r') as f:
        val_data = json.load(f)
        
    with open(os.path.join(data_dir, "test_split.json"), 'r') as f:
        test_data = json.load(f)
    
    return train_data, val_data, test_data
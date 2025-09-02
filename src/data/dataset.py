"""Dataset management and preprocessing utilities."""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset organization, splitting, and preprocessing."""
    
    def __init__(self, dataset_root: str = "data"):
        self.dataset_root = Path(dataset_root)
        self.raw_dir = self.dataset_root / "raw"
        self.processed_dir = self.dataset_root / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def create_yolo_dataset_structure(
        self,
        images_dir: str,
        labels_dir: str,
        train_split: float = 0.8,
        val_split: float = 0.15,
        test_split: float = 0.05,
        class_names: List[str] = None
    ) -> str:
        """Create YOLO-compatible dataset structure."""
        
        if class_names is None:
            class_names = ["bottle", "box"]
        
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Create output directory structure
        output_dir = self.processed_dir / "yolo_dataset"
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        images_path = Path(images_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_path.glob(ext))
        
        logger.info(f"Found {len(image_files)} images")
        
        # Filter images that have corresponding labels
        valid_pairs = []
        labels_path = Path(labels_dir)
        
        for img_file in image_files:
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
        
        logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
        
        if len(valid_pairs) == 0:
            raise ValueError("No valid image-label pairs found")
        
        # Split dataset
        train_pairs, temp_pairs = train_test_split(
            valid_pairs, 
            test_size=(val_split + test_split),
            random_state=42
        )
        
        val_pairs, test_pairs = train_test_split(
            temp_pairs,
            test_size=(test_split / (val_split + test_split)),
            random_state=42
        )
        
        # Copy files to appropriate directories
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        for split_name, pairs in splits.items():
            logger.info(f"Processing {split_name} split: {len(pairs)} samples")
            
            for img_file, label_file in pairs:
                # Copy image
                dst_img = output_dir / split_name / 'images' / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy label
                dst_label = output_dir / split_name / 'labels' / label_file.name
                shutil.copy2(label_file, dst_label)
        
        # Create dataset.yaml file
        dataset_config = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        dataset_yaml_path = output_dir / 'dataset.yaml'
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset created at: {output_dir}")
        logger.info(f"Dataset config saved at: {dataset_yaml_path}")
        
        return str(dataset_yaml_path)
    
    def analyze_dataset(self, dataset_yaml_path: str) -> Dict:
        """Analyze dataset statistics."""
        
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_root = Path(dataset_config['path'])
        class_names = dataset_config['names']
        
        analysis = {
            'class_names': class_names,
            'splits': {},
            'class_distribution': {name: 0 for name in class_names},
            'bbox_stats': {
                'widths': [],
                'heights': [],
                'areas': [],
                'aspect_ratios': [],
                'center_x': [],
                'center_y': []
            }
        }
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            split_dir = dataset_root / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob('*'))
            label_files = list(labels_dir.glob('*.txt'))
            
            split_analysis = {
                'num_images': len(image_files),
                'num_labels': len(label_files),
                'class_counts': {name: 0 for name in class_names}
            }
            
            # Analyze labels
            total_objects = 0
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x, center_y, width, height = map(float, parts[1:5])
                        
                        # Update class counts
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            split_analysis['class_counts'][class_name] += 1
                            analysis['class_distribution'][class_name] += 1
                        
                        # Update bbox statistics
                        analysis['bbox_stats']['widths'].append(width)
                        analysis['bbox_stats']['heights'].append(height)
                        analysis['bbox_stats']['areas'].append(width * height)
                        analysis['bbox_stats']['aspect_ratios'].append(width / height if height > 0 else 0)
                        analysis['bbox_stats']['center_x'].append(center_x)
                        analysis['bbox_stats']['center_y'].append(center_y)
                        
                        total_objects += 1
            
            split_analysis['total_objects'] = total_objects
            analysis['splits'][split] = split_analysis
        
        # Convert lists to numpy arrays for statistics
        for key in analysis['bbox_stats']:
            if analysis['bbox_stats'][key]:
                arr = np.array(analysis['bbox_stats'][key])
                analysis['bbox_stats'][key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr))
                }
            else:
                analysis['bbox_stats'][key] = {}
        
        return analysis
    
    def convert_labels_to_yolo(
        self,
        labels_dir: str,
        output_dir: str,
        class_mapping: Dict[str, int] = None
    ) -> None:
        """Convert labels to YOLO format."""
        
        if class_mapping is None:
            class_mapping = {"Bottle": 0, "Box": 1}
        
        labels_path = Path(labels_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        label_files = list(labels_path.glob('*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_name = parts[0]
                    x_min, y_min, x_max, y_max = map(float, parts[1:5])
                    
                    # Convert to YOLO format (class_id, center_x, center_y, width, height)
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        
                        # Assuming coordinates are already normalized
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        converted_line = f"{class_id} {center_x} {center_y} {width} {height}\n"
                        converted_lines.append(converted_line)
            
            # Save converted labels
            output_file = output_path / label_file.name
            with open(output_file, 'w') as f:
                f.writelines(converted_lines)
        
        logger.info(f"Converted {len(label_files)} label files to YOLO format")
    
    def validate_dataset(self, dataset_yaml_path: str) -> Dict:
        """Validate dataset integrity."""
        
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_root = Path(dataset_config['path'])
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if required directories exist
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for dir_path in required_dirs:
            full_path = dataset_root / dir_path
            if not full_path.exists():
                validation_results['errors'].append(f"Missing directory: {full_path}")
                validation_results['valid'] = False
        
        if not validation_results['valid']:
            return validation_results
        
        # Check image-label pairs
        for split in ['train', 'val', 'test']:
            images_dir = dataset_root / split / 'images'
            labels_dir = dataset_root / split / 'labels'
            
            if not images_dir.exists():
                continue
            
            image_files = {f.stem for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']}
            label_files = {f.stem for f in labels_dir.glob('*.txt')}
            
            # Check for missing labels
            missing_labels = image_files - label_files
            if missing_labels:
                validation_results['warnings'].append(
                    f"{split} split: {len(missing_labels)} images without labels"
                )
            
            # Check for orphaned labels
            orphaned_labels = label_files - image_files
            if orphaned_labels:
                validation_results['warnings'].append(
                    f"{split} split: {len(orphaned_labels)} labels without images"
                )
            
            validation_results['statistics'][split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'matched_pairs': len(image_files & label_files)
            }
        
        return validation_results


class ImagePreprocessor:
    """Image preprocessing utilities for training and inference."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
    
    def resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image with padding to maintain aspect ratio."""
        if target_size is None:
            target_size = self.target_size
        
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image from [0, 1] to [0, 255] range."""
        return (image * 255).astype(np.uint8)
    
    def apply_augmentation(self, image: np.ndarray, augmentation_params: Dict) -> np.ndarray:
        """Apply basic augmentations to image."""
        # This is a simplified version - YOLO handles most augmentations internally
        
        if augmentation_params.get('horizontal_flip', False) and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if augmentation_params.get('vertical_flip', False) and np.random.random() > 0.5:
            image = cv2.flip(image, 0)
        
        # Brightness adjustment
        brightness_factor = augmentation_params.get('brightness_factor', 0)
        if brightness_factor > 0:
            factor = 1 + np.random.uniform(-brightness_factor, brightness_factor)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image

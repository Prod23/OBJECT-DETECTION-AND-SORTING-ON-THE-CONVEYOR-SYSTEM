"""Unit tests for dataset management functionality."""

import unittest
import tempfile
import shutil
from pathlib import Path
import yaml
import numpy as np
import cv2
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import DatasetManager, ImagePreprocessor


class TestDatasetManager(unittest.TestCase):
    """Test cases for DatasetManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_manager = DatasetManager(self.temp_dir)
        
        # Create test data structure
        self.images_dir = Path(self.temp_dir) / "images"
        self.labels_dir = Path(self.temp_dir) / "labels"
        self.images_dir.mkdir(parents=True)
        self.labels_dir.mkdir(parents=True)
        
        # Create dummy images and labels
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create test images and labels."""
        # Create test images
        for i in range(10):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(self.images_dir / f"test_{i:03d}.jpg"), img)
        
        # Create test labels
        for i in range(10):
            with open(self.labels_dir / f"test_{i:03d}.txt", 'w') as f:
                # Random bounding boxes
                for _ in range(np.random.randint(1, 4)):
                    class_id = np.random.randint(0, 2)
                    x_center = np.random.uniform(0.2, 0.8)
                    y_center = np.random.uniform(0.2, 0.8)
                    width = np.random.uniform(0.1, 0.3)
                    height = np.random.uniform(0.1, 0.3)
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def test_create_yolo_dataset_structure(self):
        """Test YOLO dataset structure creation."""
        dataset_yaml = self.dataset_manager.create_yolo_dataset_structure(
            images_dir=str(self.images_dir),
            labels_dir=str(self.labels_dir),
            class_names=["bottle", "box"]
        )
        
        # Check if dataset.yaml was created
        self.assertTrue(Path(dataset_yaml).exists())
        
        # Check dataset configuration
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertEqual(config['nc'], 2)
        self.assertEqual(config['names'], ["bottle", "box"])
        self.assertIn('train', config)
        self.assertIn('val', config)
        
        # Check if split directories were created
        dataset_root = Path(config['path'])
        self.assertTrue((dataset_root / 'train' / 'images').exists())
        self.assertTrue((dataset_root / 'train' / 'labels').exists())
        self.assertTrue((dataset_root / 'val' / 'images').exists())
        self.assertTrue((dataset_root / 'val' / 'labels').exists())
    
    def test_analyze_dataset(self):
        """Test dataset analysis functionality."""
        # First create dataset
        dataset_yaml = self.dataset_manager.create_yolo_dataset_structure(
            images_dir=str(self.images_dir),
            labels_dir=str(self.labels_dir),
            class_names=["bottle", "box"]
        )
        
        # Analyze dataset
        analysis = self.dataset_manager.analyze_dataset(dataset_yaml)
        
        # Check analysis structure
        self.assertIn('class_names', analysis)
        self.assertIn('splits', analysis)
        self.assertIn('class_distribution', analysis)
        self.assertIn('bbox_stats', analysis)
        
        # Check class names
        self.assertEqual(analysis['class_names'], ["bottle", "box"])
        
        # Check splits
        self.assertIn('train', analysis['splits'])
        self.assertIn('val', analysis['splits'])
        
        # Check that we have some data
        total_images = sum(split['num_images'] for split in analysis['splits'].values())
        self.assertGreater(total_images, 0)
    
    def test_validate_dataset(self):
        """Test dataset validation."""
        # Create dataset
        dataset_yaml = self.dataset_manager.create_yolo_dataset_structure(
            images_dir=str(self.images_dir),
            labels_dir=str(self.labels_dir),
            class_names=["bottle", "box"]
        )
        
        # Validate dataset
        validation_results = self.dataset_manager.validate_dataset(dataset_yaml)
        
        # Check validation structure
        self.assertIn('valid', validation_results)
        self.assertIn('errors', validation_results)
        self.assertIn('warnings', validation_results)
        self.assertIn('statistics', validation_results)
        
        # Should be valid since we created proper test data
        self.assertTrue(validation_results['valid'])
    
    def test_convert_labels_to_yolo(self):
        """Test label conversion to YOLO format."""
        # Create labels in different format
        custom_labels_dir = Path(self.temp_dir) / "custom_labels"
        custom_labels_dir.mkdir()
        
        # Create custom format labels
        for i in range(5):
            with open(custom_labels_dir / f"test_{i:03d}.txt", 'w') as f:
                f.write("Bottle 0.1 0.1 0.3 0.3\n")
                f.write("Box 0.5 0.5 0.7 0.7\n")
        
        output_dir = Path(self.temp_dir) / "converted_labels"
        
        # Convert labels
        self.dataset_manager.convert_labels_to_yolo(
            labels_dir=str(custom_labels_dir),
            output_dir=str(output_dir),
            class_mapping={"Bottle": 0, "Box": 1}
        )
        
        # Check if converted labels exist
        self.assertTrue(output_dir.exists())
        converted_files = list(output_dir.glob("*.txt"))
        self.assertEqual(len(converted_files), 5)
        
        # Check format of converted labels
        with open(converted_files[0], 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                self.assertEqual(len(parts), 5)
                # First part should be class ID (0 or 1)
                self.assertIn(int(parts[0]), [0, 1])


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor(target_size=(640, 640))
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_resize_with_padding(self):
        """Test image resizing with padding."""
        resized = self.preprocessor.resize_with_padding(self.test_image)
        
        # Check output size
        self.assertEqual(resized.shape[:2], (640, 640))
        self.assertEqual(resized.shape[2], 3)
        
        # Check data type
        self.assertEqual(resized.dtype, np.uint8)
    
    def test_normalize_image(self):
        """Test image normalization."""
        normalized = self.preprocessor.normalize_image(self.test_image)
        
        # Check value range
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
        
        # Check data type
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_denormalize_image(self):
        """Test image denormalization."""
        # First normalize, then denormalize
        normalized = self.preprocessor.normalize_image(self.test_image)
        denormalized = self.preprocessor.denormalize_image(normalized)
        
        # Check data type
        self.assertEqual(denormalized.dtype, np.uint8)
        
        # Check that we get back similar values (allowing for rounding errors)
        diff = np.abs(self.test_image.astype(float) - denormalized.astype(float))
        self.assertLess(diff.mean(), 1.0)  # Average difference should be small
    
    def test_apply_augmentation(self):
        """Test basic augmentation application."""
        aug_params = {
            'horizontal_flip': True,
            'brightness_factor': 0.1
        }
        
        augmented = self.preprocessor.apply_augmentation(self.test_image, aug_params)
        
        # Check that output has same shape
        self.assertEqual(augmented.shape, self.test_image.shape)
        self.assertEqual(augmented.dtype, self.test_image.dtype)


if __name__ == '__main__':
    unittest.main()

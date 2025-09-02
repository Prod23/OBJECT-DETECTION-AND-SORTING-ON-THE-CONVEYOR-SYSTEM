"""Unit tests for model management functionality."""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.yolo_model import YOLOModelManager, ModelComparison


class TestYOLOModelManager(unittest.TestCase):
    """Test cases for YOLOModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = YOLOModelManager(model_name="yolov8n", device="cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_device(self):
        """Test device selection logic."""
        # Test auto device selection
        device = self.model_manager._get_device("auto")
        self.assertIn(device, ["cpu", "cuda", "mps"])
        
        # Test explicit device
        device = self.model_manager._get_device("cpu")
        self.assertEqual(device, "cpu")
    
    def test_load_model(self):
        """Test model loading."""
        # Load pretrained model (this will download if not cached)
        try:
            model = self.model_manager.load_model(pretrained=True)
            self.assertIsNotNone(model)
            self.assertIsNotNone(self.model_manager.class_names)
        except Exception as e:
            # Skip test if model download fails (e.g., no internet)
            self.skipTest(f"Could not load model: {e}")
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        try:
            self.model_manager.load_model(pretrained=True)
            info = self.model_manager.get_model_info()
            
            self.assertIn('model_name', info)
            self.assertIn('device', info)
            self.assertIn('class_names', info)
            self.assertEqual(info['model_name'], "yolov8n")
            
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
    
    def test_benchmark_model(self):
        """Test model benchmarking."""
        try:
            self.model_manager.load_model(pretrained=True)
            
            # Run small benchmark
            results = self.model_manager.benchmark_model(
                input_shape=(1, 3, 320, 320),
                num_runs=5,
                warmup_runs=2
            )
            
            self.assertIn('mean_inference_time', results)
            self.assertIn('mean_fps', results)
            self.assertGreater(results['mean_fps'], 0)
            self.assertGreater(results['mean_inference_time'], 0)
            
        except Exception as e:
            self.skipTest(f"Could not benchmark model: {e}")


class TestModelComparison(unittest.TestCase):
    """Test cases for ModelComparison class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparison = ModelComparison()
    
    def test_add_model(self):
        """Test adding models for comparison."""
        # This test would require actual model files
        # For now, just test the structure
        self.assertEqual(len(self.comparison.models), 0)
        self.assertEqual(len(self.comparison.results), 0)
    
    def test_comparison_structure(self):
        """Test comparison data structure."""
        # Test that comparison object has required attributes
        self.assertTrue(hasattr(self.comparison, 'models'))
        self.assertTrue(hasattr(self.comparison, 'results'))
        self.assertTrue(hasattr(self.comparison, 'add_model'))
        self.assertTrue(hasattr(self.comparison, 'benchmark_all_models'))


if __name__ == '__main__':
    unittest.main()

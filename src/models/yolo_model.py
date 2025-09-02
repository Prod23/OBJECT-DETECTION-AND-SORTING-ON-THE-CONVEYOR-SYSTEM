"""YOLO model management and utilities."""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


class YOLOModelManager:
    """Manages YOLO model creation, loading, and operations."""
    
    def __init__(self, model_name: str = "yolov8m", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.class_names = None
        
    def _get_device(self, device: str) -> str:
        """Get appropriate device for model."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self, weights_path: Optional[str] = None, pretrained: bool = True) -> YOLO:
        """Load YOLO model with specified weights."""
        try:
            if weights_path and Path(weights_path).exists():
                logger.info(f"Loading model from weights: {weights_path}")
                self.model = YOLO(weights_path)
            elif pretrained:
                logger.info(f"Loading pretrained {self.model_name} model")
                self.model = YOLO(f"{self.model_name}.pt")
            else:
                logger.info(f"Loading {self.model_name} model without pretrained weights")
                self.model = YOLO(f"{self.model_name}.yaml")
            
            # Move to device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # Extract class names if available
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'class_names': self.class_names,
            'num_classes': len(self.class_names) if self.class_names else None,
        }
        
        # Try to get model parameters
        try:
            if hasattr(self.model.model, 'parameters'):
                total_params = sum(p.numel() for p in self.model.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
                })
        except Exception as e:
            logger.warning(f"Could not get model parameters: {e}")
        
        return info
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        opset_version: int = 11,
        dynamic_axes: bool = True
    ) -> str:
        """Export model to ONNX format."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export using ultralytics built-in method
            if hasattr(self.model, 'export'):
                exported_path = self.model.export(
                    format='onnx',
                    imgsz=input_shape[2:],
                    opset=opset_version,
                    dynamic=dynamic_axes
                )
                
                # Move to desired location if different
                if str(exported_path) != str(output_path):
                    import shutil
                    shutil.move(str(exported_path), str(output_path))
                
                logger.info(f"Model exported to ONNX: {output_path}")
                return str(output_path)
            else:
                raise ValueError("Model does not support ONNX export")
                
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            raise
    
    def export_to_tensorrt(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        precision: str = "fp16"
    ) -> str:
        """Export model to TensorRT format."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export using ultralytics built-in method
            if hasattr(self.model, 'export'):
                exported_path = self.model.export(
                    format='engine',
                    imgsz=input_shape[2:],
                    half=(precision == "fp16")
                )
                
                # Move to desired location if different
                if str(exported_path) != str(output_path):
                    import shutil
                    shutil.move(str(exported_path), str(output_path))
                
                logger.info(f"Model exported to TensorRT: {output_path}")
                return str(output_path)
            else:
                raise ValueError("Model does not support TensorRT export")
                
        except Exception as e:
            logger.error(f"Error exporting to TensorRT: {e}")
            raise
    
    def benchmark_model(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark model inference speed."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup iterations...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
        
        # Benchmark runs
        logger.info(f"Running {num_runs} benchmark iterations...")
        import time
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        benchmark_results = {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'median_inference_time': float(np.median(times)),
            'mean_fps': float(1.0 / np.mean(times)),
            'device': self.device,
            'input_shape': input_shape
        }
        
        logger.info(f"Benchmark results: {benchmark_results['mean_fps']:.2f} FPS")
        return benchmark_results


class ONNXInferenceEngine:
    """ONNX inference engine for optimized deployment."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model."""
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX model loaded: {self.model_path}")
            logger.info(f"Input: {self.input_name}")
            logger.info(f"Outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on input data."""
        if self.session is None:
            raise ValueError("Model not loaded")
        
        try:
            # Ensure input is float32
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            return outputs
            
        except Exception as e:
            logger.error(f"Error during ONNX inference: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ONNX model information."""
        if self.session is None:
            raise ValueError("Model not loaded")
        
        input_info = self.session.get_inputs()[0]
        output_info = [output for output in self.session.get_outputs()]
        
        return {
            'model_path': self.model_path,
            'providers': self.providers,
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'outputs': [
                {
                    'name': output.name,
                    'shape': output.shape,
                    'type': output.type
                }
                for output in output_info
            ]
        }


class ModelComparison:
    """Utility for comparing different YOLO models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model_path: str, device: str = "auto"):
        """Add a model for comparison."""
        try:
            manager = YOLOModelManager(device=device)
            manager.load_model(model_path)
            self.models[name] = manager
            logger.info(f"Added model for comparison: {name}")
        except Exception as e:
            logger.error(f"Error adding model {name}: {e}")
    
    def benchmark_all_models(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark all added models."""
        results = {}
        
        for name, manager in self.models.items():
            logger.info(f"Benchmarking model: {name}")
            try:
                benchmark_results = manager.benchmark_model(input_shape, num_runs)
                results[name] = benchmark_results
            except Exception as e:
                logger.error(f"Error benchmarking model {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """Get comparison summary as DataFrame."""
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmark_all_models() first.")
        
        import pandas as pd
        
        summary_data = []
        for name, results in self.results.items():
            if 'error' not in results:
                model_info = self.models[name].get_model_info()
                summary_data.append({
                    'Model': name,
                    'Mean FPS': results.get('mean_fps', 0),
                    'Mean Inference Time (ms)': results.get('mean_inference_time', 0) * 1000,
                    'Total Parameters': model_info.get('total_parameters', 0),
                    'Model Size (MB)': model_info.get('model_size_mb', 0),
                    'Device': results.get('device', 'unknown')
                })
        
        return pd.DataFrame(summary_data)
    
    def export_comparison_report(self, output_path: str):
        """Export comparison report to file."""
        summary_df = self.get_comparison_summary()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_path.with_suffix('.csv')
        summary_df.to_csv(csv_path, index=False)
        
        # Save detailed results as JSON
        import json
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Comparison report saved: {csv_path}, {json_path}")
        return str(csv_path), str(json_path)

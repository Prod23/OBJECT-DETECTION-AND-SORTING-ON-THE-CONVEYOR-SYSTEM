"""Training pipeline for YOLO object detection models."""

import os
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
from datetime import datetime
import json
import yaml

from ultralytics import YOLO
from src.utils.config import ConfigManager
from src.utils.logging import setup_logger, MetricsLogger
from src.utils.visualization import MetricsVisualizer
from src.models.yolo_model import YOLOModelManager

logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Main training class for YOLO models."""
    
    def __init__(self, config_path: str = "config/train.yaml"):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_training_config(config_path)
        
        # Setup logging
        self.experiment_name = self.config.get('logging', {}).get('experiment_name', 'yolo_training')
        self.logger = setup_logger(
            name=__name__,
            log_level=self.config.get('logging', {}).get('log_level', 'INFO'),
            log_file=f"{self.experiment_name}.log"
        )
        
        # Initialize components
        self.model_manager = None
        self.metrics_logger = None
        self.visualizer = MetricsVisualizer()
        
        # Training state
        self.model = None
        self.best_metrics = {}
        self.training_history = []
        
        # Setup MLflow if enabled
        if self.config.get('logging', {}).get('mlflow', {}).get('enabled', False):
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        mlflow_config = self.config.get('logging', {}).get('mlflow', {})
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        experiment_name = mlflow_config.get('experiment_name', 'conveyor-object-detection')
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            self.logger.warning(f"Could not setup MLflow: {e}")
    
    def prepare_training(self):
        """Prepare training environment and model."""
        # Set random seeds for reproducibility
        if self.config.get('seed'):
            torch.manual_seed(self.config['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config['seed'])
        
        # Initialize model manager
        model_config = self.config.get('model', {})
        device = self.config.get('training', {}).get('device', 'auto')
        
        self.model_manager = YOLOModelManager(
            model_name=model_config.get('name', 'yolov8m'),
            device=device
        )
        
        # Load model
        self.model = self.model_manager.load_model(
            pretrained=model_config.get('pretrained', True)
        )
        
        # Initialize metrics logger
        save_dir = self.config.get('logging', {}).get('save_dir', 'runs/train')
        self.metrics_logger = MetricsLogger(
            log_dir=save_dir,
            experiment_name=self.experiment_name
        )
        
        self.logger.info("Training preparation completed")
    
    def train(self, resume: Optional[str] = None) -> Dict[str, Any]:
        """Execute training process."""
        if self.model is None:
            self.prepare_training()
        
        training_config = self.config.get('training', {})
        data_config = self.config.get('data', {})
        
        # Prepare training arguments
        train_args = {
            'data': data_config.get('dataset_path', 'data/dataset.yaml'),
            'epochs': training_config.get('epochs', 300),
            'batch': training_config.get('batch_size', 16),
            'imgsz': training_config.get('imgsz', 640),
            'device': training_config.get('device', 'auto'),
            'workers': training_config.get('workers', 8),
            'patience': training_config.get('patience', 50),
            'save_period': training_config.get('save_period', 10),
            'project': self.config.get('logging', {}).get('save_dir', 'runs/train'),
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': self.config.get('model', {}).get('pretrained', True),
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'lr0': training_config.get('lr0', 0.01),
            'lrf': training_config.get('lrf', 0.01),
            'momentum': training_config.get('momentum', 0.937),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'warmup_epochs': training_config.get('warmup_epochs', 3),
            'warmup_momentum': training_config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': training_config.get('warmup_bias_lr', 0.1),
            'box': training_config.get('box', 7.5),
            'cls': training_config.get('cls', 0.5),
            'dfl': training_config.get('dfl', 1.5),
            'seed': self.config.get('seed', 42),
            'deterministic': self.config.get('deterministic', True)
        }
        
        # Add validation settings
        val_config = self.config.get('validation', {})
        train_args.update({
            'val': val_config.get('val_period', 1) > 0,
            'save_json': val_config.get('save_json', True),
            'save_hybrid': val_config.get('save_hybrid', False),
            'conf': val_config.get('conf_thres', 0.001),
            'iou': val_config.get('iou_thres', 0.6),
            'max_det': val_config.get('max_det', 300),
            'half': val_config.get('half', False),
            'dnn': val_config.get('dnn', False)
        })
        
        # Add augmentation settings
        aug_config = data_config.get('augmentation', {})
        if aug_config.get('enabled', True):
            train_args.update({
                'mosaic': aug_config.get('mosaic', 1.0),
                'mixup': aug_config.get('mixup', 0.1),
                'copy_paste': aug_config.get('copy_paste', 0.3),
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'hsv_v': aug_config.get('hsv_v', 0.4),
                'degrees': aug_config.get('degrees', 10.0),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.9),
                'shear': aug_config.get('shear', 2.0),
                'perspective': aug_config.get('perspective', 0.0),
                'flipud': aug_config.get('flipud', 0.0),
                'fliplr': aug_config.get('fliplr', 0.5)
            })
        
        # Resume training if specified
        if resume:
            train_args['resume'] = resume
        
        # Start MLflow run if enabled
        mlflow_enabled = self.config.get('logging', {}).get('mlflow', {}).get('enabled', False)
        if mlflow_enabled:
            mlflow.start_run(run_name=self.experiment_name)
            # Log configuration
            mlflow.log_params(self._flatten_config(self.config))
        
        try:
            self.logger.info("Starting training...")
            self.logger.info(f"Training arguments: {train_args}")
            
            # Train the model
            results = self.model.train(**train_args)
            
            # Process results
            training_results = self._process_training_results(results)
            
            # Log final metrics
            if mlflow_enabled:
                self._log_final_metrics_to_mlflow(training_results)
            
            self.logger.info("Training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            if mlflow_enabled and mlflow.active_run():
                mlflow.end_run()
    
    def _process_training_results(self, results) -> Dict[str, Any]:
        """Process and organize training results."""
        # Extract metrics from results
        metrics_data = {}
        
        if hasattr(results, 'results_dict'):
            metrics_data = results.results_dict
        
        # Get best metrics
        best_fitness = getattr(results, 'best_fitness', 0)
        best_epoch = getattr(results, 'best_epoch', 0)
        
        # Organize results
        training_results = {
            'best_epoch': best_epoch,
            'best_fitness': best_fitness,
            'metrics': metrics_data,
            'model_path': getattr(results, 'save_dir', None),
            'weights_path': {
                'best': str(Path(results.save_dir) / 'weights' / 'best.pt') if hasattr(results, 'save_dir') else None,
                'last': str(Path(results.save_dir) / 'weights' / 'last.pt') if hasattr(results, 'save_dir') else None
            }
        }
        
        # Save training summary
        self._save_training_summary(training_results)
        
        return training_results
    
    def _save_training_summary(self, results: Dict[str, Any]):
        """Save training summary to file."""
        summary_path = Path(self.config.get('logging', {}).get('save_dir', 'runs/train')) / f"{self.experiment_name}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': results,
            'model_info': self.model_manager.get_model_info() if self.model_manager else {}
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_path}")
    
    def _log_final_metrics_to_mlflow(self, results: Dict[str, Any]):
        """Log final metrics to MLflow."""
        try:
            # Log best metrics
            if 'metrics' in results:
                for key, value in results['metrics'].items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"final_{key}", value)
            
            # Log model artifacts
            if results.get('weights_path', {}).get('best'):
                mlflow.log_artifact(results['weights_path']['best'], "model")
            
            # Log model info
            if self.model_manager:
                model_info = self.model_manager.get_model_info()
                for key, value in model_info.items():
                    if isinstance(value, (int, float, str)):
                        mlflow.log_param(f"model_{key}", value)
        
        except Exception as e:
            self.logger.warning(f"Could not log metrics to MLflow: {e}")
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration for MLflow logging."""
        flattened = {}
        for key, value in config.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, new_key))
            else:
                flattened[new_key] = value
        return flattened
    
    def evaluate(self, weights_path: str, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate trained model."""
        if data_path is None:
            data_path = self.config.get('data', {}).get('dataset_path', 'data/dataset.yaml')
        
        # Load model for evaluation
        eval_model = YOLO(weights_path)
        
        # Run validation
        val_config = self.config.get('validation', {})
        val_results = eval_model.val(
            data=data_path,
            imgsz=self.config.get('training', {}).get('imgsz', 640),
            batch=self.config.get('training', {}).get('batch_size', 16),
            conf=val_config.get('conf_thres', 0.001),
            iou=val_config.get('iou_thres', 0.6),
            max_det=val_config.get('max_det', 300),
            half=val_config.get('half', False),
            device=self.config.get('training', {}).get('device', 'auto'),
            dnn=val_config.get('dnn', False),
            plots=True,
            save_json=val_config.get('save_json', True),
            save_hybrid=val_config.get('save_hybrid', False)
        )
        
        # Process evaluation results
        eval_metrics = {
            'mAP50': val_results.box.map50 if hasattr(val_results, 'box') else 0,
            'mAP50_95': val_results.box.map if hasattr(val_results, 'box') else 0,
            'precision': val_results.box.mp if hasattr(val_results, 'box') else 0,
            'recall': val_results.box.mr if hasattr(val_results, 'box') else 0,
            'fitness': val_results.fitness if hasattr(val_results, 'fitness') else 0
        }
        
        self.logger.info(f"Evaluation completed: {eval_metrics}")
        return eval_metrics
    
    def export_model(
        self,
        weights_path: str,
        export_formats: List[str] = None,
        output_dir: str = "exports"
    ) -> Dict[str, str]:
        """Export trained model to different formats."""
        if export_formats is None:
            export_formats = ['onnx']
        
        model = YOLO(weights_path)
        exported_paths = {}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for format_name in export_formats:
            try:
                self.logger.info(f"Exporting model to {format_name}...")
                
                export_path = model.export(
                    format=format_name,
                    imgsz=self.config.get('training', {}).get('imgsz', 640),
                    half=format_name in ['engine', 'tflite'],  # Use half precision for TensorRT and TFLite
                    int8=format_name == 'tflite',  # Use int8 for TFLite
                    dynamic=format_name == 'onnx',  # Dynamic axes for ONNX
                    simplify=format_name == 'onnx'  # Simplify ONNX model
                )
                
                exported_paths[format_name] = str(export_path)
                self.logger.info(f"Model exported to {format_name}: {export_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to export to {format_name}: {e}")
                exported_paths[format_name] = None
        
        return exported_paths


class TrainingCallback:
    """Custom callback for training monitoring."""
    
    def __init__(self, metrics_logger: MetricsLogger):
        self.metrics_logger = metrics_logger
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        if hasattr(trainer, 'metrics'):
            epoch = trainer.epoch
            metrics = trainer.metrics
            
            # Log metrics
            self.metrics_logger.log_epoch_metrics(epoch, metrics)
    
    def on_val_end(self, trainer):
        """Called at the end of validation."""
        if hasattr(trainer, 'metrics'):
            # Additional validation logging can be added here
            pass

"""Configuration management utilities for the conveyor detection system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "yolov8m"
    pretrained: bool = True
    num_classes: int = 2
    class_names: list = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["bottle", "box"]


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 300
    batch_size: int = 16
    imgsz: int = 640
    patience: int = 50
    save_period: int = 10
    workers: int = 8
    device: str = "auto"
    optimizer: str = "AdamW"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    weights_path: str = "runs/train/exp/weights/best.pt"
    device: str = "auto"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    imgsz: int = 640
    save_results: bool = True
    show_results: bool = True


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = self.config_dir / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_file}: {e}")
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = self.config_dir / config_path
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            raise ValueError(f"Error saving configuration to {config_file}: {e}")
    
    def load_training_config(self, config_path: str = "train.yaml") -> Dict[str, Any]:
        """Load training configuration."""
        return self.load_config(config_path)
    
    def load_inference_config(self, config_path: str = "inference.yaml") -> Dict[str, Any]:
        """Load inference configuration."""
        return self.load_config(config_path)
    
    def validate_config(self, config: Dict[str, Any], required_keys: list) -> bool:
        """Validate that required keys exist in configuration."""
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, with override taking precedence."""
        merged = base_config.copy()
        
        def deep_update(base_dict, override_dict):
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(merged, override_config)
        return merged


def setup_environment_variables(config: Dict[str, Any]) -> None:
    """Set up environment variables from configuration."""
    env_vars = config.get('environment', {})
    for key, value in env_vars.items():
        os.environ[key] = str(value)


def get_device_config(device: str = "auto") -> str:
    """Get appropriate device configuration."""
    import torch
    
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def create_experiment_name(base_name: str, config: Dict[str, Any]) -> str:
    """Create a unique experiment name based on configuration."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('model', {}).get('name', 'unknown')
    batch_size = config.get('training', {}).get('batch_size', 'unknown')
    
    return f"{base_name}_{model_name}_bs{batch_size}_{timestamp}"

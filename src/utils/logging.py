"""Logging utilities for the conveyor detection system."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json
import csv


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """Set up a logger with both console and file handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path / log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training and validation metrics."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Initialize log files
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"
        
        # Initialize CSV file with headers
        self._init_csv_file()
        
        # Metrics storage
        self.metrics_history = []
    
    def _init_csv_file(self):
        """Initialize CSV file with headers."""
        headers = [
            'epoch', 'timestamp', 'train_loss', 'val_loss',
            'train_box_loss', 'train_cls_loss', 'train_dfl_loss',
            'val_box_loss', 'val_cls_loss', 'val_dfl_loss',
            'precision', 'recall', 'mAP50', 'mAP50_95',
            'learning_rate', 'gpu_memory'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_epoch_metrics(self, epoch: int, metrics: dict):
        """Log metrics for a single epoch."""
        timestamp = datetime.now().isoformat()
        
        # Add metadata
        metrics_entry = {
            'epoch': epoch,
            'timestamp': timestamp,
            **metrics
        }
        
        # Store in memory
        self.metrics_history.append(metrics_entry)
        
        # Save to JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                epoch, timestamp,
                metrics.get('train_loss', ''),
                metrics.get('val_loss', ''),
                metrics.get('train_box_loss', ''),
                metrics.get('train_cls_loss', ''),
                metrics.get('train_dfl_loss', ''),
                metrics.get('val_box_loss', ''),
                metrics.get('val_cls_loss', ''),
                metrics.get('val_dfl_loss', ''),
                metrics.get('precision', ''),
                metrics.get('recall', ''),
                metrics.get('mAP50', ''),
                metrics.get('mAP50_95', ''),
                metrics.get('learning_rate', ''),
                metrics.get('gpu_memory', '')
            ]
            writer.writerow(row)
    
    def get_best_metrics(self, metric_name: str = 'mAP50') -> dict:
        """Get the best metrics based on a specific metric."""
        if not self.metrics_history:
            return {}
        
        best_entry = max(
            self.metrics_history,
            key=lambda x: x.get(metric_name, 0)
        )
        return best_entry
    
    def get_latest_metrics(self) -> dict:
        """Get the latest metrics."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]


class InferenceLogger:
    """Logger for inference results and performance."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "inference"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Initialize log files
        self.detections_file = self.log_dir / f"{experiment_name}_detections.json"
        self.performance_file = self.log_dir / f"{experiment_name}_performance.json"
        
        # Storage
        self.detections = []
        self.performance_metrics = []
    
    def log_detection(self, frame_id: int, detections: list, metadata: dict = None):
        """Log detection results for a frame."""
        detection_entry = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'metadata': metadata or {}
        }
        
        self.detections.append(detection_entry)
        
        # Save periodically (every 100 frames)
        if len(self.detections) % 100 == 0:
            self._save_detections()
    
    def log_performance(self, fps: float, processing_time: float, frame_count: int):
        """Log performance metrics."""
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'fps': fps,
            'processing_time': processing_time,
            'frame_count': frame_count
        }
        
        self.performance_metrics.append(performance_entry)
        
        # Keep only last 1000 entries
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def _save_detections(self):
        """Save detections to file."""
        with open(self.detections_file, 'w') as f:
            json.dump(self.detections, f, indent=2)
    
    def save_all(self):
        """Save all logged data."""
        self._save_detections()
        
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
    
    def get_average_performance(self, window: int = 100) -> dict:
        """Get average performance metrics over a window."""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = self.performance_metrics[-window:]
        
        avg_fps = sum(m['fps'] for m in recent_metrics) / len(recent_metrics)
        avg_processing_time = sum(m['processing_time'] for m in recent_metrics) / len(recent_metrics)
        
        return {
            'average_fps': avg_fps,
            'average_processing_time': avg_processing_time,
            'window_size': len(recent_metrics)
        }

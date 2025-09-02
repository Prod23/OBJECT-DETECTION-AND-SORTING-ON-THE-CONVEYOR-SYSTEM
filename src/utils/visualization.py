"""Visualization utilities for the conveyor detection system."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class DetectionVisualizer:
    """Visualizer for object detection results."""
    
    def __init__(self, class_names: List[str], colors: Optional[List[Tuple[int, int, int]]] = None):
        self.class_names = class_names
        self.colors = colors or self._generate_colors(len(class_names))
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = plt.cm.Set1(hue)[:3]
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        conf_threshold: float = 0.5,
        show_labels: bool = True,
        show_conf: bool = True,
        line_thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        img_copy = image.copy()
        
        for det in detections:
            if det['confidence'] < conf_threshold:
                continue
            
            # Extract detection info
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_id = int(det['class_id'])
            confidence = det['confidence']
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label text
            if show_labels and show_conf:
                label = f"{self.class_names[class_id]}: {confidence:.2f}"
            elif show_labels:
                label = self.class_names[class_id]
            elif show_conf:
                label = f"{confidence:.2f}"
            else:
                continue
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                img_copy,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_copy,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return img_copy
    
    def create_detection_grid(
        self,
        images: List[np.ndarray],
        detections_list: List[List[Dict[str, Any]]],
        titles: Optional[List[str]] = None,
        grid_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """Create a grid of images with detections."""
        n_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        # Get image dimensions
        h, w = images[0].shape[:2]
        
        # Create grid
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        for i, (img, dets) in enumerate(zip(images, detections_list)):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            # Draw detections
            img_with_dets = self.draw_detections(img, dets)
            
            # Add title if provided
            if titles and i < len(titles):
                cv2.putText(
                    img_with_dets,
                    titles[i],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
            
            # Place in grid
            y_start, y_end = row * h, (row + 1) * h
            x_start, x_end = col * w, (col + 1) * w
            grid[y_start:y_end, x_start:x_end] = img_with_dets
        
        return grid


class MetricsVisualizer:
    """Visualizer for training and validation metrics."""
    
    def __init__(self, style: str = "darkgrid"):
        plt.style.use('default')
        sns.set_style(style)
    
    def plot_training_curves(
        self,
        metrics_data: Dict[str, List[float]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Loss curves
        epochs = range(1, len(metrics_data.get('train_loss', [])) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, metrics_data.get('train_loss', []), 'b-', label='Train')
        axes[0, 0].plot(epochs, metrics_data.get('val_loss', []), 'r-', label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Box loss
        axes[0, 1].plot(epochs, metrics_data.get('train_box_loss', []), 'b-', label='Train')
        axes[0, 1].plot(epochs, metrics_data.get('val_box_loss', []), 'r-', label='Validation')
        axes[0, 1].set_title('Box Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Class loss
        axes[0, 2].plot(epochs, metrics_data.get('train_cls_loss', []), 'b-', label='Train')
        axes[0, 2].plot(epochs, metrics_data.get('val_cls_loss', []), 'r-', label='Validation')
        axes[0, 2].set_title('Classification Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Precision and Recall
        axes[1, 0].plot(epochs, metrics_data.get('precision', []), 'g-', label='Precision')
        axes[1, 0].plot(epochs, metrics_data.get('recall', []), 'orange', label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # mAP scores
        axes[1, 1].plot(epochs, metrics_data.get('mAP50', []), 'purple', label='mAP@0.5')
        axes[1, 1].plot(epochs, metrics_data.get('mAP50_95', []), 'brown', label='mAP@0.5:0.95')
        axes[1, 1].set_title('Mean Average Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        axes[1, 2].plot(epochs, metrics_data.get('learning_rate', []), 'red')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(
        self,
        metrics_data: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive dashboard with plotly."""
        epochs = list(range(1, len(metrics_data.get('train_loss', [])) + 1))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Loss Curves', 'Box Loss', 'Classification Loss',
                          'Precision & Recall', 'mAP Scores', 'Learning Rate'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('train_loss', []),
                      mode='lines', name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('val_loss', []),
                      mode='lines', name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Box loss
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('train_box_loss', []),
                      mode='lines', name='Train Box Loss', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('val_box_loss', []),
                      mode='lines', name='Val Box Loss', line=dict(color='red')),
            row=1, col=2
        )
        
        # Classification loss
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('train_cls_loss', []),
                      mode='lines', name='Train Cls Loss', line=dict(color='blue')),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('val_cls_loss', []),
                      mode='lines', name='Val Cls Loss', line=dict(color='red')),
            row=1, col=3
        )
        
        # Precision and Recall
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('precision', []),
                      mode='lines', name='Precision', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('recall', []),
                      mode='lines', name='Recall', line=dict(color='orange')),
            row=2, col=1, secondary_y=True
        )
        
        # mAP scores
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('mAP50', []),
                      mode='lines', name='mAP@0.5', line=dict(color='purple')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('mAP50_95', []),
                      mode='lines', name='mAP@0.5:0.95', line=dict(color='brown')),
            row=2, col=2
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics_data.get('learning_rate', []),
                      mode='lines', name='Learning Rate', line=dict(color='red')),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Training Metrics Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class DataAnalysisVisualizer:
    """Visualizer for dataset analysis."""
    
    def __init__(self):
        sns.set_style("whitegrid")
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot class distribution."""
        plt.figure(figsize=figsize)
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Create bar plot
        bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Number of Instances', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_bbox_statistics(
        self,
        bbox_data: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Plot bounding box statistics."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Bounding Box Statistics', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(bbox_data['width'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (normalized)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Height distribution
        axes[0, 1].hist(bbox_data['height'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (normalized)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Aspect ratio distribution
        aspect_ratios = bbox_data['width'] / bbox_data['height']
        axes[0, 2].hist(aspect_ratios, bins=50, alpha=0.7, color='red')
        axes[0, 2].set_title('Aspect Ratio Distribution')
        axes[0, 2].set_xlabel('Width/Height Ratio')
        axes[0, 2].set_ylabel('Frequency')
        
        # Area distribution
        areas = bbox_data['width'] * bbox_data['height']
        axes[1, 0].hist(areas, bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Area Distribution')
        axes[1, 0].set_xlabel('Area (normalized)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Center X distribution
        axes[1, 1].hist(bbox_data['center_x'], bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('Center X Distribution')
        axes[1, 1].set_xlabel('Center X (normalized)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Center Y distribution
        axes[1, 2].hist(bbox_data['center_y'], bins=50, alpha=0.7, color='brown')
        axes[1, 2].set_title('Center Y Distribution')
        axes[1, 2].set_xlabel('Center Y (normalized)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

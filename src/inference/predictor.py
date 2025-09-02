"""Real-time inference pipeline for conveyor object detection."""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import deque
import threading
import queue

from ultralytics import YOLO
from src.utils.config import ConfigManager
from src.utils.logging import setup_logger, InferenceLogger
from src.utils.visualization import DetectionVisualizer
from src.models.yolo_model import YOLOModelManager, ONNXInferenceEngine

logger = logging.getLogger(__name__)


class ObjectTracker:
    """Simple object tracker for conveyor belt objects."""
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid: Tuple[float, float]) -> int:
        """Register a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
    
    def deregister(self, object_id: int):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Update tracker with new detections."""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Extract centroids from detections
        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            tracked_objects = {}
            for i, centroid in enumerate(input_centroids):
                object_id = self.register(centroid)
                tracked_objects[object_id] = {
                    **detections[i],
                    'object_id': object_id,
                    'centroid': centroid
                }
            return tracked_objects
        
        # Compute distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        distances = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
            axis=2
        )
        
        # Find minimum distances
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        
        used_row_indices = set()
        used_col_indices = set()
        tracked_objects = {}
        
        # Update existing objects
        for (row, col) in zip(rows, cols):
            if row in used_row_indices or col in used_col_indices:
                continue
            
            if distances[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            
            tracked_objects[object_id] = {
                **detections[col],
                'object_id': object_id,
                'centroid': input_centroids[col]
            }
            
            used_row_indices.add(row)
            used_col_indices.add(col)
        
        # Handle unmatched detections and objects
        unused_row_indices = set(range(0, distances.shape[0])) - used_row_indices
        unused_col_indices = set(range(0, distances.shape[1])) - used_col_indices
        
        # Mark unmatched objects as disappeared
        for row in unused_row_indices:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # Register new objects
        for col in unused_col_indices:
            object_id = self.register(input_centroids[col])
            tracked_objects[object_id] = {
                **detections[col],
                'object_id': object_id,
                'centroid': input_centroids[col]
            }
        
        return tracked_objects


class ConveyorInferenceEngine:
    """Main inference engine for conveyor object detection."""
    
    def __init__(self, config_path: str = "config/inference.yaml"):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_inference_config(config_path)
        
        # Setup logging
        self.logger = setup_logger(
            name=__name__,
            log_level=self.config.get('logging', {}).get('log_level', 'INFO')
        )
        
        # Initialize components
        self.model = None
        self.model_manager = None
        self.onnx_engine = None
        self.tracker = None
        self.visualizer = None
        self.inference_logger = None
        
        # Performance tracking
        self.fps_counter = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Threading for real-time processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_running = False
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components."""
        # Initialize model
        model_config = self.config.get('model', {})
        weights_path = model_config.get('weights_path', 'runs/train/exp/weights/best.pt')
        device = model_config.get('device', 'auto')
        
        if weights_path.endswith('.onnx'):
            # Use ONNX engine for optimized inference
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
            self.onnx_engine = ONNXInferenceEngine(weights_path, providers)
            self.logger.info("Initialized ONNX inference engine")
        else:
            # Use YOLO model
            self.model_manager = YOLOModelManager(device=device)
            self.model = self.model_manager.load_model(weights_path, pretrained=False)
            self.logger.info("Initialized YOLO model")
        
        # Initialize tracker
        realtime_config = self.config.get('realtime', {})
        postprocess_config = self.config.get('postprocess', {})
        
        if postprocess_config.get('track_objects', True):
            self.tracker = ObjectTracker(
                max_disappeared=postprocess_config.get('max_disappeared', 10),
                max_distance=postprocess_config.get('centroid_distance_threshold', 50)
            )
        
        # Initialize visualizer
        class_names = ["bottle", "box"]  # Default class names
        if self.model and hasattr(self.model, 'names'):
            class_names = list(self.model.names.values())
        
        self.visualizer = DetectionVisualizer(class_names)
        
        # Initialize inference logger
        output_config = self.config.get('output', {})
        if output_config.get('save_results', True):
            self.inference_logger = InferenceLogger(
                log_dir=output_config.get('save_dir', 'runs/inference'),
                experiment_name=f"inference_{int(time.time())}"
            )
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference."""
        input_config = self.config.get('input', {})
        imgsz = input_config.get('imgsz', 640)
        
        # Resize frame
        if isinstance(imgsz, int):
            target_size = (imgsz, imgsz)
        else:
            target_size = tuple(imgsz)
        
        # Resize while maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to target size
        padded = np.full((*target_size, 3), 114, dtype=np.uint8)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def postprocess_detections(self, results, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess model outputs to detection format."""
        detections = []
        detection_config = self.config.get('detection', {})
        conf_threshold = detection_config.get('conf_threshold', 0.5)
        
        if self.onnx_engine:
            # Handle ONNX outputs
            # This is a simplified version - actual implementation depends on ONNX model output format
            pass
        else:
            # Handle YOLO results
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        if conf >= conf_threshold:
                            # Get box coordinates
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            class_id = int(boxes.cls[i])
                            
                            # Scale coordinates back to original image size
                            h, w = original_shape
                            input_size = self.config.get('input', {}).get('imgsz', 640)
                            scale = max(w / input_size, h / input_size)
                            
                            x1 *= scale
                            y1 *= scale
                            x2 *= scale
                            y2 *= scale
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_id': class_id,
                                'class_name': self.visualizer.class_names[class_id] if class_id < len(self.visualizer.class_names) else 'unknown'
                            })
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Process a single frame."""
        start_time = time.time()
        
        # Preprocess
        processed_frame = self.preprocess_frame(frame)
        
        # Inference
        if self.onnx_engine:
            # ONNX inference
            input_data = processed_frame.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
            outputs = self.onnx_engine.predict(input_data)
            results = outputs  # Process ONNX outputs
        else:
            # YOLO inference
            results = self.model(processed_frame, verbose=False)
        
        # Postprocess
        detections = self.postprocess_detections(results, frame.shape[:2])
        
        # Apply tracking if enabled
        if self.tracker:
            tracked_objects = self.tracker.update(detections)
            detections = list(tracked_objects.values())
        
        processing_time = time.time() - start_time
        return detections, processing_time
    
    def process_video_stream(self, source: Union[str, int]) -> None:
        """Process video stream in real-time."""
        # Initialize video capture
        if isinstance(source, str) and source.lower() == 'webcam':
            source = self.config.get('input', {}).get('webcam_id', 0)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video stream initialized: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if saving results
        output_config = self.config.get('output', {})
        video_writer = None
        if output_config.get('save_results', True):
            output_dir = Path(output_config.get('save_dir', 'runs/inference'))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"detection_results_{int(time.time())}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*output_config.get('video', {}).get('fourcc', 'mp4v'))
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections, processing_time = self.process_frame(frame)
                
                # Update performance metrics
                self.processing_times.append(processing_time)
                current_time = time.time()
                if len(self.fps_counter) > 0:
                    fps = 1.0 / (current_time - self.fps_counter[-1]) if len(self.fps_counter) > 0 else 0
                else:
                    fps = 0
                self.fps_counter.append(current_time)
                
                # Log detections
                if self.inference_logger:
                    self.inference_logger.log_detection(frame_count, detections)
                    if frame_count % 100 == 0:  # Log performance every 100 frames
                        self.inference_logger.log_performance(fps, processing_time, frame_count)
                
                # Visualize results
                if output_config.get('show_results', True) or video_writer:
                    vis_frame = self.visualizer.draw_detections(
                        frame,
                        detections,
                        conf_threshold=self.config.get('detection', {}).get('conf_threshold', 0.5),
                        show_labels=output_config.get('show_labels', True),
                        show_conf=output_config.get('show_conf', True),
                        line_thickness=output_config.get('line_thickness', 2)
                    )
                    
                    # Add FPS counter
                    realtime_config = self.config.get('realtime', {})
                    if realtime_config.get('display_fps', True):
                        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show frame
                    if output_config.get('show_results', True):
                        cv2.imshow('Conveyor Detection', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Save frame
                    if video_writer:
                        video_writer.write(vis_frame)
                
                frame_count += 1
                
                # Limit FPS if specified
                max_fps = self.config.get('realtime', {}).get('max_fps', None)
                if max_fps:
                    elapsed = time.time() - current_time
                    target_time = 1.0 / max_fps
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
        
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Save final logs
            if self.inference_logger:
                self.inference_logger.save_all()
            
            # Print final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            self.logger.info(f"Processing completed:")
            self.logger.info(f"  Total frames: {frame_count}")
            self.logger.info(f"  Total time: {total_time:.2f}s")
            self.logger.info(f"  Average FPS: {avg_fps:.2f}")
            self.logger.info(f"  Average processing time: {avg_processing_time*1000:.2f}ms")
    
    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process a single image."""
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Process
        detections, processing_time = self.process_frame(frame)
        
        # Save results if configured
        output_config = self.config.get('output', {})
        if output_config.get('save_results', True):
            output_dir = Path(output_config.get('save_dir', 'runs/inference'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save annotated image
            vis_frame = self.visualizer.draw_detections(frame, detections)
            output_path = output_dir / f"result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(output_path), vis_frame)
            
            # Save detection results
            if output_config.get('save_txt', True):
                txt_path = output_dir / f"result_{Path(image_path).stem}.txt"
                with open(txt_path, 'w') as f:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        f.write(f"{det['class_id']} {det['confidence']:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")
        
        self.logger.info(f"Processed image: {image_path} ({len(detections)} detections, {processing_time*1000:.2f}ms)")
        return detections
    
    def benchmark_performance(self, num_frames: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        self.logger.info(f"Benchmarking performance with {num_frames} frames...")
        
        # Create dummy frame
        input_config = self.config.get('input', {})
        imgsz = input_config.get('imgsz', 640)
        dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.process_frame(dummy_frame)
        
        # Benchmark
        times = []
        for _ in range(num_frames):
            start_time = time.time()
            self.process_frame(dummy_frame)
            times.append(time.time() - start_time)
        
        times = np.array(times)
        
        results = {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'mean_fps': float(1.0 / np.mean(times)),
            'frames_tested': num_frames
        }
        
        self.logger.info(f"Benchmark results: {results['mean_fps']:.2f} FPS average")
        return results

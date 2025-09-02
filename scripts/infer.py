#!/usr/bin/env python3
"""
Inference script for conveyor object detection system.

Usage:
    python scripts/infer.py --weights runs/train/exp/weights/best.pt --source webcam
    python scripts/infer.py --weights runs/train/exp/weights/best.pt --source test_videos/ --save-results
    python scripts/infer.py --weights runs/train/exp/weights/best.pt --source image.jpg
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import ConveyorInferenceEngine
from src.utils.logging import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference for conveyor object detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference.yaml",
        help="Path to inference configuration file"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt, .onnx, .engine)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Input source: webcam, video file, image file, or directory"
    )
    
    parser.add_argument(
        "--webcam-id",
        type=int,
        default=0,
        help="Webcam ID (0, 1, 2, etc.)"
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for detections (overrides config)"
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU threshold for NMS (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: auto, cpu, 0, 1, etc. (overrides config)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size (overrides config)"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detection results"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in real-time"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable display (overrides config)"
    )
    
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as text files"
    )
    
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence scores in saved results"
    )
    
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter by class IDs: --classes 0 1"
    )
    
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="Maximum detections per image (overrides config)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=100,
        help="Number of frames for benchmarking"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def override_config(engine, args):
    """Override configuration with command line arguments."""
    config = engine.config
    
    # Override model configuration
    if args.weights:
        config.setdefault('model', {})['weights_path'] = args.weights
    
    if args.device:
        config.setdefault('model', {})['device'] = args.device
    
    # Override input configuration
    if args.source:
        config.setdefault('input', {})['source'] = args.source
    
    if args.webcam_id is not None:
        config.setdefault('input', {})['webcam_id'] = args.webcam_id
    
    if args.imgsz:
        config.setdefault('input', {})['imgsz'] = args.imgsz
    
    # Override detection parameters
    if args.conf_threshold is not None:
        config.setdefault('detection', {})['conf_threshold'] = args.conf_threshold
    
    if args.iou_threshold is not None:
        config.setdefault('detection', {})['iou_threshold'] = args.iou_threshold
    
    if args.max_det is not None:
        config.setdefault('detection', {})['max_detections'] = args.max_det
    
    if args.classes is not None:
        config.setdefault('detection', {})['classes'] = args.classes
    
    # Override output configuration
    if args.save_results:
        config.setdefault('output', {})['save_results'] = True
    
    if args.save_dir:
        config.setdefault('output', {})['save_dir'] = args.save_dir
    
    if args.show:
        config.setdefault('output', {})['show_results'] = True
    
    if args.no_show:
        config.setdefault('output', {})['show_results'] = False
    
    if args.save_txt:
        config.setdefault('output', {})['save_txt'] = True
    
    if args.save_conf:
        config.setdefault('output', {})['save_conf'] = True


def process_source(engine, source, logger):
    """Process different types of input sources."""
    source_path = Path(source)
    
    if source.lower() == 'webcam':
        # Webcam input
        logger.info("Starting webcam inference...")
        engine.process_video_stream('webcam')
        
    elif source_path.is_file():
        # Single file input
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Video file
            logger.info(f"Processing video file: {source}")
            engine.process_video_stream(str(source_path))
        else:
            # Image file
            logger.info(f"Processing image file: {source}")
            detections = engine.process_image(str(source_path))
            logger.info(f"Found {len(detections)} detections")
            
    elif source_path.is_dir():
        # Directory input
        logger.info(f"Processing directory: {source}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images")
        
        total_detections = 0
        for image_file in image_files:
            detections = engine.process_image(str(image_file))
            total_detections += len(detections)
        
        logger.info(f"Processed {len(image_files)} images with {total_detections} total detections")
        
    else:
        # Try as video stream (URL or device ID)
        try:
            if source.isdigit():
                source = int(source)
            logger.info(f"Processing video stream: {source}")
            engine.process_video_stream(source)
        except Exception as e:
            logger.error(f"Could not process source '{source}': {e}")
            return False
    
    return True


def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("infer", log_level=log_level)
    
    try:
        # Initialize inference engine
        logger.info(f"Initializing inference engine with config: {args.config}")
        engine = ConveyorInferenceEngine(args.config)
        
        # Override config with command line arguments
        override_config(engine, args)
        
        if args.benchmark:
            # Benchmark mode
            logger.info(f"Running performance benchmark with {args.benchmark_frames} frames...")
            results = engine.benchmark_performance(args.benchmark_frames)
            
            logger.info("Benchmark Results:")
            logger.info(f"  Mean FPS: {results['mean_fps']:.2f}")
            logger.info(f"  Mean inference time: {results['mean_inference_time']*1000:.2f}ms")
            logger.info(f"  Min inference time: {results['min_inference_time']*1000:.2f}ms")
            logger.info(f"  Max inference time: {results['max_inference_time']*1000:.2f}ms")
            logger.info(f"  Std inference time: {results['std_inference_time']*1000:.2f}ms")
            
        else:
            # Normal inference mode
            success = process_source(engine, args.source, logger)
            if not success:
                return 1
        
        logger.info("Inference completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

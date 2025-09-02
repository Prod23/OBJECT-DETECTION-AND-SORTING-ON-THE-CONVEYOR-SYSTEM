#!/usr/bin/env python3
"""
Training script for conveyor object detection system.

Usage:
    python scripts/train.py --config config/train.yaml
    python scripts/train.py --config config/train.yaml --resume runs/train/exp/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import YOLOTrainer
from src.utils.logging import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model for conveyor object detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML file (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: auto, cpu, 0, 1, etc. (overrides config)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of data loading workers (overrides config)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project directory for saving results (overrides config)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (overrides config)"
    )
    
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate the model without training"
    )
    
    parser.add_argument(
        "--export",
        nargs="*",
        default=None,
        help="Export formats after training: onnx, engine, tflite, etc."
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def override_config(trainer, args):
    """Override configuration with command line arguments."""
    config = trainer.config
    
    # Override data path
    if args.data:
        config.setdefault('data', {})['dataset_path'] = args.data
    
    # Override training parameters
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    
    if args.workers:
        config.setdefault('training', {})['workers'] = args.workers
    
    # Override logging parameters
    if args.project:
        config.setdefault('logging', {})['save_dir'] = args.project
    
    if args.name:
        config.setdefault('logging', {})['experiment_name'] = args.name


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("train", log_level=log_level)
    
    try:
        # Initialize trainer
        logger.info(f"Initializing trainer with config: {args.config}")
        trainer = YOLOTrainer(args.config)
        
        # Override config with command line arguments
        override_config(trainer, args)
        
        if args.evaluate_only:
            # Evaluation only mode
            if not args.resume:
                logger.error("--resume must be specified for evaluation-only mode")
                return 1
            
            logger.info("Running evaluation only...")
            results = trainer.evaluate(args.resume)
            logger.info(f"Evaluation results: {results}")
            
        else:
            # Training mode
            logger.info("Starting training...")
            results = trainer.train(resume=args.resume)
            logger.info("Training completed successfully!")
            
            # Export model if requested
            if args.export:
                logger.info(f"Exporting model to formats: {args.export}")
                weights_path = results.get('weights_path', {}).get('best')
                if weights_path:
                    exported_paths = trainer.export_model(
                        weights_path=weights_path,
                        export_formats=args.export
                    )
                    logger.info(f"Model exported: {exported_paths}")
                else:
                    logger.warning("No weights found for export")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

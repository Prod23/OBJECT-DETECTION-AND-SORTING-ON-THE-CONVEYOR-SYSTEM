#!/usr/bin/env python3
"""
Data preparation script for conveyor object detection system.

Usage:
    python scripts/prepare_data.py --images data/raw/images --labels data/raw/labels
    python scripts/prepare_data.py --images data/raw/images --labels data/raw/labels --output data/processed
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import DatasetManager
from src.utils.logging import setup_logger
from src.utils.visualization import DataAnalysisVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for YOLO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to images directory"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed dataset"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training split ratio"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio"
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.05,
        help="Test split ratio"
    )
    
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["bottle", "box"],
        help="Class names in order"
    )
    
    parser.add_argument(
        "--convert-labels",
        action="store_true",
        help="Convert labels to YOLO format"
    )
    
    parser.add_argument(
        "--class-mapping",
        type=str,
        default=None,
        help="JSON file with class name to ID mapping for label conversion"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform dataset analysis and generate visualizations"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset integrity"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main data preparation function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("prepare_data", log_level=log_level)
    
    try:
        # Initialize dataset manager
        dataset_manager = DatasetManager(args.output)
        
        # Convert labels if requested
        if args.convert_labels:
            logger.info("Converting labels to YOLO format...")
            
            class_mapping = {"Bottle": 0, "Box": 1}  # Default mapping
            if args.class_mapping:
                with open(args.class_mapping, 'r') as f:
                    class_mapping = json.load(f)
            
            converted_labels_dir = Path(args.output) / "converted_labels"
            dataset_manager.convert_labels_to_yolo(
                labels_dir=args.labels,
                output_dir=str(converted_labels_dir),
                class_mapping=class_mapping
            )
            labels_dir = str(converted_labels_dir)
        else:
            labels_dir = args.labels
        
        # Create YOLO dataset structure
        logger.info("Creating YOLO dataset structure...")
        dataset_yaml_path = dataset_manager.create_yolo_dataset_structure(
            images_dir=args.images,
            labels_dir=labels_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            class_names=args.class_names
        )
        
        logger.info(f"Dataset created successfully: {dataset_yaml_path}")
        
        # Validate dataset if requested
        if args.validate:
            logger.info("Validating dataset...")
            validation_results = dataset_manager.validate_dataset(dataset_yaml_path)
            
            if validation_results['valid']:
                logger.info("Dataset validation passed!")
            else:
                logger.warning("Dataset validation failed!")
                for error in validation_results['errors']:
                    logger.error(f"  Error: {error}")
            
            for warning in validation_results['warnings']:
                logger.warning(f"  Warning: {warning}")
            
            # Print statistics
            logger.info("Dataset statistics:")
            for split, stats in validation_results['statistics'].items():
                logger.info(f"  {split}: {stats['matched_pairs']} pairs ({stats['images']} images, {stats['labels']} labels)")
        
        # Analyze dataset if requested
        if args.analyze:
            logger.info("Analyzing dataset...")
            analysis = dataset_manager.analyze_dataset(dataset_yaml_path)
            
            # Print analysis summary
            logger.info("Dataset Analysis Summary:")
            logger.info(f"  Classes: {analysis['class_names']}")
            logger.info(f"  Total class distribution: {analysis['class_distribution']}")
            
            for split, stats in analysis['splits'].items():
                logger.info(f"  {split} split: {stats['total_objects']} objects in {stats['num_images']} images")
                for class_name, count in stats['class_counts'].items():
                    logger.info(f"    {class_name}: {count}")
            
            # Print bbox statistics
            logger.info("Bounding Box Statistics:")
            for stat_name, stat_values in analysis['bbox_stats'].items():
                if stat_values:
                    logger.info(f"  {stat_name}: mean={stat_values['mean']:.3f}, std={stat_values['std']:.3f}")
            
            # Generate visualizations
            try:
                visualizer = DataAnalysisVisualizer()
                output_dir = Path(args.output) / "analysis"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Class distribution plot
                visualizer.plot_class_distribution(
                    analysis['class_distribution'],
                    save_path=str(output_dir / "class_distribution.png")
                )
                
                # Bbox statistics plots (if we have data)
                if any(analysis['bbox_stats'].values()):
                    import pandas as pd
                    
                    # Create DataFrame for bbox statistics
                    bbox_data = []
                    # This would need to be populated with actual bbox data
                    # For now, we'll skip the detailed bbox plots
                    
                logger.info(f"Analysis visualizations saved to: {output_dir}")
                
            except Exception as e:
                logger.warning(f"Could not generate visualizations: {e}")
            
            # Save analysis to file
            analysis_file = Path(args.output) / "dataset_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis saved to: {analysis_file}")
        
        logger.info("Data preparation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

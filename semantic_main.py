#!/usr/bin/env python
"""
Command-line interface for SAMGeo with semantic segmentation.
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
import config

# Import pipeline modules
from pipeline.semantic_pipeline import run_pipeline_with_semantics
from visualization.visualization import run_visualization, list_available_visualizations
from utils.io_utils import setup_logging


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="SAMGeo: Segment Anything Model for Geospatial Data with Semantic Classification"
    )

    # Main arguments
    parser.add_argument(
        "--mode",
        choices=["pipeline", "visualize", "both"],
        default="both",
        help="Operation mode: pipeline (segmentation + classification), visualize, or both",
    )

    # Input/output arguments
    parser.add_argument("--image", type=str, help="Path to input satellite image")
    parser.add_argument("--mask", type=str, help="Path to output segmentation mask")
    parser.add_argument("--vector", type=str, help="Path to output vector data")
    parser.add_argument(
        "--geo-vector", type=str, help="Path to output geographic vector data"
    )
    parser.add_argument(
        "--semantic-model", type=str, help="Path to semantic model weights"
    )
    parser.add_argument(
        "--semantic-mask", type=str, help="Path to output semantic mask"
    )
    parser.add_argument(
        "--classified", type=str, help="Path to output classified vector data"
    )

    # Visualization arguments
    parser.add_argument(
        "--shapefile", type=str, help="Path to shapefile for visualization"
    )
    parser.add_argument(
        "--viz-type",
        type=str,
        choices=list_available_visualizations(),
        help="Visualization type",
    )
    parser.add_argument(
        "--viz-output", type=str, help="Directory to save visualizations"
    )

    # Additional options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no-metadata", action="store_true", help="Disable metadata generation"
    )

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Set debug mode
    if args.debug:
        config.DEBUG = True

    # Set up logging
    logger = setup_logging()

    # Start timing
    start_time = datetime.now()

    try:
        # Run pipeline if requested
        if args.mode in ["pipeline", "both"]:
            logger.info(
                "Running SAM segmentation pipeline with semantic classification..."
            )

            # Check if input image is provided
            if args.image is None:
                logger.error("Input image path is required for pipeline mode")
                sys.exit(1)

            # Run pipeline
            classified_gdf, output_files = run_pipeline_with_semantics(
                image_path=args.image,
                output_mask_path=args.mask,
                output_vector_path=args.vector,
                output_geo_path=args.geo_vector,
                semantic_model_path=args.semantic_model,
                semantic_mask_path=args.semantic_mask,
                output_classified_path=args.classified,
            )

            logger.info("Pipeline completed successfully")

        # Run visualization if requested
        if args.mode in ["visualize", "both"]:
            logger.info("Running visualization...")

            # Determine shapefile path
            shapefile_path = args.shapefile
            if shapefile_path is None:
                if args.mode == "both" and args.classified:
                    # Use the specified classified output path
                    shapefile_path = args.classified
                elif (
                    args.mode == "both"
                    and "output_files" in locals()
                    and "classified_vector" in output_files
                ):
                    # Use the output from the pipeline
                    shapefile_path = output_files["classified_vector"]
                elif (
                    args.mode == "both"
                    and "output_files" in locals()
                    and "geo_vector" in output_files
                ):
                    # Use the unclassified output from the pipeline
                    shapefile_path = output_files["geo_vector"]

            if shapefile_path is None:
                logger.error("No shapefile specified and none produced by pipeline")
                sys.exit(1)

            logger.info(f"Using shapefile for visualization: {shapefile_path}")

            # Determine semantic mask path
            semantic_mask_path = args.semantic_mask
            if (
                semantic_mask_path is None
                and args.mode == "both"
                and "output_files" in locals()
            ):
                semantic_mask_path = output_files.get("semantic_mask")

            # Run visualization
            results = run_visualization(
                shapefile_path=shapefile_path,
                image_path=args.image,
                semantic_mask_path=semantic_mask_path,
                output_dir=args.viz_output,
                visualization_type=args.viz_type,
            )

            logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)

    # End timing
    end_time = datetime.now()
    logger.info(
        f"Total execution time: {(end_time - start_time).total_seconds():.2f} seconds"
    )


if __name__ == "__main__":
    main()

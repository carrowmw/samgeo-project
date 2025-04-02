#!/usr/bin/env python
"""
Main entry point for SAMGeo pipeline and visualization.
"""
import os
import sys
import argparse
from datetime import datetime

# Import local modules
import config
from pipeline import run_pipeline
from visualization import run_visualization, list_available_visualizations
from utils.io_utils import (
    find_latest_file,
    save_metadata,
    create_run_metadata,
    setup_logging,
)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="SAMGeo: Segment Anything Model for Geospatial Data"
    )

    # Main arguments
    parser.add_argument(
        "--mode",
        choices=["pipeline", "visualize", "both"],
        default="both",
        help="Operation mode: pipeline, visualize, or both",
    )

    # Pipeline arguments
    parser.add_argument("--image", type=str, help="Path to input satellite image")
    parser.add_argument("--mask", type=str, help="Path to output segmentation mask")
    parser.add_argument("--vector", type=str, help="Path to output vector data")
    parser.add_argument(
        "--geo-vector", type=str, help="Path to output geographic vector data"
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
    parser.add_argument(
        "--orig-image",
        type=str,
        help="Path to original image for overlay visualization",
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

    # Start timestamp
    start_time = datetime.now()

    try:
        # Run pipeline if requested
        if args.mode in ["pipeline", "both"]:
            logger.info("Running SAM segmentation pipeline...")

            # Check if input image is provided
            if args.image is None:
                logger.error("Input image path is required for pipeline mode")
                sys.exit(1)

            # Run pipeline
            gdf, output_files = run_pipeline(
                image_path=args.image,
                output_mask_path=args.mask,
                output_vector_path=args.vector,
                output_geo_path=args.geo_vector,
            )

            # Generate metadata
            if not args.no_metadata:
                logger.info("Generating pipeline metadata...")
                processing_time = (datetime.now() - start_time).total_seconds()
                metadata = create_run_metadata(
                    image_path=args.image,
                    mask_path=output_files["mask"],
                    vector_path=output_files["vector"],
                    geo_vector_path=output_files["geo_vector"],
                    processing_time=processing_time,
                )

                # Save metadata
                metadata_path = os.path.join(
                    config.OUTPUT_DIR,
                    f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
                save_metadata(metadata_path, metadata)
                logger.info(f"Metadata saved to {metadata_path}")

            logger.info("Pipeline completed successfully")

        # Run visualization if requested
        if args.mode in ["visualize", "both"]:
            logger.info("Running visualization...")

            # Determine shapefile path
            shapefile_path = args.shapefile
            if shapefile_path is None:
                if (
                    args.mode == "both"
                    and "geo_vector" in locals()
                    and "output_files" in locals()
                ):
                    # Use the output from the pipeline
                    shapefile_path = output_files["geo_vector"]
                else:
                    # Find the latest shapefile
                    shapefile_path = find_latest_file(config.OUTPUT_DIR, "*.shp")
                    if shapefile_path is None:
                        logger.error("No shapefile found and none provided")
                        sys.exit(1)

            # Determine original image path
            orig_image_path = args.orig_image
            if orig_image_path is None and args.image is not None:
                orig_image_path = args.image

            # Run visualization
            results = run_visualization(
                shapefile_path=shapefile_path,
                image_path=orig_image_path,
                output_dir=args.viz_output,
                visualization_type=args.viz_type,
            )

            logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)

    # End timestamp
    end_time = datetime.now()
    logger.info(
        f"Total execution time: {(end_time - start_time).total_seconds():.2f} seconds"
    )


if __name__ == "__main__":
    main()

# pipeline/pipeline.py
"""
Complete pipeline orchestration for SAMGeo.
"""
import os
import sys
from datetime import datetime

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import pipeline modules
from .processor import process_image
from .segmentation import initialise_sam_class
from .vectorization import polygonise_raster_data, convert_pixel_to_geo_coords


def run_pipeline(
    image_path, output_mask_path=None, output_vector_path=None, output_geo_path=None
):
    """
    Run the complete SAMGeo pipeline.

    Args:
        image_path: Path to the input satellite image
        output_mask_path: Path to save the segmentation mask
        output_vector_path: Path to save the vector data
        output_geo_path: Path to save the vector data with geographic coordinates

    Returns:
        gdf: GeoDataFrame with vectorized polygons
        output_files: Dictionary with paths to output files
    """
    # Set default output paths if not provided
    if output_mask_path is None:
        output_mask_path = config.DEFAULT_MASK_OUTPUT

    if output_vector_path is None:
        output_vector_path = config.DEFAULT_VECTOR_OUTPUT

    if output_geo_path is None:
        output_geo_path = config.DEFAULT_GEO_VECTOR_OUTPUT

    # Start timing
    start_time = datetime.now()

    print("\n" + "=" * 50)
    print("SAM Segmentation Pipeline")
    print("=" * 50)

    print(f"Processing image: {image_path}")

    # Process the image
    processed_image, geo_info, scale_factor = process_image(
        image_path, scale=config.SCALE_IMAGE, scale_factor=config.SCALE_FACTOR
    )

    print("\nRunning SAM segmentation...")
    # Run SAM
    sam = initialise_sam_class(processed_image, output_mask_path, geo_info)

    print("\nConverting to vector data...")
    # Convert masks to vectors
    gdf = polygonise_raster_data(sam, output_mask_path, output_vector_path, geo_info)

    # Make sure coordinates are correct
    print("\nVerifying coordinates...")
    if gdf.crs != geo_info["crs"]:
        print("Coordinate systems don't match, converting...")
        gdf = convert_pixel_to_geo_coords(
            gdf, geo_info["transform"], geo_info["crs"], scale_factor
        )

    # Save the final version
    print(f"\nSaving geographic shapefile to {output_geo_path}")
    gdf.to_file(output_geo_path)

    # Display the data
    print("\nResults summary:")
    print(f"Total features: {len(gdf)}")

    # End timing
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"\nProcessing time: {processing_time:.2f} seconds")

    print("\nSegmentation complete!")
    print("=" * 50)

    # Return the final GeoDataFrame and output file paths
    output_files = {
        "mask": output_mask_path,
        "vector": output_vector_path,
        "geo_vector": output_geo_path,
    }

    return gdf, output_files

# pipeline/semantic_pipeline.py
"""
Enhanced pipeline with semantic segmentation for SAMGeo.
"""
import os
import sys
from datetime import datetime

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import pipeline modules
from .pipeline import run_pipeline
from .semantic_segmentation import run_semantic_segmentation


def run_pipeline_with_semantics(
    image_path,
    output_mask_path=None,
    output_vector_path=None,
    output_geo_path=None,
    semantic_model_path=None,
    semantic_mask_path=None,
    output_classified_path=None,
):
    """
    Run the complete SAMGeo pipeline with semantic segmentation.

    Args:
        image_path: Path to the input satellite image
        output_mask_path: Path to save the segmentation mask
        output_vector_path: Path to save the vector data
        output_geo_path: Path to save the vector data with geographic coordinates
        semantic_model_path: Path to the semantic segmentation model
        semantic_mask_path: Path to save the semantic mask
        output_classified_path: Path to save the classified vector data

    Returns:
        gdf: GeoDataFrame with vectorized polygons with semantic labels
        output_files: Dictionary with paths to output files
    """
    # Set default output paths if not provided
    if output_mask_path is None:
        output_mask_path = config.DEFAULT_MASK_OUTPUT

    if output_vector_path is None:
        output_vector_path = config.DEFAULT_VECTOR_OUTPUT

    if output_geo_path is None:
        output_geo_path = config.DEFAULT_GEO_VECTOR_OUTPUT

    if semantic_mask_path is None:
        semantic_mask_path = os.path.join(
            config.OUTPUT_DIR,
            f"semantic_mask_{os.path.basename(image_path).split('.')[0]}.tif",
        )

    if output_classified_path is None:
        output_classified_path = os.path.join(
            config.OUTPUT_DIR,
            f"semantic_classified_{os.path.basename(image_path).split('.')[0]}.shp",
        )

    # Start timing
    start_time = datetime.now()

    print("\n" + "=" * 50)
    print("SAM Segmentation Pipeline with Semantic Classification")
    print("=" * 50)

    print(f"Processing image: {image_path}")

    # Run the standard SAM segmentation pipeline
    print("\nStep 1: Running SAM segmentation...")
    gdf, output_files = run_pipeline(
        image_path, output_mask_path, output_vector_path, output_geo_path
    )

    # Run semantic segmentation
    print("\nStep 2: Running semantic classification...")
    classified_gdf = run_semantic_segmentation(
        gdf,
        image_path,
        model_path=semantic_model_path,
        output_mask_path=semantic_mask_path,
        output_path=output_classified_path,
    )

    # Display the data
    print("\nResults summary:")
    print(f"Total features: {len(classified_gdf)}")

    # Print class distribution
    if "class" in classified_gdf.columns:
        class_counts = classified_gdf["class"].value_counts()
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(
                f"- {class_name}: {count} features ({count/len(classified_gdf)*100:.1f}%)"
            )

    # End timing
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"\nProcessing time: {processing_time:.2f} seconds")

    print("\nSemantic Segmentation complete!")
    print("=" * 50)

    # Return the final GeoDataFrame and output file paths
    output_files.update(
        {
            "semantic_mask": semantic_mask_path,
            "classified_vector": output_classified_path,
        }
    )

    return classified_gdf, output_files

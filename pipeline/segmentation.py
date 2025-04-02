# pipeline/segmentation.py
"""
SAM segmentation module for SAMGeo pipeline.
"""
import os
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from samgeo import SamGeo
import sys

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def initialise_sam_class(processed_image, mask_output_path, geo_info):
    """
    Initialize and run SAM on the processed image.

    Args:
        processed_image: The processed image ready for SAM
        mask_output_path: Path to save the segmentation mask
        geo_info: Dictionary with georeferencing information

    Returns:
        sam: The initialized SAM model
    """
    if config.DEBUG:
        print("Initializing SAM model...")

    try:
        sam = SamGeo(
            model_type=config.SAM_MODEL_TYPE,
            checkpoint=config.SAM_CHECKPOINT,
            sam_kwargs=None,
        )
        if config.DEBUG:
            print("SAM model initialized successfully")

        # Generate mask using SAM with parameters from config
        if config.DEBUG:
            print("Generating segmentation mask...")
        sam.generate(processed_image, mask_output_path, **config.SAM_PARAMS)
        if config.DEBUG:
            print(f"Segmentation mask saved to {mask_output_path}")

        # Add visualization for debugging
        if config.DEBUG:
            debug_sam_input_path = os.path.join(config.DEBUG_DIR, "sam_input.png")
            plt.figure(figsize=config.VISUALIZATION_PARAMS["figsize"])
            plt.imshow(processed_image)
            plt.title("Processed Image")
            plt.savefig(debug_sam_input_path)
            plt.close()
            print(f"Saved SAM input image to {debug_sam_input_path}")

        # Now fix the georeferencing of the output mask
        add_georeferencing_to_mask(mask_output_path, geo_info)

        return sam

    except Exception as e:
        print(f"Error initializing SAM: {str(e)}")
        raise


def add_georeferencing_to_mask(mask_path, geo_info):
    """
    Add proper georeferencing to the mask TIF file.

    Args:
        mask_path: Path to the segmentation mask
        geo_info: Dictionary with georeferencing information
    """
    if config.DEBUG:
        print("Adding georeferencing to mask...")

    try:
        # Read the mask
        with rasterio.open(mask_path) as src:
            mask_data = src.read()
            mask_profile = src.profile

            # Debug info
            if config.DEBUG:
                print(f"Mask shape: {mask_data.shape}")
                print(f"Mask profile: {mask_profile}")

        # Update the profile with the correct georeferencing info
        mask_profile.update(
            crs=geo_info["crs"],
            transform=geo_info["transform"],
            width=geo_info["width"],
            height=geo_info["height"],
        )

        # Check if dimensions match
        if mask_data.shape[1:] != (geo_info["height"], geo_info["width"]):
            print("WARNING: Mask dimensions don't match expected dimensions!")
            if config.DEBUG:
                print(f"Mask dimensions: {mask_data.shape[1:]}")
                print(f"Expected: {(geo_info['height'], geo_info['width'])}")

            # Resize mask if needed
            resized_data = np.zeros(
                (mask_data.shape[0], geo_info["height"], geo_info["width"]),
                dtype=mask_data.dtype,
            )

            for i in range(mask_data.shape[0]):
                resized_data[i] = cv2.resize(
                    mask_data[i],
                    (geo_info["width"], geo_info["height"]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_data = resized_data

        # Write back the mask with proper georeferencing
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(mask_data)

        if config.DEBUG:
            print("Georeferencing added successfully")

        # Visualize the mask for debugging
        if config.DEBUG:
            debug_mask_path = os.path.join(config.DEBUG_DIR, "mask.png")
            plt.figure(figsize=config.VISUALIZATION_PARAMS["figsize"])
            plt.imshow(mask_data[0], cmap="viridis")
            plt.title("Segmentation Mask")
            plt.savefig(debug_mask_path)
            plt.close()
            print(f"Saved mask visualization to {debug_mask_path}")

    except Exception as e:
        print(f"Error adding georeferencing to mask: {str(e)}")
        raise

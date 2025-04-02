# pipeline/processor.py
"""
Image processing module for SAMGeo pipeline.
"""
import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import sys

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def process_image(
    image_path, scale=config.SCALE_IMAGE, scale_factor=config.SCALE_FACTOR
):
    """
    Process the satellite image to prepare it for SAM.

    Args:
        image_path: Path to the satellite image
        scale: Whether to scale the image
        scale_factor: Scale factor for resizing

    Returns:
        processed_image: The processed image ready for SAM
        geo_info: Dictionary with georeferencing information
        scale_factor: The actual scale factor used
    """
    if config.DEBUG:
        print(f"Processing image: {image_path}")

    # Open the image with rasterio to get geo information
    with rasterio.open(image_path) as src:
        transform = src.transform
        profile = src.profile
        crs = src.crs
        image = src.read()

        # Debug information
        if config.DEBUG:
            print(f"Image shape: {image.shape}")
            print(f"CRS: {crs}")
            print(f"Transform: {transform}")

    # Store these for later reference
    geo_info = {
        "transform": transform,
        "crs": crs,
        "profile": profile,
        "width": src.width,
        "height": src.height,
    }

    # Transpose from (bands, H, W) to (H, W, bands)
    if image.shape[0] <= 4:  # Check if first dimension is channels
        image = np.transpose(image, (1, 2, 0))
        if config.DEBUG:
            print(f"Transposed image shape: {image.shape}")

    # Convert to 8-bit
    image_float = image.astype(np.float32)

    # Check for NaN values
    if np.isnan(image_float).any():
        print("WARNING: Image contains NaN values. Replacing with zeros.")
        image_float = np.nan_to_num(image_float)

    # Check min and max to ensure we don't divide by zero
    min_val = image_float.min()
    max_val = image_float.max()
    if config.DEBUG:
        print(f"Image min/max values: {min_val}/{max_val}")

    if max_val == min_val:
        print("WARNING: Image has no contrast (min=max). Adding small epsilon.")
        max_val += 0.001

    image_normalized = (image_float - min_val) / (max_val - min_val)
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    # Take only the first 3 channels (RGB) if available
    if image_uint8.shape[2] >= 3:
        image_rgb = image_uint8[:, :, :3]
        if config.DEBUG:
            print("Using RGB channels")
    else:
        # Handle grayscale or other formats by duplicating channels
        if config.DEBUG:
            print(
                f"WARNING: Image has {image_uint8.shape[2]} channels. Converting to RGB."
            )
        image_rgb = np.zeros(
            (image_uint8.shape[0], image_uint8.shape[1], 3), dtype=np.uint8
        )
        for i in range(min(3, image_uint8.shape[2])):
            image_rgb[:, :, i] = image_uint8[:, :, i]

    # Save debug image to check processing
    if config.DEBUG:
        debug_processed_path = os.path.join(config.DEBUG_DIR, "processed_image.png")
        cv2.imwrite(debug_processed_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved processed image to {debug_processed_path}")

    # Resize if needed
    if scale:
        if config.DEBUG:
            print(f"Scaling image by factor of {scale_factor}")
        image_resized = cv2.resize(
            image_rgb,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )

        # Update the transform for the scaled image
        scaled_transform = Affine(
            transform.a / scale_factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e / scale_factor,
            transform.f,
        )
        geo_info["transform"] = scaled_transform
        geo_info["width"] = int(src.width * scale_factor)
        geo_info["height"] = int(src.height * scale_factor)

        # Save resized image for debugging
        if config.DEBUG:
            debug_resized_path = os.path.join(config.DEBUG_DIR, "resized_image.png")
            cv2.imwrite(
                debug_resized_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
            )
            print(f"Saved resized image to {debug_resized_path}")

        return image_resized, geo_info, scale_factor
    else:
        return image_rgb, geo_info, 1.0

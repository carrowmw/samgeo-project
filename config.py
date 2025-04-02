"""
Centralized configuration for SAMGeo pipeline and visualization.
"""

import os
from datetime import datetime

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, VISUALIZATION_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Default filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_MASK_OUTPUT = os.path.join(OUTPUT_DIR, f"segment_{timestamp}.tif")
DEFAULT_VECTOR_OUTPUT = os.path.join(OUTPUT_DIR, f"segment_{timestamp}.shp")
DEFAULT_GEO_VECTOR_OUTPUT = os.path.join(OUTPUT_DIR, f"segment_geo_{timestamp}.shp")
DEFAULT_VIZ_OUTPUT = os.path.join(VISUALIZATION_DIR, f"segments_{timestamp}.html")

# SAM model settings
SAM_MODEL_TYPE = "vit_h"  # Options: vit_h, vit_l, vit_b
SAM_CHECKPOINT = os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth")

# Image processing parameters
SCALE_IMAGE = False  # Whether to scale the image
SCALE_FACTOR = 1  # Scale factor for image resizing (0.5 = 50% of original size)

# SAM segmentation parameters
SAM_PARAMS = {
    "batch": False,  # Use batch processing
    "foreground": True,  # Segment foreground objects
    "erosion_kernel": (1, 1),  # Erosion kernel size
    "mask_multiplier": 255,  # Value to multiply masks by
    "points_per_side": 32,  # Number of points per side for automatic segmentation
    "pred_iou_thresh": 0.88,  # Prediction IoU threshold
    "stability_score_thresh": 0.95,  # Stability score threshold
}

# Vectorization parameters
VECTORIZATION_PARAMS = {
    "simplify_tolerance": 0.0001,  # Simplification tolerance (lower = more detail)
    "min_area": 0,  # Minimum area to keep (in square units of the CRS)
    "filter_value": 0,  # Mask value to filter out (usually 0 = background)
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    "default_style": "plotly",  # Options: basic, with_basemap, with_image, interactive, folium
    "figsize": (12, 10),  # Figure size for matplotlib plots
    "cmap": "viridis",  # Colormap for plots
    "alpha": 0.7,  # Transparency level for segments
    "zoom": 16,  # Initial zoom level for interactive maps
    "mapbox_style": "carto-positron",  # Mapbox style for Plotly
    "colorscale": "Viridis",  # Colorscale for Plotly
}

# Debugging settings
DEBUG = True  # Enable/disable debug mode
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
if DEBUG:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# visualization/basic_viz.py
"""
Matplotlib-based visualization functions for SAMGeo.
"""
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
import numpy as np
import sys

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_segments_basic(shapefile_path, output_path=None, title="Segmentation Results"):
    """
    Create a basic visualization of segments using Matplotlib.

    Args:
        shapefile_path: Path to the shapefile
        output_path: Path to save the visualization
        title: Plot title

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
        gdf: GeoDataFrame with segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Print basic info
    print(f"Number of segments: {len(gdf)}")
    if len(gdf) > 0:
        print(f"Geometry type: {gdf.geometry.type.iloc[0]}")

    # Create plot
    fig, ax = plt.subplots(figsize=config.VISUALIZATION_PARAMS["figsize"])

    # Plot with random colors if many segments, otherwise use the value column
    if len(gdf) > 20:
        gdf.plot(ax=ax, column=gdf.index, cmap="tab20", legend=False)
    else:
        if "value" in gdf.columns:
            gdf.plot(
                ax=ax,
                column="value",
                cmap=config.VISUALIZATION_PARAMS["cmap"],
                legend=True,
            )
        else:
            gdf.plot(ax=ax, cmap=config.VISUALIZATION_PARAMS["cmap"])

    # Add styling
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Basic visualization saved to {output_path}")

    return fig, ax, gdf


def plot_segments_with_image(
    shapefile_path, image_path, output_path=None, title="Segments with Image"
):
    """
    Plot segments overlaid on the original satellite image.

    Args:
        shapefile_path: Path to the shapefile
        image_path: Path to the original satellite image
        output_path: Path to save the visualization
        title: Plot title

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
        gdf: GeoDataFrame with segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Read the image
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform

        # Check if we need to transpose
        if image.shape[0] in [1, 3, 4]:  # Channels first format
            image = np.transpose(image, (1, 2, 0))

        # If more than 3 channels, take only the first 3
        if len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, :3]

        # Normalize for display
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            for i in range(min(3, image.shape[2] if len(image.shape) > 2 else 1)):
                band = image[:, :, i] if len(image.shape) > 2 else image
                band_min, band_max = np.nanmin(band), np.nanmax(band)
                if band_max > band_min:
                    band = (band - band_min) / (band_max - band_min)
                    if len(image.shape) > 2:
                        image[:, :, i] = band
                    else:
                        image = band

    # Create figure
    fig, ax = plt.subplots(figsize=config.VISUALIZATION_PARAMS["figsize"])

    # Plot the image
    if len(image.shape) > 2 and image.shape[2] == 3:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap="gray")

    # Plot segments on top
    if gdf.crs != src.crs and gdf.crs is not None and src.crs is not None:
        try:
            gdf = gdf.to_crs(src.crs)
        except Exception as e:
            print(f"Warning: CRS conversion failed: {e}")

    gdf.boundary.plot(ax=ax, color="red", linewidth=1)

    # Add title
    ax.set_title(title, fontsize=16)

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Image overlay visualization saved to {output_path}")

    return fig, ax, gdf

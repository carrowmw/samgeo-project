# utils/geo_utils.py
"""
Geospatial utility functions for SAMGeo.
"""
import os
import rasterio
from rasterio.warp import transform_bounds, calculate_default_transform, reproject
from rasterio.crs import CRS
import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping
import sys

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def reproject_raster(src_path, dst_path, dst_crs="EPSG:4326"):
    """
    Reproject a raster to a different CRS.

    Args:
        src_path: Path to the source raster
        dst_path: Path to the destination raster
        dst_crs: Destination CRS
    """
    with rasterio.open(src_path) as src:
        # Calculate the optimal resolution and transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Update metadata for the output raster
        metadata = src.meta.copy()
        metadata.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        # Create the output raster
        with rasterio.open(dst_path, "w", **metadata) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                )

    return dst_path


def reproject_shapefile(src_path, dst_path, dst_crs="EPSG:4326"):
    """
    Reproject a shapefile to a different CRS.

    Args:
        src_path: Path to the source shapefile
        dst_path: Path to the destination shapefile
        dst_crs: Destination CRS
    """
    # Read the shapefile
    gdf = gpd.read_file(src_path)

    # Reproject
    gdf_reprojected = gdf.to_crs(dst_crs)

    # Save the reprojected shapefile
    gdf_reprojected.to_file(dst_path)

    return dst_path


def get_shapefile_info(shapefile_path):
    """
    Get basic information about a shapefile.

    Args:
        shapefile_path: Path to the shapefile

    Returns:
        info: Dictionary with shapefile information
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Get basic information
    info = {
        "crs": gdf.crs,
        "num_features": len(gdf),
        "geometry_types": gdf.geom_type.value_counts().to_dict(),
        "columns": list(gdf.columns),
        "bounds": gdf.total_bounds.tolist(),
        "file_size": os.path.getsize(shapefile_path),
    }

    # Get statistics for numeric columns
    for column in gdf.select_dtypes(include=[np.number]).columns:
        info[f"stats_{column}"] = {
            "min": gdf[column].min(),
            "max": gdf[column].max(),
            "mean": gdf[column].mean(),
            "median": gdf[column].median(),
        }

    return info


def get_raster_info(raster_path):
    """
    Get basic information about a raster.

    Args:
        raster_path: Path to the raster

    Returns:
        info: Dictionary with raster information
    """
    with rasterio.open(raster_path) as src:
        info = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": (src.res[0], src.res[1]),
            "nodata": src.nodata,
            "dtypes": [src.dtypes[i] for i in range(src.count)],
            "file_size": os.path.getsize(raster_path),
        }

        # Get basic statistics for each band
        stats = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            if src.nodata is not None:
                band = band[band != src.nodata]

            if band.size > 0:
                stats.append(
                    {
                        "min": float(np.min(band)),
                        "max": float(np.max(band)),
                        "mean": float(np.mean(band)),
                        "std": float(np.std(band)),
                    }
                )
            else:
                stats.append(
                    {
                        "min": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                    }
                )

        info["band_stats"] = stats

    return info


def create_bbox_shapefile(raster_path, output_path):
    """
    Create a shapefile with the bounding box of a raster.

    Args:
        raster_path: Path to the raster
        output_path: Path to save the shapefile

    Returns:
        gdf: GeoDataFrame with the bounding box
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    # Create a shapely box from the bounds
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=crs)

    # Save the shapefile
    gdf.to_file(output_path)

    return gdf

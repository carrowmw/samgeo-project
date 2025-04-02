# pipeline/vectorization.py
"""
Raster to vector conversion module for SAMGeo pipeline.
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import Polygon, mapping, shape
import matplotlib.pyplot as plt


# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def polygonise_raster_data(sam, raster_mask_path, vector_mask_output_path, geo_info):
    """
    Convert raster masks to vector polygons with proper CRS.

    Args:
        sam: The SAM model
        raster_mask_path: Path to the segmentation mask
        vector_mask_output_path: Path to save the vector data
        geo_info: Dictionary with georeferencing information

    Returns:
        gdf: GeoDataFrame with vectorized polygons
    """
    if config.DEBUG:
        print("Converting raster to vector polygons...")

    try:
        # Try the manual approach first for better control
        with rasterio.open(raster_mask_path) as src:
            mask = src.read(1)  # Read the first band

            # Debug info
            if config.DEBUG:
                print(f"Mask min/max: {mask.min()}/{mask.max()}")
                print(f"Unique values in mask: {np.unique(mask)}")

            # For debug: Save a visualization of the mask
            if config.DEBUG:
                debug_mask_before_poly_path = os.path.join(
                    config.DEBUG_DIR, "mask_before_poly.png"
                )
                plt.figure(figsize=config.VISUALIZATION_PARAMS["figsize"])
                plt.imshow(mask, cmap="viridis")
                plt.colorbar(label="Mask Values")
                plt.title("Mask Before Polygonization")
                plt.savefig(debug_mask_before_poly_path)
                plt.close()
                print(
                    f"Saved mask before polygonization to {debug_mask_before_poly_path}"
                )

            # Get the shapes with the correct transform
            results = (
                {"properties": {"value": value}, "geometry": geometry}
                for geometry, value in shapes(mask, transform=src.transform)
            )
            geoms = list(results)

            if config.DEBUG:
                print(f"Found {len(geoms)} geometries")

            # Filter out small polygons and zero values
            filter_value = config.VECTORIZATION_PARAMS["filter_value"]
            min_area = config.VECTORIZATION_PARAMS["min_area"]

            filtered_geoms = [
                g
                for g in geoms
                if g["properties"]["value"] > filter_value
                and shape(g["geometry"]).area > min_area
            ]

            if config.DEBUG:
                print(f"After filtering: {len(filtered_geoms)} geometries")

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(filtered_geoms, crs=geo_info["crs"])

        # Apply simplification to preserve details but reduce complexity
        simplify_tolerance = config.VECTORIZATION_PARAMS["simplify_tolerance"]
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance)

        # Save to file
        gdf.to_file(vector_mask_output_path)
        if config.DEBUG:
            print(f"Saved vector data to {vector_mask_output_path}")

        # Visualize the polygons
        if config.DEBUG:
            debug_polygons_path = os.path.join(config.DEBUG_DIR, "polygons.png")
            fig, ax = plt.subplots(figsize=config.VISUALIZATION_PARAMS["figsize"])
            gdf.plot(column="value", ax=ax, legend=True)
            plt.title("Vectorized Polygons")
            plt.savefig(debug_polygons_path)
            plt.close()
            print(f"Saved polygon visualization to {debug_polygons_path}")

        return gdf

    except Exception as e:
        print(f"Error in manual polygonization: {str(e)}")

        # Fall back to SamGeo's built-in function if manual approach fails
        print("Falling back to SamGeo polygonization...")
        try:
            sam.tiff_to_vector(
                raster_mask_path,
                vector_mask_output_path,
                simplify_tolerance=config.VECTORIZATION_PARAMS["simplify_tolerance"],
            )

            # Read the shapefile and set the CRS explicitly
            gdf = gpd.read_file(vector_mask_output_path)
            gdf.crs = geo_info["crs"]
            gdf.to_file(vector_mask_output_path)

            # Visualize for debug
            if config.DEBUG:
                debug_polygons_samgeo_path = os.path.join(
                    config.DEBUG_DIR, "polygons_samgeo.png"
                )
                fig, ax = plt.subplots(figsize=config.VISUALIZATION_PARAMS["figsize"])
                gdf.plot(column="value", ax=ax, legend=True)
                plt.title("Vectorized Polygons (SamGeo method)")
                plt.savefig(debug_polygons_samgeo_path)
                plt.close()
                print(
                    f"Saved SamGeo polygon visualization to {debug_polygons_samgeo_path}"
                )

            return gdf

        except Exception as e2:
            print(f"Error in SamGeo polygonization: {str(e2)}")
            raise


def convert_pixel_to_geo_coords(gdf_pixel, transform, crs, scale_factor=1.0):
    """
    Convert pixel coordinates to geographic coordinates.

    Args:
        gdf_pixel: GeoDataFrame with pixel coordinates
        transform: Affine transform
        crs: Coordinate reference system
        scale_factor: Scale factor for resizing

    Returns:
        gdf_geo: GeoDataFrame with geographic coordinates
    """
    if config.DEBUG:
        print("Converting coordinates if needed...")

    # If the GeoDataFrame already has a CRS and it matches what we expect, return it
    if gdf_pixel.crs is not None and gdf_pixel.crs == crs:
        if config.DEBUG:
            print("Coordinates already in the correct CRS, skipping conversion")
        return gdf_pixel

    gdf_geo = gdf_pixel.copy()
    if config.DEBUG:
        print(f"Converting from pixel coordinates to {crs}")

    # Function to transform a geometry from pixel to geographic coordinates
    def transform_geometry(geom):
        if geom.geom_type == "Polygon":
            # Get coordinates, adjusting for any scaling that was done
            pixel_coords = [
                (x / scale_factor, y / scale_factor) for x, y in geom.exterior.coords
            ]

            # Apply the affine transform to convert to geographic coordinates
            geo_coords = [
                rasterio.transform.xy(transform, y, x) for x, y in pixel_coords
            ]

            # Create a new polygon with geographic coordinates
            exterior = [(x, y) for x, y in geo_coords]

            # Handle interior rings (holes) if any
            interiors = []
            for interior in geom.interiors:
                pixel_interior = [
                    (x / scale_factor, y / scale_factor) for x, y in interior.coords
                ]
                geo_interior = [
                    rasterio.transform.xy(transform, y, x) for x, y in pixel_interior
                ]
                interiors.append([(x, y) for x, y in geo_interior])

            return Polygon(exterior, interiors)

        # Add handling for other geometry types if needed
        return geom

    # Apply the transformation to each geometry
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(transform_geometry)

    # Set the CRS to match the original image
    gdf_geo.crs = crs
    if config.DEBUG:
        print("Coordinate conversion complete")

    return gdf_geo

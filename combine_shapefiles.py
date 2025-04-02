#!/usr/bin/env python
"""
Combine Shapefiles Script

This script finds all shapefiles in the outputs directory and combines them
into a single shapefile. It preserves attributes and handles CRS differences.

Usage:
    python combine_shapefiles.py [--output OUTPUT_PATH] [--input INPUT_DIR]

Example:
    python combine_shapefiles.py --output combined_segments.shp --input ./outputs
"""

import os
import sys
import glob
import argparse
from datetime import datetime

try:
    import geopandas as gpd
    import pandas as pd
except ImportError:
    print("Error: This script requires geopandas and pandas.")
    print("Install them with: pip install geopandas pandas")
    sys.exit(1)

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def find_shapefiles(directory):
    """
    Find all shapefiles in the specified directory.

    Args:
        directory: Directory to search for shapefiles

    Returns:
        List of paths to shapefiles
    """
    # Search for all .shp files in the directory
    shapefiles = glob.glob(os.path.join(directory, "*.shp"))

    # Also search in subdirectories
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            subdir = os.path.join(root, dir_name)
            shapefiles.extend(glob.glob(os.path.join(subdir, "*.shp")))

    return shapefiles


def combine_shapefiles(shapefiles, output_path, target_crs=None):
    """
    Combine multiple shapefiles into a single shapefile.

    Args:
        shapefiles: List of paths to shapefiles
        output_path: Path to save the combined shapefile
        target_crs: Target CRS for the output (if None, use the CRS of the first shapefile)

    Returns:
        Path to the combined shapefile
    """
    if not shapefiles:
        print("No shapefiles found.")
        return None

    print(f"Found {len(shapefiles)} shapefiles.")

    # Initialize an empty list to store GeoDataFrames
    gdfs = []

    # Loop through each shapefile
    for shapefile in shapefiles:
        try:
            # Read the shapefile
            gdf = gpd.read_file(shapefile)

            # Skip empty shapefiles
            if len(gdf) == 0:
                print(f"Skipping empty shapefile: {shapefile}")
                continue

            # Add source filename as attribute
            gdf["source_file"] = os.path.basename(shapefile)

            # If no target CRS specified, use the CRS of the first valid shapefile
            if target_crs is None and gdf.crs is not None:
                target_crs = gdf.crs
                print(f"Using CRS from first shapefile: {target_crs}")

            # Add to the list
            gdfs.append(gdf)
            print(f"Added {len(gdf)} features from {os.path.basename(shapefile)}")

        except Exception as e:
            print(f"Error reading {shapefile}: {str(e)}")

    if not gdfs:
        print("No valid shapefiles found.")
        return None

    # Combine all GeoDataFrames
    combined_gdf = pd.concat(gdfs, ignore_index=True)

    # Ensure all geometries are valid
    combined_gdf = combined_gdf[~combined_gdf.is_empty]
    combined_gdf = combined_gdf[combined_gdf.is_valid]

    # Convert to target CRS if needed
    if target_crs is not None and combined_gdf.crs != target_crs:
        try:
            combined_gdf = combined_gdf.to_crs(target_crs)
            print(f"Converted all geometries to {target_crs}")
        except Exception as e:
            print(f"Warning: Could not convert to target CRS: {str(e)}")

    # Save to file
    combined_gdf.to_file(output_path)
    print(f"Combined {len(combined_gdf)} features into {output_path}")

    return output_path


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Combine shapefiles into a single shapefile."
    )
    parser.add_argument(
        "--output", type=str, help="Path to save the combined shapefile"
    )
    parser.add_argument(
        "--input", type=str, help="Directory containing shapefiles to combine"
    )

    args = parser.parse_args()

    # Set default paths if not provided
    input_dir = args.input if args.input else config.OUTPUT_DIR

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            config.OUTPUT_DIR, f"combined_segments_{timestamp}.shp"
        )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find and combine shapefiles
    shapefiles = find_shapefiles(input_dir)
    result_path = combine_shapefiles(shapefiles, output_path)

    if result_path:
        print("\nSummary:")
        print(f"- Input directory: {input_dir}")
        print(f"- Shapefiles found: {len(shapefiles)}")
        print(f"- Combined shapefile: {result_path}")


if __name__ == "__main__":
    main()

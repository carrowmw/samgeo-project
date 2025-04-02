# visualization/visualization.py
"""
Visualization orchestration module for SAMGeo.
"""
import os
import sys
from datetime import datetime

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import visualization modules
from .basic_viz import plot_segments_basic, plot_segments_with_image
from .interactive_viz import create_plotly_map, create_folium_map
from .semantic_viz import (
    plot_classified_segments,
    plot_classified_with_image,
    create_semantic_plotly_map,
    create_semantic_folium_map,
    visualize_semantic_mask,
)


def list_available_visualizations():
    """
    List all available visualization types.

    Returns:
        list: Available visualization types
    """
    return [
        "basic",  # Basic matplotlib plot
        "with_image",  # Overlay on original image
        "plotly",  # Interactive Plotly map
        "folium",  # Interactive Folium map
        "semantic",  # Semantic classification visualization
        "semantic_image",  # Semantic classification overlaid on image
        "semantic_plotly",  # Interactive semantic Plotly map
        "semantic_folium",  # Interactive semantic Folium map
        "semantic_mask",  # Semantic mask visualization
        "all",  # Generate all visualizations
    ]


def run_visualization(
    shapefile_path,
    image_path=None,
    semantic_mask_path=None,
    output_dir=None,
    visualization_type=None,
):
    """
    Run the visualization process.

    Args:
        shapefile_path: Path to the shapefile
        image_path: Path to the original image (required for 'with_image' type)
        semantic_mask_path: Path to semantic mask (required for 'semantic_mask' type)
        output_dir: Directory to save visualizations
        visualization_type: Type of visualization to generate

    Returns:
        results: Dictionary with generated figures/maps
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = config.VISUALIZATION_DIR

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set default visualization type if not provided
    if visualization_type is None:
        visualization_type = config.VISUALIZATION_PARAMS["default_style"]

    # Validate visualization type
    available_types = list_available_visualizations()
    if visualization_type not in available_types:
        print(f"Warning: Unknown visualization type '{visualization_type}'")
        print(f"Available types: {', '.join(available_types)}")
        print(f"Using default: {config.VISUALIZATION_PARAMS['default_style']}")
        visualization_type = config.VISUALIZATION_PARAMS["default_style"]

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dictionary to store results
    results = {}

    print("\n" + "=" * 50)
    print("SAM Segmentation Visualization")
    print("=" * 50)

    print(f"Shapefile: {shapefile_path}")
    print(f"Visualization type: {visualization_type}")
    print(f"Output directory: {output_dir}")

    # Check if the shapefile has semantic classification
    try:
        import geopandas as gpd

        gdf = gpd.read_file(shapefile_path)
        has_semantic = "class" in gdf.columns
    except Exception as e:
        print(f"Warning: Could not check if shapefile has semantic classification: {e}")
        has_semantic = False

    # Generate all visualizations if requested
    if visualization_type == "all":
        print("\nGenerating all visualizations...")

        # Basic plot
        output_path = os.path.join(output_dir, f"segments_basic_{timestamp}.png")
        fig, ax, gdf = plot_segments_basic(shapefile_path, output_path)
        results["basic"] = {"fig": fig, "ax": ax, "gdf": gdf, "path": output_path}

        # Plot with image if image path provided
        if image_path:
            output_path = os.path.join(
                output_dir, f"segments_with_image_{timestamp}.png"
            )
            fig, ax, gdf = plot_segments_with_image(
                shapefile_path, image_path, output_path
            )
            results["with_image"] = {
                "fig": fig,
                "ax": ax,
                "gdf": gdf,
                "path": output_path,
            }

        # Plotly map
        output_path = os.path.join(output_dir, f"segments_plotly_{timestamp}.html")
        fig, gdf = create_plotly_map(shapefile_path, output_path)
        results["plotly"] = {"fig": fig, "gdf": gdf, "path": output_path}

        # Folium map
        output_path = os.path.join(output_dir, f"segments_folium_{timestamp}.html")
        m, gdf = create_folium_map(shapefile_path, output_path)
        results["folium"] = {"map": m, "gdf": gdf, "path": output_path}

        # Semantic visualizations if applicable
        if has_semantic:
            # Semantic visualization
            output_path = os.path.join(
                output_dir, f"semantic_classification_{timestamp}.png"
            )
            fig, ax, gdf = plot_classified_segments(shapefile_path, output_path)
            results["semantic"] = {
                "fig": fig,
                "ax": ax,
                "gdf": gdf,
                "path": output_path,
            }

            # Semantic with image visualization
            if image_path:
                output_path = os.path.join(
                    output_dir, f"semantic_with_image_{timestamp}.png"
                )
                fig, ax, gdf = plot_classified_with_image(
                    shapefile_path, image_path, output_path
                )
                results["semantic_image"] = {
                    "fig": fig,
                    "ax": ax,
                    "gdf": gdf,
                    "path": output_path,
                }

            # Semantic Plotly map
            output_path = os.path.join(output_dir, f"semantic_plotly_{timestamp}.html")
            fig, gdf = create_semantic_plotly_map(shapefile_path, output_path)
            results["semantic_plotly"] = {"fig": fig, "gdf": gdf, "path": output_path}

            # Semantic Folium map
            output_path = os.path.join(output_dir, f"semantic_folium_{timestamp}.html")
            m, gdf = create_semantic_folium_map(shapefile_path, output_path)
            results["semantic_folium"] = {"map": m, "gdf": gdf, "path": output_path}

        # Semantic mask visualization if applicable
        if semantic_mask_path:
            output_path = os.path.join(output_dir, f"semantic_mask_{timestamp}.png")
            fig, ax = visualize_semantic_mask(semantic_mask_path, output_path)
            results["semantic_mask"] = {"fig": fig, "ax": ax, "path": output_path}

    # Generate specific visualization type
    else:
        if visualization_type == "basic":
            output_path = os.path.join(output_dir, f"segments_basic_{timestamp}.png")
            fig, ax, gdf = plot_segments_basic(shapefile_path, output_path)
            results["basic"] = {"fig": fig, "ax": ax, "gdf": gdf, "path": output_path}

        elif visualization_type == "with_image":
            if image_path:
                output_path = os.path.join(
                    output_dir, f"segments_with_image_{timestamp}.png"
                )
                fig, ax, gdf = plot_segments_with_image(
                    shapefile_path, image_path, output_path
                )
                results["with_image"] = {
                    "fig": fig,
                    "ax": ax,
                    "gdf": gdf,
                    "path": output_path,
                }
            else:
                print(
                    "Warning: Image path required for 'with_image' visualization type."
                )
                print("Falling back to basic visualization...")
                output_path = os.path.join(
                    output_dir, f"segments_basic_{timestamp}.png"
                )
                fig, ax, gdf = plot_segments_basic(shapefile_path, output_path)
                results["basic"] = {
                    "fig": fig,
                    "ax": ax,
                    "gdf": gdf,
                    "path": output_path,
                }

        elif visualization_type == "plotly":
            output_path = os.path.join(output_dir, f"segments_plotly_{timestamp}.html")
            fig, gdf = create_plotly_map(shapefile_path, output_path)
            results["plotly"] = {"fig": fig, "gdf": gdf, "path": output_path}

        elif visualization_type == "folium":
            output_path = os.path.join(output_dir, f"segments_folium_{timestamp}.html")
            m, gdf = create_folium_map(shapefile_path, output_path)
            results["folium"] = {"map": m, "gdf": gdf, "path": output_path}

        elif visualization_type == "semantic":
            if has_semantic:
                output_path = os.path.join(
                    output_dir, f"semantic_classification_{timestamp}.png"
                )
                fig, ax, gdf = plot_classified_segments(shapefile_path, output_path)
                results["semantic"] = {
                    "fig": fig,
                    "ax": ax,
                    "gdf": gdf,
                    "path": output_path,
                }
            else:
                print(
                    "Warning: Shapefile does not have semantic classification information."
                )
                print("Falling back to basic visualization...")
                output_path = os.path.join(
                    output_dir, f"segments_basic_{timestamp}.png"
                )
                fig, ax, gdf = plot_segments_basic(shapefile_path, output_path)
                results["basic"] = {
                    "fig": fig,
                    "ax": ax,
                    "gdf": gdf,
                    "path": output_path,
                }

        elif visualization_type == "semantic_image":
            if has_semantic and image_path:
                output_path = os.path.join(
                    output_dir, f"semantic_with_image_{timestamp}.png"
                )
                fig, ax, gdf = plot_classified_with_image(
                    shapefile_path, image_path, output_path
                )
                results["semantic_image"] = {
                    "fig": fig,
                    "ax": ax,
                    "gdf": gdf,
                    "path": output_path,
                }
            else:
                print(
                    "Warning: Semantic image visualization requires both classification data and image path."
                )
                print("Falling back to basic or with_image visualization...")
                if image_path:
                    output_path = os.path.join(
                        output_dir, f"segments_with_image_{timestamp}.png"
                    )
                    fig, ax, gdf = plot_segments_with_image(
                        shapefile_path, image_path, output_path
                    )
                    results["with_image"] = {
                        "fig": fig,
                        "ax": ax,
                        "gdf": gdf,
                        "path": output_path,
                    }
                else:
                    output_path = os.path.join(
                        output_dir, f"segments_basic_{timestamp}.png"
                    )
                    fig, ax, gdf = plot_segments_basic(shapefile_path, output_path)
                    results["basic"] = {
                        "fig": fig,
                        "ax": ax,
                        "gdf": gdf,
                        "path": output_path,
                    }

        elif visualization_type == "semantic_plotly":
            if has_semantic:
                output_path = os.path.join(
                    output_dir, f"semantic_plotly_{timestamp}.html"
                )
                fig, gdf = create_semantic_plotly_map(shapefile_path, output_path)
                results["semantic_plotly"] = {
                    "fig": fig,
                    "gdf": gdf,
                    "path": output_path,
                }
            else:
                print(
                    "Warning: Shapefile does not have semantic classification information."
                )
                print("Falling back to standard Plotly visualization...")
                output_path = os.path.join(
                    output_dir, f"segments_plotly_{timestamp}.html"
                )
                fig, gdf = create_plotly_map(shapefile_path, output_path)
                results["plotly"] = {"fig": fig, "gdf": gdf, "path": output_path}

        elif visualization_type == "semantic_folium":
            if has_semantic:
                output_path = os.path.join(
                    output_dir, f"semantic_folium_{timestamp}.html"
                )
                m, gdf = create_semantic_folium_map(shapefile_path, output_path)
                results["semantic_folium"] = {"map": m, "gdf": gdf, "path": output_path}
            else:
                print(
                    "Warning: Shapefile does not have semantic classification information."
                )
                print("Falling back to standard Folium visualization...")
                output_path = os.path.join(
                    output_dir, f"segments_folium_{timestamp}.html"
                )
                m, gdf = create_folium_map(shapefile_path, output_path)
                results["folium"] = {"map": m, "gdf": gdf, "path": output_path}

        elif visualization_type == "semantic_mask":
            if semantic_mask_path:
                output_path = os.path.join(output_dir, f"semantic_mask_{timestamp}.png")
                fig, ax = visualize_semantic_mask(semantic_mask_path, output_path)
                results["semantic_mask"] = {"fig": fig, "ax": ax, "path": output_path}
            else:
                print(
                    "Warning: Semantic mask path is required for semantic mask visualization."
                )
                print("No visualization was produced.")

    print("\nVisualization complete!")
    for viz_type, result in results.items():
        print(f"- {viz_type}: {result['path']}")
    print("=" * 50)

    return results

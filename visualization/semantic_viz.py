# visualization/semantic_viz.py
"""
Visualization functions for semantic segmentation results.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from rasterio.plot import show
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import needed modules
from pipeline.semantic_segmentation import LAND_COVER_CLASSES

# Define colors for each class
CLASS_COLORS = {
    "Background": "#000000",  # Black
    "Building": "#800000",  # Maroon
    "Road": "#8040A0",  # Purple
    "Water": "#0080FF",  # Blue
    "Vegetation": "#00FF00",  # Green
    "Bare Ground": "#D2B48C",  # Tan
    "Agricultural Land": "#FFFF00",  # Yellow
    "Industrial": "#FF0000",  # Red
    "Residential": "#FFA000",  # Orange
    "Commercial": "#00FFFF",  # Cyan
    "Unknown": "#808080",  # Gray
}


def plot_classified_segments(
    shapefile_path, output_path=None, title="Semantic Classification", figsize=None
):
    """
    Create a visualization of semantic classification results.

    Args:
        shapefile_path: Path to the shapefile with classifications
        output_path: Path to save the visualization
        title: Plot title
        figsize: Figure size tuple

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
        gdf: GeoDataFrame with classified segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Check if classification exists
    if "class" not in gdf.columns:
        print("Warning: No 'class' column found in the shapefile.")
        print("This may not be semantically classified data.")
        return None, None, gdf

    # Use default figure size if not provided
    if figsize is None:
        figsize = config.VISUALIZATION_PARAMS["figsize"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique classes
    unique_classes = gdf["class"].unique()

    # Create categorical color map
    cmap = {}
    for cls in unique_classes:
        cmap[cls] = CLASS_COLORS.get(
            cls, "#808080"
        )  # Default to gray if class not in colors

    # Plot each class separately to have a proper legend
    patches = []
    for cls_name in unique_classes:
        # Get only polygons of this class
        cls_gdf = gdf[gdf["class"] == cls_name]

        # Plot with the color for this class
        color = cmap[cls_name]
        cls_gdf.plot(
            ax=ax,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )

        # Create patch for legend
        patch = mpatches.Patch(color=color, label=cls_name)
        patches.append(patch)

    # Add legend
    ax.legend(handles=patches, loc="best", title="Land Cover Classes")

    # Add styling
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Classification visualization saved to {output_path}")

    return fig, ax, gdf


def plot_classified_with_image(
    shapefile_path,
    image_path,
    output_path=None,
    title="Semantic Classification with Image",
    figsize=None,
):
    """
    Create a visualization of semantic classification overlaid on the original image.

    Args:
        shapefile_path: Path to the shapefile with classifications
        image_path: Path to the original image
        output_path: Path to save the visualization
        title: Plot title
        figsize: Figure size tuple

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
        gdf: GeoDataFrame with classified segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Check if classification exists
    if "class" not in gdf.columns:
        print("Warning: No 'class' column found in the shapefile.")
        print("This may not be semantically classified data.")
        return None, None, gdf

    # Read the image
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs

        # If image has more than 3 channels, use only the first 3
        if image.shape[0] > 3:
            image = image[:3]

        # Convert to RGB format if needed
        if image.shape[0] in [1, 3]:  # Channels first format
            image = np.transpose(image, (1, 2, 0))

        # If grayscale, convert to RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            if len(image.shape) == 3:
                image = image[:, :, 0]
            image = np.stack([image, image, image], axis=2)

        # Normalize for display if not uint8
        if image.dtype != np.uint8:
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)
            image = (image * 255).astype(np.uint8)

    # Make sure GeoDataFrame has the right CRS
    if gdf.crs is None:
        print("Warning: Shapefile has no CRS information. Assuming same as image.")
        gdf.crs = crs
    elif gdf.crs != crs:
        print(f"Warning: CRS mismatch. Converting shapefile from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)

    # Use default figure size if not provided
    if figsize is None:
        figsize = config.VISUALIZATION_PARAMS["figsize"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image
    ax.imshow(image)

    # Get unique classes
    unique_classes = gdf["class"].unique()

    # Create categorical color map
    cmap = {}
    for cls in unique_classes:
        cmap[cls] = CLASS_COLORS.get(
            cls, "#808080"
        )  # Default to gray if class not in colors

    # Plot each class separately
    patches = []
    for cls_name in unique_classes:
        # Get only polygons of this class
        cls_gdf = gdf[gdf["class"] == cls_name]

        # Plot with the color for this class
        color = cmap[cls_name]
        cls_gdf.boundary.plot(
            ax=ax,
            color=color,
            linewidth=1.5,
        )

        # Create patch for legend
        patch = mpatches.Patch(color=color, label=cls_name)
        patches.append(patch)

    # Add legend
    ax.legend(handles=patches, loc="best", title="Land Cover Classes")

    # Add styling
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Classification with image visualization saved to {output_path}")

    return fig, ax, gdf


def create_semantic_plotly_map(
    shapefile_path, output_path=None, title="Interactive Semantic Classification"
):
    """
    Create an interactive Plotly map of semantic classification results.

    Args:
        shapefile_path: Path to the shapefile with classifications
        output_path: Path to save the HTML visualization
        title: Map title

    Returns:
        fig: Plotly figure
        gdf: GeoDataFrame with classified segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Check if classification exists
    if "class" not in gdf.columns:
        print("Warning: No 'class' column found in the shapefile.")
        print("This may not be semantically classified data.")
        return None, gdf

    # Convert to WGS84 if needed for proper display in Plotly
    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    # Get unique classes
    unique_classes = gdf["class"].unique()

    # Create categorical color map
    colors = {}
    for cls in unique_classes:
        colors[cls] = CLASS_COLORS.get(
            cls, "#808080"
        )  # Default to gray if class not in colors

    # Calculate centroid for initial view
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # Create figure
    fig = go.Figure()

    # Add each class as a separate layer
    for cls_name in unique_classes:
        # Get only polygons of this class
        cls_gdf = gdf[gdf["class"] == cls_name]

        if len(cls_gdf) == 0:
            continue

        # Add layer for this class
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=cls_gdf.__geo_interface__,
                locations=cls_gdf.index,
                z=cls_gdf.index,  # Using index just to get colors right
                colorscale=[[0, colors[cls_name]], [1, colors[cls_name]]],
                marker_opacity=0.7,
                marker_line_width=1,
                marker_line_color="black",
                showscale=False,
                name=cls_name,
                hovertemplate="<b>%{hovertext}</b><br>"
                + "Class: "
                + cls_name
                + "<br>"
                + "<extra></extra>",
                hovertext=cls_gdf.index.astype(str),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        mapbox=dict(
            style=config.VISUALIZATION_PARAMS["mapbox_style"],
            center=dict(lat=center_lat, lon=center_lon),
            zoom=config.VISUALIZATION_PARAMS["zoom"],
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=800,
        width=1000,
        legend=dict(
            title="Land Cover Classes", yanchor="top", y=0.99, xanchor="left", x=0.01
        ),
    )

    # Save if output path provided
    if output_path:
        fig.write_html(output_path)
        print(f"Interactive semantic classification map saved to {output_path}")

    return fig, gdf


def create_semantic_folium_map(
    shapefile_path, output_path=None, title="Interactive Semantic Classification"
):
    """
    Create an interactive Folium map of semantic classification results.

    Args:
        shapefile_path: Path to the shapefile with classifications
        output_path: Path to save the HTML visualization
        title: Map title

    Returns:
        m: Folium map
        gdf: GeoDataFrame with classified segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Check if classification exists
    if "class" not in gdf.columns:
        print("Warning: No 'class' column found in the shapefile.")
        print("This may not be semantically classified data.")
        return None, gdf

    # Convert to WGS84 for proper display in Folium
    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    # Calculate centroid for initial view
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=config.VISUALIZATION_PARAMS["zoom"],
        tiles="OpenStreetMap",
    )

    # Get unique classes
    unique_classes = sorted(gdf["class"].unique())

    # Create feature groups for each class
    feature_groups = {}
    for cls_name in unique_classes:
        feature_groups[cls_name] = folium.FeatureGroup(name=cls_name)

    # Add polygons to corresponding feature groups
    for idx, row in gdf.iterrows():
        cls_name = row["class"]
        color = CLASS_COLORS.get(cls_name, "#808080")

        # Create tooltip
        tooltip = folium.Tooltip(f"ID: {idx}<br>Class: {cls_name}")

        # Create GeoJson polygon
        geojson = folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, color=color: {
                "fillColor": color,
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            tooltip=tooltip,
        )

        # Add to appropriate feature group
        geojson.add_to(feature_groups[cls_name])

    # Add all feature groups to map
    for cls_name, fg in feature_groups.items():
        fg.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add title
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # Add legend as a custom HTML element
    legend_html = """
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 200px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                    padding: 10px; border-radius: 5px;">
        <h4 style="margin-top:0;">Land Cover Classes</h4>
    """

    for cls_name in unique_classes:
        color = CLASS_COLORS.get(cls_name, "#808080")
        legend_html += f"""
            <div>
                <span style="display:inline-block; width:12px; height:12px;
                        background-color:{color}; margin-right:5px;"></span>
                {cls_name}
            </div>
        """

    legend_html += """</div>"""

    m.get_root().html.add_child(folium.Element(legend_html))

    # Save if output path provided
    if output_path:
        m.save(output_path)
        print(f"Interactive semantic classification Folium map saved to {output_path}")

    return m, gdf


def visualize_semantic_mask(
    mask_path, output_path=None, title="Semantic Segmentation Mask", figsize=None
):
    """
    Visualize a semantic segmentation mask.

    Args:
        mask_path: Path to the semantic mask
        output_path: Path to save the visualization
        title: Plot title
        figsize: Figure size tuple

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
    """
    # Open the mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # Use default figure size if not provided
    if figsize is None:
        figsize = config.VISUALIZATION_PARAMS["figsize"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap
    unique_values = np.unique(mask)
    num_classes = len(unique_values)

    # Create color map for visualization
    colors = []
    for val in unique_values:
        if val >= 0 and val < len(LAND_COVER_CLASSES):
            class_name = LAND_COVER_CLASSES[val]
            rgb_color = mcolors.to_rgb(CLASS_COLORS.get(class_name, "#808080"))
        else:
            rgb_color = mcolors.to_rgb("#808080")  # Default to gray
        colors.append(rgb_color)

    # Create custom colormap
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, num_classes + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the mask
    im = ax.imshow(mask, cmap=cmap, norm=norm)

    # Create legend
    patches = []
    for val in unique_values:
        if val >= 0 and val < len(LAND_COVER_CLASSES):
            class_name = LAND_COVER_CLASSES[val]
        else:
            class_name = f"Class {val}"

        color = colors[np.where(unique_values == val)[0][0]]
        patch = mpatches.Patch(color=color, label=class_name)
        patches.append(patch)

    # Add legend
    ax.legend(handles=patches, loc="best", title="Land Cover Classes")

    # Add styling
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Semantic mask visualization saved to {output_path}")

    return fig, ax

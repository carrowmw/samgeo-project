# visualization/interactive_viz.py
"""
Interactive visualization functions for SAMGeo.
"""
import os
import plotly.graph_objects as go
import folium
import geopandas as gpd
from shapely.geometry import mapping
import json
import sys

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_plotly_map(
    shapefile_path, output_path=None, column=None, title="Interactive Segment Map"
):
    """
    Create an interactive Plotly map of segments.

    Args:
        shapefile_path: Path to the shapefile
        output_path: Path to save the HTML visualization
        column: Column to use for coloring
        title: Map title

    Returns:
        fig: Plotly figure
        gdf: GeoDataFrame with segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Convert to WGS84 if needed for proper display in Plotly
    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    # Choose coloring column
    if column is None:
        if "value" in gdf.columns:
            column = "value"
        else:
            column = gdf.index.name if gdf.index.name else "index"
            gdf[column] = gdf.index

    # Create color scale values
    if column in gdf.columns:
        color_values = gdf[column].tolist()
    else:
        color_values = list(range(len(gdf)))

    # Create GeoJSON for Plotly
    features = []
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        geo_dict = mapping(geometry)
        feature = {
            "type": "Feature",
            "id": idx,
            "properties": {"id": idx},
            "geometry": geo_dict,
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    # Calculate centroid for initial view
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # Create Plotly figure
    fig = go.Figure()

    # Add polygons
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=list(range(len(gdf))),
            z=color_values,
            colorscale=config.VISUALIZATION_PARAMS["colorscale"],
            marker_opacity=config.VISUALIZATION_PARAMS["alpha"],
            marker_line_width=1,
            showscale=True,
            hoverinfo="text",
            text=[
                f"ID: {idx}<br>Value: {val}"
                for idx, val in zip(gdf.index, color_values)
            ],
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
    )

    # Save if output path provided
    if output_path:
        fig.write_html(output_path)
        print(f"Interactive Plotly map saved to {output_path}")

    return fig, gdf


def create_folium_map(
    shapefile_path, output_path=None, column=None, title="Interactive Segment Map"
):
    """
    Create an interactive Folium map of segments.

    Args:
        shapefile_path: Path to the shapefile
        output_path: Path to save the HTML visualization
        column: Column to use for styling
        title: Map title

    Returns:
        m: Folium map
        gdf: GeoDataFrame with segments
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

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

    # Choose coloring column
    if column is None and "value" in gdf.columns:
        column = "value"

    # Define style function
    if column and column in gdf.columns:
        # Calculate color range
        vmin, vmax = gdf[column].min(), gdf[column].max()

        def style_function(feature):
            value = feature["properties"][column]
            # Normalize value between 0 and 1
            normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            # Generate a color from blue to red
            return {
                "fillColor": f"hsl({120 * (1 - normalized)}, 70%, 50%)",
                "color": "black",
                "weight": 1,
                "fillOpacity": config.VISUALIZATION_PARAMS["alpha"],
            }

        # Add tooltip fields
        tooltip_fields = [column]
        tooltip_aliases = [f"{column}:"]
    else:

        def style_function(feature):
            return {
                "fillColor": "blue",
                "color": "black",
                "weight": 1,
                "fillOpacity": config.VISUALIZATION_PARAMS["alpha"],
            }

        # No tooltip fields if no column specified
        tooltip_fields = []
        tooltip_aliases = []

    # Add GeoJSON layer
    folium.GeoJson(
        gdf,
        name="Segments",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields, aliases=tooltip_aliases, localize=True
        ),
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add title
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # Save if output path provided
    if output_path:
        m.save(output_path)
        print(f"Interactive Folium map saved to {output_path}")

    return m, gdf

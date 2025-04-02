# SAMGeo: Segment Anything Model for Geospatial Data

A modular Python package for segmenting satellite imagery using Meta's Segment Anything Model (SAM).

## Features

- Process satellite imagery for segmentation
- Apply SAM to identify features in geospatial data
- Convert raster segmentation masks to vector formats
- Visualize results with multiple methods
- Modular architecture with centralized configuration

## Project Structure

```
samgeo_project/
│
├── config.py                # Centralized configuration
├── main.py                  # Main entry point
│
├── pipeline/
│   ├── __init__.py
│   ├── processor.py         # Image processing module
│   ├── segmentation.py      # SAM segmentation module
│   ├── vectorization.py     # Raster to vector conversion
│   └── pipeline.py          # Complete pipeline orchestration
│
├── visualization/
│   ├── __init__.py
│   ├── basic_viz.py         # Matplotlib based visualizations
│   ├── interactive_viz.py   # Plotly and Folium visualizations
│   └── visualization.py     # Visualization orchestration
│
└── utils/
    ├── __init__.py
    ├── geo_utils.py         # Geospatial utilities
    └── io_utils.py          # I/O utilities
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/samgeo_project.git
   cd samgeo_project
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n samgeo python=3.9
   conda activate samgeo

   # Install core dependencies
   pip install samgeo geopandas rasterio matplotlib plotly folium
   ```

3. Download the SAM model checkpoint:
   ```bash
   mkdir -p models
   # Download the model from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   # Place it in the models directory
   ```

## Usage

### Running from the Command Line

#### Run the Full Pipeline (Segmentation + Visualization)

```bash
python main.py --image path/to/satellite_image.tif
```

#### Run Only the Pipeline

```bash
python main.py --mode pipeline --image path/to/satellite_image.tif
```

#### Run Only Visualization

```bash
python main.py --mode visualize --shapefile path/to/segments.shp --viz-type plotly
```

#### Available Visualization Types

- `basic`: Simple matplotlib plot
- `with_image`: Overlay segments on original image
- `plotly`: Interactive Plotly map
- `folium`: Interactive Folium web map
- `all`: Generate all visualization types

### Using as a Python Module

#### Running the Pipeline

```python
from pipeline import run_pipeline

# Run the pipeline
gdf, output_files = run_pipeline(
    image_path="path/to/satellite_image.tif"
)

print(f"Output files: {output_files}")
```

#### Visualizing Results

```python
from visualization import run_visualization

# Visualize the results
results = run_visualization(
    shapefile_path="path/to/shapefile.shp",
    visualization_type="plotly"
)
```

## Configuration

You can customize the pipeline behavior by modifying `config.py`. Key settings include:

- `SCALE_FACTOR`: Image scaling factor (default: 0.5)
- `SAM_PARAMS`: Parameters for the SAM model
- `VECTORIZATION_PARAMS`: Parameters for raster to vector conversion
- `VISUALIZATION_PARAMS`: Parameters for visualization

## Examples

### Basic Pipeline Example

```bash
# Run segmentation and save outputs with custom names
python main.py --image data/satellite.tif --mask outputs/mask.tif --vector outputs/segments.shp
```

### Interactive Visualization

```bash
# Generate an interactive Plotly visualization
python main.py --mode visualize --shapefile outputs/segments.shp --viz-type plotly
```

### Combined with Original Image

```bash
# Overlay segments on the original image
python main.py --mode visualize --shapefile outputs/segments.shp --viz-type with_image --orig-image data/satellite.tif
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Segment Anything Model (SAM)](https://segment-anything.com/)
- [SAMGeo](https://github.com/opengeos/samgeo)
- [GeoPandas](https://geopandas.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
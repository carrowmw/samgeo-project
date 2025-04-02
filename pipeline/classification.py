# pipeline/classification.py


import numpy as np
import rasterio
import torchvision
import torch

from pipeline.processor import preprocess_image


def classify_segments(gdf, image_path, model_path=None):
    """
    Classify segmented polygons into semantic categories.

    Args:
        gdf: GeoDataFrame with segmented polygons
        image_path: Path to the original image
        model_path: Path to the classification model weights

    Returns:
        gdf: GeoDataFrame with added classification column
    """
    # Load the original image
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform

        # Create RGB image if needed
        if image.shape[0] in [1, 3, 4]:  # Channels first format
            image = np.transpose(image, (1, 2, 0))

        # If more than 3 channels, take only the first 3
        if len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, :3]

    # Load the classification model
    if model_path is None:
        # Use a pre-trained model like ResNet50
        model = torchvision.models.resnet50(pretrained=True)
        # Replace the final layer to match our classes
        num_classes = 10  # Example: 10 classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        # Load custom model
        model = torch.load(model_path)

    # Set model to evaluation mode
    model.eval()

    # Create a class label lookup
    class_labels = {
        0: "Building",
        1: "Road",
        2: "Water",
        3: "Vegetation",
        4: "Bare ground",
        # Add more classes as needed
    }

    # Classify each polygon
    classifications = []
    for idx, row in gdf.iterrows():
        # Get the bounding box of the polygon
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)

        # Convert to pixel coordinates
        pixel_bounds = rasterio.transform.rowcol(
            transform, [bounds[0], bounds[2]], [bounds[1], bounds[3]]
        )

        # Extract the region of interest
        minrow = max(0, min(pixel_bounds[0]))
        maxrow = min(image.shape[0], max(pixel_bounds[0]))
        mincol = max(0, min(pixel_bounds[1]))
        maxcol = min(image.shape[1], max(pixel_bounds[1]))

        # Create a mask for the polygon
        mask = rasterio.features.rasterize(
            [(row.geometry, 1)],
            out_shape=(maxrow - minrow, maxcol - mincol),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Extract the image region
        roi = image[minrow:maxrow, mincol:maxcol]

        # Apply mask to get just the polygon area
        masked_roi = roi * mask[:, :, np.newaxis]

        # Preprocess for classification model
        input_tensor = preprocess_image(masked_roi)

        # Classify
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class
        _, predicted = torch.max(output, 1)
        class_id = predicted.item()
        class_name = class_labels.get(class_id, f"Class_{class_id}")

        classifications.append(class_name)

    # Add classifications to the GeoDataFrame
    gdf["class"] = classifications

    return gdf

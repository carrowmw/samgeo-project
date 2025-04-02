# pipeline/semantic_segmentation.py
"""
Semantic segmentation module for SAMGeo pipeline.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import shape, box
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Define the land cover classes for satellite imagery
LAND_COVER_CLASSES = {
    0: "Background",
    1: "Building",
    2: "Road",
    3: "Water",
    4: "Vegetation",
    5: "Bare Ground",
    6: "Agricultural Land",
    7: "Industrial",
    8: "Residential",
    9: "Commercial",
}


def load_semantic_model(model_path=None, num_classes=len(LAND_COVER_CLASSES)):
    """
    Load the DeepLabV3+ semantic segmentation model.

    Args:
        model_path: Path to pre-trained model weights (ignored in this version)
        num_classes: Number of semantic classes

    Returns:
        model: The loaded model
    """
    # Initialize the DeepLabV3+ model
    print("Initializing DeepLabV3+ with pre-trained weights")
    # Load pre-trained model on COCO
    model = deeplabv3_resnet101(pretrained=True, progress=True)

    # Modify the classifier to match our number of classes
    model.classifier[4] = torch.nn.Conv2d(
        256, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )

    # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model


def segment_classify_image(model, image_path, output_path=None):
    """
    Perform semantic segmentation on the entire image.

    Args:
        model: The semantic segmentation model
        image_path: Path to input image
        output_path: Path to save the classified image

    Returns:
        semantic_mask: Semantic classification mask
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Open the image
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile
        transform = src.transform

        # Create RGB image for the model
        if image.shape[0] in [1, 3, 4]:  # Channels first format
            image = np.transpose(image, (1, 2, 0))

        # If more than 3 channels, take only the first 3
        if len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, :3]

        # If fewer than 3 channels, duplicate channels
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)
        elif image.shape[2] == 2:
            # Add a third channel
            zeros = np.zeros_like(image[:, :, 0:1])
            image = np.concatenate([image, zeros], axis=2)

    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0

    # Apply preprocessing
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Get image dimensions
    height, width = image.shape[:2]

    # Process image in tiles if it's large
    tile_size = 512  # Size of each tile
    overlap = 64  # Overlap between tiles

    # Initialize empty mask for the entire image
    semantic_mask = np.zeros((height, width), dtype=np.int64)

    # Process image in tiles
    for y in tqdm(range(0, height, tile_size - overlap), desc="Processing tiles"):
        for x in range(0, width, tile_size - overlap):
            # Define tile coordinates
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            y_start = y
            x_start = x

            # Extract tile
            tile = image[y_start:y_end, x_start:x_end]

            # Skip tiles that are too small
            if tile.shape[0] < 64 or tile.shape[1] < 64:
                continue

            # Preprocess tile
            input_tensor = preprocess(tile)
            input_batch = input_tensor.unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                output = model(input_batch)["out"][0]
                output = F.interpolate(
                    output.unsqueeze(0),
                    size=(tile.shape[0], tile.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                # Get predictions
                _, predictions = torch.max(output, 0)
                predictions = predictions.cpu().numpy()

            # Calculate valid area (exclude overlap except at edges)
            valid_y_start = 0 if y == 0 else overlap // 2
            valid_x_start = 0 if x == 0 else overlap // 2
            valid_y_end = (
                y_end - y_start if y + tile_size >= height else tile_size - overlap // 2
            )
            valid_x_end = (
                x_end - x_start if x + tile_size >= width else tile_size - overlap // 2
            )

            # Update semantic mask with valid area
            y_mask_start = y_start + valid_y_start
            y_mask_end = y_start + valid_y_end
            x_mask_start = x_start + valid_x_start
            x_mask_end = x_start + valid_x_end

            semantic_mask[y_mask_start:y_mask_end, x_mask_start:x_mask_end] = (
                predictions[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            )

    # Save semantic mask if output path is provided
    if output_path:
        # Update profile for the output
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(semantic_mask.astype(rasterio.uint8), 1)

        # Also save a colored version for visualization
        color_output_path = output_path.replace(".tif", "_colored.png")
        visualize_semantic_mask(semantic_mask, color_output_path)

    return semantic_mask, transform


def classify_segments(gdf, semantic_mask, transform, model=None, image_path=None):
    """
    Assign semantic labels to segmented polygons based on majority class.

    Args:
        gdf: GeoDataFrame with segmented polygons
        semantic_mask: Semantic classification mask array
        transform: Geospatial transform for the mask
        model: Optional semantic model (if mask not provided)
        image_path: Path to original image (needed if model is provided)

    Returns:
        gdf: GeoDataFrame with added classification column
    """
    # If semantic mask not provided, generate it using the model
    if semantic_mask is None and model is not None and image_path is not None:
        semantic_mask, transform = segment_classify_image(model, image_path)

    if semantic_mask is None:
        raise ValueError(
            "Either semantic_mask or both model and image_path must be provided"
        )

    # Initialize classification lists
    classifications = []
    class_probabilities = []

    # Get mask shape
    mask_height, mask_width = semantic_mask.shape

    # Process each polygon
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Classifying segments"):
        # Get the polygon geometry
        geometry = row.geometry

        # Create a mask for the polygon
        minx, miny, maxx, maxy = geometry.bounds

        # Convert bounds to pixel coordinates
        ul_row, ul_col = rasterio.transform.rowcol(transform, minx, maxy)
        lr_row, lr_col = rasterio.transform.rowcol(transform, maxx, miny)

        # Ensure bounds are within image
        ul_row = max(0, min(mask_height - 1, ul_row))
        ul_col = max(0, min(mask_width - 1, ul_col))
        lr_row = max(0, min(mask_height - 1, lr_row))
        lr_col = max(0, min(mask_width - 1, lr_col))

        # Skip if bounds are invalid
        if ul_row >= lr_row or ul_col >= lr_col:
            classifications.append("Unknown")
            class_probabilities.append({})
            continue

        # Create binary mask for the polygon
        polygon_mask = rasterize(
            [(geometry, 1)],
            out_shape=(lr_row - ul_row, lr_col - ul_col),
            transform=rasterio.transform.from_bounds(
                minx, miny, maxx, maxy, lr_col - ul_col, lr_row - ul_row
            ),
            fill=0,
            dtype=np.uint8,
        )

        # Extract the region of interest from the semantic mask
        roi = semantic_mask[ul_row:lr_row, ul_col:lr_col]

        # Skip if ROI is empty
        if roi.size == 0 or polygon_mask.size == 0:
            classifications.append("Unknown")
            class_probabilities.append({})
            continue

        # Apply the polygon mask
        masked_classes = roi[polygon_mask > 0]

        # Skip if no pixels are within the polygon
        if masked_classes.size == 0:
            classifications.append("Unknown")
            class_probabilities.append({})
            continue

        # Count occurrences of each class
        class_counts = {}
        for class_id in np.unique(masked_classes):
            count = np.sum(masked_classes == class_id)
            class_counts[int(class_id)] = count

        # Find majority class
        if class_counts:
            majority_class_id = max(class_counts, key=class_counts.get)
            majority_class_name = LAND_COVER_CLASSES.get(
                majority_class_id, f"Class_{majority_class_id}"
            )

            # Calculate class probabilities
            total_pixels = sum(class_counts.values())
            probs = {
                LAND_COVER_CLASSES.get(k, f"Class_{k}"): v / total_pixels
                for k, v in class_counts.items()
            }
        else:
            majority_class_name = "Unknown"
            probs = {}

        classifications.append(majority_class_name)
        class_probabilities.append(probs)

    # Add classifications to GeoDataFrame
    gdf["class"] = classifications

    # Add probabilities for each class as separate columns
    for class_name in LAND_COVER_CLASSES.values():
        gdf[f"prob_{class_name}"] = [
            probs.get(class_name, 0) for probs in class_probabilities
        ]

    # Add confidence (probability of assigned class)
    gdf["confidence"] = [
        probs.get(class_name, 0) if probs else 0
        for class_name, probs in zip(classifications, class_probabilities)
    ]

    return gdf


def visualize_semantic_mask(semantic_mask, output_path=None):
    """
    Create a colored visualization of the semantic mask.

    Args:
        semantic_mask: Semantic classification mask
        output_path: Path to save visualization

    Returns:
        colored_mask: Colored visualization of the semantic mask
    """
    # Define colors for each class
    colors = {
        0: [0, 0, 0],  # Background - Black
        1: [128, 0, 0],  # Building - Maroon
        2: [128, 64, 128],  # Road - Purple
        3: [0, 128, 255],  # Water - Blue
        4: [0, 255, 0],  # Vegetation - Green
        5: [210, 180, 140],  # Bare Ground - Tan
        6: [255, 255, 0],  # Agricultural Land - Yellow
        7: [255, 0, 0],  # Industrial - Red
        8: [255, 160, 0],  # Residential - Orange
        9: [0, 255, 255],  # Commercial - Cyan
    }

    # Create RGB image
    height, width = semantic_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill colors
    for class_id, color in colors.items():
        colored_mask[semantic_mask == class_id] = color

    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))

    return colored_mask


def train_semantic_model(
    training_data_path, output_model_path, num_epochs=10, batch_size=16
):
    """
    Train a DeepLabV3+ model for semantic segmentation.

    Args:
        training_data_path: Path to training data directory
        output_model_path: Path to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        model: Trained model
    """
    from torch.utils.data import Dataset, DataLoader

    # Define custom dataset
    class SatelliteSegmentationDataset(Dataset):
        def __init__(self, data_path, transform=None, target_transform=None):
            self.data_path = data_path
            self.transform = transform
            self.target_transform = target_transform

            # List image files
            self.image_files = [
                f
                for f in os.listdir(os.path.join(data_path, "images"))
                if f.endswith((".png", ".jpg", ".tif"))
            ]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            # Load image
            img_name = self.image_files[idx]
            img_path = os.path.join(self.data_path, "images", img_name)

            # Load image based on extension
            if img_path.endswith(".tif"):
                with rasterio.open(img_path) as src:
                    image = src.read()
                    if image.shape[0] in [1, 3, 4]:  # Channels first format
                        image = np.transpose(image, (1, 2, 0))
            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Ensure image has 3 channels
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=2)
            elif image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)
            elif image.shape[2] > 3:
                image = image[:, :, :3]

            # Load mask (either .png or .tif with same base name)
            mask_name = img_name.split(".")[0] + ".png"
            mask_path = os.path.join(self.data_path, "masks", mask_name)

            if not os.path.exists(mask_path):
                mask_name = img_name.split(".")[0] + ".tif"
                mask_path = os.path.join(self.data_path, "masks", mask_name)

            # Load mask
            if mask_path.endswith(".tif"):
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                mask = self.target_transform(mask)
            else:
                mask = torch.from_numpy(mask).long()

            return image, mask

    # Set up transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset and loaders
    dataset = SatelliteSegmentationDataset(training_data_path, transform=transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Create model
    model = deeplabv3_resnet101(pretrained=True)

    # Modify output layer for our classes
    num_classes = len(LAND_COVER_CLASSES)
    model.classifier[4] = torch.nn.Conv2d(
        256, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

                val_loss += loss.item()

        # Print metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, output_model_path)
            print(f"Saved best model to {output_model_path}")

    # Load best model
    model = torch.load(output_model_path)

    return model


def run_semantic_segmentation(
    gdf, image_path, model_path=None, output_mask_path=None, output_path=None
):
    """
    Run semantic segmentation on segmented polygons.

    Args:
        gdf: GeoDataFrame with segmented polygons
        image_path: Path to the original image
        model_path: Path to the semantic segmentation model
        output_mask_path: Path to save the semantic mask
        output_path: Path to save the classified polygons

    Returns:
        classified_gdf: GeoDataFrame with semantic classes
    """
    # Load model
    model = load_semantic_model(model_path)

    # Generate semantic mask
    if output_mask_path:
        semantic_mask, transform = segment_classify_image(
            model, image_path, output_mask_path
        )
    else:
        # Generate temporary mask path
        temp_mask_path = os.path.join(
            config.DEBUG_DIR,
            f"temp_semantic_mask_{os.path.basename(image_path).split('.')[0]}.tif",
        )
        semantic_mask, transform = segment_classify_image(
            model, image_path, temp_mask_path
        )

    # Classify segments based on the semantic mask
    classified_gdf = classify_segments(gdf, semantic_mask, transform)

    # Save to file if output path provided
    if output_path:
        classified_gdf.to_file(output_path)
        print(f"Saved classified segments to {output_path}")

    return classified_gdf

# utils/io_utils.py
"""
Input/output utility functions for SAMGeo.
"""
import os
import json
import shutil
import zipfile
import glob
import sys
from datetime import datetime

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def save_metadata(output_path, metadata):
    """
    Save metadata to a JSON file.

    Args:
        output_path: Path to save the JSON file
        metadata: Dictionary with metadata
    """
    # Add timestamp
    metadata["timestamp"] = datetime.now().isoformat()

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return output_path


def load_metadata(metadata_path):
    """
    Load metadata from a JSON file.

    Args:
        metadata_path: Path to the JSON file

    Returns:
        metadata: Dictionary with metadata
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def create_zipfile(input_paths, output_path):
    """
    Create a ZIP file from multiple input files/directories.

    Args:
        input_paths: List of paths to include in the ZIP file
        output_path: Path to save the ZIP file
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for input_path in input_paths:
            if os.path.isdir(input_path):
                for root, _, files in os.walk(input_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(
                            file_path, os.path.dirname(input_path)
                        )
                        zipf.write(file_path, arcname)
            else:
                arcname = os.path.basename(input_path)
                zipf.write(input_path, arcname)

    return output_path


def find_latest_file(directory, pattern="*.shp"):
    """
    Find the latest file matching a pattern in a directory.

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match

    Returns:
        latest_file: Path to the latest file
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None

    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)

    return latest_file


def setup_logging(log_dir=None, log_level="INFO"):
    """
    Set up logging for the application.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level

    Returns:
        logger: Logger object
    """
    import logging

    # Set default log directory if not provided
    if log_dir is None:
        log_dir = os.path.join(config.OUTPUT_DIR, "logs")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set up logger
    logger = logging.getLogger("samgeo")
    logger.setLevel(getattr(logging, log_level))

    # Create a file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"samgeo_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)

    # Create a console handler
    console_handler = logging.StreamHandler()

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_run_metadata(
    image_path,
    mask_path,
    vector_path,
    geo_vector_path,
    processing_time,
    parameters=None,
):
    """
    Create metadata for a pipeline run.

    Args:
        image_path: Path to the input image
        mask_path: Path to the segmentation mask
        vector_path: Path to the vector data
        geo_vector_path: Path to the vector data with geographic coordinates
        processing_time: Processing time in seconds
        parameters: Dictionary with processing parameters

    Returns:
        metadata: Dictionary with run metadata
    """
    # Get basic file info
    image_size = os.path.getsize(image_path) if os.path.exists(image_path) else None
    mask_size = os.path.getsize(mask_path) if os.path.exists(mask_path) else None
    vector_size = os.path.getsize(vector_path) if os.path.exists(vector_path) else None
    geo_vector_size = (
        os.path.getsize(geo_vector_path) if os.path.exists(geo_vector_path) else None
    )

    # Create metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "files": {
            "image": {"path": image_path, "size": image_size},
            "mask": {"path": mask_path, "size": mask_size},
            "vector": {"path": vector_path, "size": vector_size},
            "geo_vector": {"path": geo_vector_path, "size": geo_vector_size},
        },
        "processing_time": processing_time,
        "parameters": parameters
        or {
            "scale": config.SCALE_IMAGE,
            "scale_factor": config.SCALE_FACTOR,
            "sam_params": config.SAM_PARAMS,
            "vectorization_params": config.VECTORIZATION_PARAMS,
        },
    }

    return metadata

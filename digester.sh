#!/bin/bash

# digester.sh - Script to ingest codebase while excluding large files and data files
# Dependencies: gitingest, nbstripout

set -e  # Exit on error

# Configuration
MAX_FILE_SIZE_KB=500  # Set maximum file size to 500 KB
MAX_FILE_SIZE_BYTES=$((MAX_FILE_SIZE_KB * 1024))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$PROJECT_ROOT/samgeo_project/digested_samgeo_$(date +%Y%m%d).txt"

# Check if gitingest is installed
if ! command -v gitingest &> /dev/null; then
    echo "Error: gitingest is not installed. Please install it first."
    echo "Install with: pip install gitingest"
    exit 1
fi

# Check if nbstripout is installed
if ! command -v nbstripout &> /dev/null; then
    echo "Warning: nbstripout is not installed. Notebooks will not be processed."
    echo "Consider installing with: pip install nbstripout"
    PROCESS_NOTEBOOKS=false
else
    PROCESS_NOTEBOOKS=true
fi

# Process notebooks if nbstripout is available
if [ "$PROCESS_NOTEBOOKS" = true ]; then
    echo "Processing notebooks with nbstripout..."
    find "$SCRIPT_DIR" -name "*.ipynb" -exec nbstripout {} \;
fi

echo "Starting codebase ingestion from gnn_package directory..."
echo "- Max file size: ${MAX_FILE_SIZE_KB}KB"
echo "- Output will be saved to: ${OUTPUT_FILE}"

# Run gitingest on the gnn_package directory
gitingest "$SCRIPT_DIR" \
    -s "${MAX_FILE_SIZE_BYTES}" \
    --exclude-pattern="*.pkl" \
    --exclude-pattern="*.npy" \
    --exclude-pattern="*.csv" \
    --exclude-pattern="*.parquet" \
    --exclude-pattern="*.json" \
    --exclude-pattern="*.gz" \
    --exclude-pattern="*.zip" \
    --exclude-pattern="*.tar" \
    --exclude-pattern="*.h5" \
    --exclude-pattern="*.hdf5" \
    --exclude-pattern="*.pyc" \
    --exclude-pattern="__pycache__/" \
    --exclude-pattern=".ipynb_checkpoints/" \
    --exclude-pattern="cache/" \
    --exclude-pattern="*/cache/*" \
    --exclude-pattern="*.so" \
    --exclude-pattern="*.o" \
    --exclude-pattern="*.a" \
    --exclude-pattern="*.dll" \
    --exclude-pattern="*.geojson" \
    --exclude-pattern="*.shp" \
    --exclude-pattern="*.shx" \
    --exclude-pattern="*.dbf" \
    --exclude-pattern="*.prj" \
    --exclude-pattern="*.cpg" \
    --exclude-pattern="*.pth" \
    --exclude-pattern="*.pt" \
    --exclude-pattern="*.ckpt" \
    --exclude-pattern="*.bin" \
    --exclude-pattern="*.png" \
    --exclude-pattern="*.jpg" \
    --exclude-pattern="*.jpeg" \
    --exclude-pattern="*.gif" \
    --exclude-pattern="*.svg" \
    --exclude-pattern="*.ico" \
    --exclude-pattern="*.pdf" \
    --exclude-pattern="*.tif" \
    --exclude-pattern="*.html" \
    --output="$OUTPUT_FILE"

echo "Nom nom, digestion complete! Output saved to $OUTPUT_FILE"
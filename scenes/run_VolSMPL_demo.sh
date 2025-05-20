#!/bin/bash

# Define variables
CONDA_ENV="DART"
OUTPUT_DIR="../bin"
OUTPUT_VIDEO="visualization.mp4"
IMAGE_DIR="../bin/imgs"
IMAGE_PATTERN="visualization_%d.png"

# Activate the Conda environment
echo "Activating Conda environment: $CONDA_ENV"
conda init
conda activate $CONDA_ENV

# Check and install necessary Python packages
echo "Checking and installing necessary Python packages..."
pip install numpy torch matplotlib trimesh smplx ffmpeg-python VolumetricSMPL

# Run the Python script
echo "RUN Python script ..."
python3 test_VolSMPL.py --bm_dir_path ../../iDART/data/smplx_lockedhead_20230207/models_lockedhead/ --model_type smplx --VISUALIZE

# Check if images were generated
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory '$IMAGE_DIR' not found. Ensure the Python script generates images in this directory."
    exit 1
fi

# Create a video from the images using ffmpeg
echo "Creating MP4 video from images..."
ffmpeg -framerate 8 -i "$IMAGE_DIR/$IMAGE_PATTERN" -c:v libx264 -pix_fmt yuv420p "$OUTPUT_DIR/$OUTPUT_VIDEO"

# Check if the video was created successfully
if [ -f "$OUTPUT_VIDEO" ]; then
    echo "Video created successfully: $OUTPUT_VIDEO"
else
    echo "Error: Failed to create video."
    exit 1
fi

# Deactivate the Conda environment
echo "Script completed successfully."
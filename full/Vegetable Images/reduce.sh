#!/bin/bash

# Define the source parent directory path
SOURCE_PARENT_DIR="./test"

# Define the target directory path
TARGET_DIR="./testRed"

# Loop through each folder in the parent directory
for SOURCE_FOLDER in "${SOURCE_PARENT_DIR}"/*; do
  if [ -d "${SOURCE_FOLDER}" ]; then # Check if it's a directory
    # Extract the folder name
    FOLDER_NAME=$(basename "${SOURCE_FOLDER}")
    
    # Define the target folder path
    TARGET_FOLDER="${TARGET_DIR}/${FOLDER_NAME}"

    # Create the target folder if it doesn't already exist
    mkdir -p "${TARGET_FOLDER}"

    # Randomly select 200 images from the source folder and copy them to the target folder
    find "${SOURCE_FOLDER}" -type f -name '*.jpg' | gshuf -n 200 | xargs -I {} cp {} "${TARGET_FOLDER}/"
  fi
done

echo "Process completed!"


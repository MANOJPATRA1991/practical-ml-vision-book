#!/bin/bash

# This script runs an Apache Beam data processing pipeline (jpeg_to_tfrecord_tft.py)
# to convert flower JPEG images from CSV file listings into TFRecord files,
# optionally resizing them, and storing the outputs in a Google Cloud Storage bucket or locally.

# Get the current GCP project and build a bucket name using the project name.
PROJECT=$(gcloud config get-value project)
BUCKET=${PROJECT}-mlvision   # You may need to change this to your actual bucket name.

# Set input and output locations for cloud execution.
INPUT=gs://practical-ml-vision-book-data/flowers_5_jpeg/flower_photos/all_data.csv
OUTPUT=gs://${BUCKET}/data/flowers_tftransform

# The following lines allow you to run the pipeline locally by processing just the first 100 records.
# If you want to run locally, these lines will extract the top 100 lines from the CSV to a local file.
# The INPUT and OUTPUT variables are then changed to use the local file and a local output directory.
gsutil cat $INPUT | head -100 > /tmp/top.csv
INPUT=/tmp/top.csv
OUTPUT=./flower_tftransform

# Print out which files will be used as input and output.
echo "INPUT=$INPUT OUTPUT=$OUTPUT"

# Run the jpeg_to_tfrecord_tft.py module, which creates TFRecord files from the JPEGs listed in the CSV.
# --all_data: Input CSV file of image locations and labels
# --labels_file: File containing the list of possible labels (one per line)
# --project_id: Google Cloud project ID
# --output_dir: Output location for the TFRecord files
# --resize: Desired image size as HEIGHT,WIDTH
python3 -m jpeg_to_tfrecord_tft \
       --all_data $INPUT \
       --labels_file gs://practical-ml-vision-book-data/flowers_5_jpeg/flower_photos/dict.txt \
       --project_id $PROJECT \
       --output_dir $OUTPUT \
       --resize '448,448'

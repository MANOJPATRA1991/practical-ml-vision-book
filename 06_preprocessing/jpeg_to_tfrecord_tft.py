#!/usr/bin/env python3

# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

r"""
EXPLANATION OF THE CODE:

This script implements an Apache Beam pipeline to preprocess a set of JPEG images and convert them into TFRecord files for use in TensorFlow training or serving. The pipeline can operate on images stored locally or in Google Cloud Storage (GCS), and the whole process is scalable via either running locally or submitting as a Dataflow job on Google Cloud.

**Main Functions and Flow:**

1. **Argument Parsing:**  
   The script takes a CSV file listing images (and their labels), a labels file (one label per line), project ID, output directory, Dataflow runner region, resize settings, and other parameters. All are parsed using Python's `argparse`.

2. **Setup and Output Directory:**  
   Output directory is cleared (to avoid confusion with remnants of previous runs) either locally or using `gsutil` for GCS.

3. **Image and Label Reading:**  
   The CSV is read using TFX's `CsvTFXIO` class. Each row provides a file path and a label string.
   The label list is also read in and used to map string labels into integer labels.

4. **Beam Pipeline:**
   - Each image filename is read, and its raw bytes are loaded.
   - Each image is assigned an integer label, according to the label list.
   - Images and labels are bundled into dictionaries.

5. **Preprocessing (with Tensorflow Transform, TFT):**  
   - Each image is decoded from JPEG bytes.
   - The image is converted to float, resized and padded (if needed) to the specified height and width.
   - The result is returned as a dictionary: { "image", "label", "label_int" }.

6. **Train/Validation/Test Split:**  
   Each processed record is randomly assigned to either train (80%), valid (10%), or test (10%).

7. **Writing to TFRecord:**  
   Each split is written to its own set of TFRecord files (train: 16 shards; valid/test: 2 shards each), gzipped. Beam manages sharding and sinks.

8. **Exporting the Preprocessing Graph:**  
   The transform graph (all preprocessing steps above) is saved as a SavedModel directory for consistent application at inference or for use in future runs.

**Key Helper Functions:**
- `_string_feature`, `_int64_feature`, `_float_feature`: Utility methods for TFRecord features (good for manually creating records, but not used directly as the coder handles serialization).
- `decode_image`: Decodes JPEG bytes into a TensorFlow tensor.
- `assign_record_to_split`: Assigns each record randomly to train/valid/test splits.
- `write_records`: Filters and writes records for a specific split.

**Canonically used with the following kinds of command lines:**
```
python3 -m jpeg_to_tfrecord_tft.py \
    --all_data gs://bucket/data.csv \
    --labels_file gs://bucket/labels.txt \
    --project_id my-gcp-project \
    --output_dir gs://bucket/tfrecords \
    --resize 448,448
```

**Typical File Formats:**
- **CSV**: Each row has 'image-file-path,label'
- **Labels File**: Each line is a string label.

This pipeline makes it easy to prepare huge image datasets for TensorFlow training with minimal manual effort and standardizes preprocessing for reliable, repeatable machine learning processes.
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import apache_beam as beam
import tensorflow as tf
import numpy as np
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tfx_bsl.public import tfxio

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 448, 448, 3
LABELS = []

from typing import Any
from typing import Iterator, Tuple, Dict

def _string_feature(value: str) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _int64_feature(value: list[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value: list[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def decode_image(img_bytes: Any) -> tf.Tensor:
    IMG_CHANNELS = 3
    return tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)

def assign_record_to_split(rec: dict) -> tuple[str, dict]:
    rnd = np.random.rand()
    if rnd < 0.8:
        return ('train', rec)
    if rnd < 0.9:
        return ('valid', rec)
    return ('test', rec)

def yield_records_for_split(x: Tuple[str, dict], desired_split: str) -> Iterator[dict]:
    split, rec = x
    # print(split, desired_split, split == desired_split)
    if split == desired_split:
        yield rec

def write_records(OUTPUT_DIR, splits, split):
    # same 80:10:10 split
    # The flowers dataset takes about 1GB, so 20 files means 50MB each
    
    # This function writes the preprocessed image records for a given split ('train', 'valid', or 'test')
    # into TFRecord files using Apache Beam. The number of output shards is set higher for the training split
    # (16 files) to improve parallelism in downstream training, and lower for validation and test (2 files each).
    # The input 'splits' PCollection is filtered to only include records for the desired split using FlatMap.
    # The filtered records are then written to TFRecord files (gzipped) under OUTPUT_DIR, with separate subfolders
    # for each split. This results in compressed TFRecord files ready for efficient use with TensorFlow.
    
    # nshards determines the number of output files ("shards") the TFRecord data is split into.
    # Sharding improves parallel read/write performance and efficiency during training and evaluation.
    # By using more shards for large splits (e.g., training) and fewer for smaller splits (validation/test), 
    # we balance throughput and file management. Each shard will contain a roughly equal portion of 
    # the total examples for that split.
    nshards = 16 if (split == 'train') else 2
    _ = (splits
         | 'only_{}'.format(split) >> beam.FlatMap(
             lambda x: yield_records_for_split(x, split))
         | 'write_{}'.format(split) >> beam.io.tfrecordio.WriteToTFRecord(
             os.path.join(OUTPUT_DIR, split),
             file_name_suffix='.gz', num_shards=nshards)
        )

def decode_image(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
    return img

def tft_preprocess(img_record): 
    # tft_preprocess gets a batch, but decode_jpeg can only read individual files
    
    # Decode each image in the input batch from its JPEG-encoded bytes.
    # tf.map_fn applies the decode_image function to each element in the 'img_bytes' field.
    # The function expects a batch of image bytes ('img_bytes' may be a tensor of strings).
    # fn_output_signature=tf.float32 specifies that each output will be a tensor of type float32.
    img = tf.map_fn(decode_image, img_record['img_bytes'],
                    fn_output_signature=tf.float32)
    
    # Convert the images to float32 and rescale the pixel values to the [0, 1] range.
    # This ensures that downstream models receive float data instead of raw uint8s.
    img = tf.image.convert_image_dtype(img, tf.float32) # [0,1]

    # Resize images to the desired fixed size (IMG_HEIGHT, IMG_WIDTH), 
    # adding padding as needed to maintain aspect ratio.
    img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)

    # Return a dictionary mapping output feature names to their processed values:
    # - 'image' is the transformed image tensor.
    # - 'label' is the original string label.
    # - 'label_int' is the integer-encoded label.
    return {
        'image': img,
        'label': img_record['label'],
        'label_int': img_record['label_int']
    }

def create_input_record(filename, label):
    # Convert the Arrow/pyarrow object 'label' to a nested python list.
    label_list = label.to_pylist()
    # Do the same for 'filename'.
    filename_list = filename.to_pylist()
    
    # These assertions check that each field yields a nested list containing exactly
    # one element: a sub-list that itself also contains one single value (the string filename/label).
    assert len(filename_list) == 1 and len(filename_list[0]) == 1
    assert len(label_list) == 1 and len(label_list[0]) == 1

    # Read the image bytes from disk at the filename specified.
    # filename_list[0][0] extracts the actual string path from the nested list structure.
    contents = tf.io.read_file(filename_list[0][0]).numpy()

    # Return a dictionary with:
    #  - 'img_bytes': byte content of the image,
    #  - 'label': the human-readable label string (bytes),
    #  - 'label_int': the integer index of the label in the global LABELS list.
    # label_list[0][0] is a bytes string, so decode to compare to LABELS.
    return {
        'img_bytes': contents,
        'label': label_list[0][0],
        'label_int': LABELS.index(label_list[0][0].decode())
    }

def run_main(arguments):
    global IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, LABELS
    
    JOBNAME = (
            'preprocess-images-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

    PROJECT = arguments['project_id']
    OUTPUT_DIR = arguments['output_dir']

    # set RUNNER using command-line arg or based on output_dir path
    on_cloud = OUTPUT_DIR.startswith('gs://')
    if arguments['runner']:
        RUNNER = arguments['runner']
        on_cloud = (RUNNER == 'DataflowRunner')
    else:
        RUNNER = 'DataflowRunner' if on_cloud else 'DirectRunner'

    # clean-up output directory since Beam will name files 0000-of-0004 etc.
    # and this could cause confusion if earlier run has 0000-of-0005, for eg
    if on_cloud:
        try:
            subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
        except subprocess.CalledProcessError:
            pass
    else:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR)
   
    # tf.config.run_functions_eagerly(not on_cloud)

    # read list of labels
    with tf.io.gfile.GFile(arguments['labels_file'], 'r') as f:
        LABELS = [line.rstrip() for line in f]
    print('Read in {} labels, from {} to {}'.format(
        len(LABELS), LABELS[0], LABELS[-1]))
    if len(LABELS) < 2:
        print('Require at least two labels')
        sys.exit(-1)

    # resize the input images
    ht, wd = arguments['resize'].split(',')
    IMG_HEIGHT = int(ht)
    IMG_WIDTH = int(wd)
    print("Will resize input images to {}x{}".format(IMG_HEIGHT, IMG_WIDTH))
        
    # make it repeatable
    np.random.seed(10)

    # set up Beam pipeline to convert images to TF Records
    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': JOBNAME,
        'project': PROJECT,
        'max_num_workers': 20, # autoscale up to 20
        'region': arguments['region'],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'save_main_session': True,
        'requirements_file': 'requirements.txt'
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)

    # RAW_DATA_SCHEMA defines the structure of the original CSV data that will be ingested.
    # It specifies that each record is expected to have a 'filename' and 'label', both as strings.
    RAW_DATA_SCHEMA = tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec({
            'filename': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        })

    # IMG_BYTES_METADATA defines the schema after image processing and transformation.
    # It describes that each processed record will contain:
    #   - 'img_bytes': The JPEG image data as a string (byte string)
    #   - 'label': The label as a string.
    #   - 'label_int': The label as an integer class ID (for downstream model training).
    # This metadata is used by tf.transform/Beam to describe the structure of the output TFRecords.
    IMG_BYTES_METADATA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
        tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec({
            'img_bytes': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'label_int': tf.io.FixedLenFeature([], tf.int64)
        })
    )
    # This line sets up CsvTFXIO, which is a utility from TensorFlow Transform (tf.Transform)
    # that handles reading CSV data into TensorFlow types for Beam pipelines.
    # Arguments explained:
    # - file_pattern: The CSV file(s) to read (path from our script arguments).
    # - column_names: List of column names ("filename", "label") expected in the CSV.
    # - schema: The expected data schema for input rows (in our case, RAW_DATA_SCHEMA).
    # - telemetry_descriptors: Tags for monitoring and tracking, mainly useful for debugging/data lineage.
    csv_tfxio = tfxio.CsvTFXIO(
        file_pattern=arguments['all_data'],
        column_names=['filename', 'label'],
        schema=RAW_DATA_SCHEMA,
        telemetry_descriptors=['standalone_tft']
    )

    # This is the main Beam pipeline that orchestrates the entire preprocessing workflow.
    # It sets up a pipeline with the specified runner (Dataflow or Direct) and options.
    # The pipeline is executed within a context that manages temporary files for Beam operations.
    with beam.Pipeline(RUNNER, options=opts) as p:
        with tft_beam.Context(temp_dir=os.path.join(OUTPUT_DIR, 'tmp', 'beam_context')):
            # The first step reads the CSV data using CsvTFXIO's BeamSource.
            # It processes the CSV file(s) specified in arguments['all_data'] into a stream of records.
            # Each record is a tuple containing (filename, label) from the CSV.
            img_records = (p
                      # Use batch_size=1 so that each CSV row is processed individually.
                      # For image preprocessing, handling one record at a time is simpler, and downstream transforms expect single examples.
                      | 'read_csv' >> csv_tfxio.BeamSource(batch_size=1)
                      | 'img_record' >> beam.Map(
                          # The next step converts each CSV record into a dictionary of features.
                          # It calls create_input_record to parse the filename and label,
                          # and returns a dictionary with 'img_bytes', 'label', and 'label_int'.
                          lambda x: create_input_record(x[0], x[1])))

            # The raw_dataset is a tuple containing:
            # - The input records (img_records) from the CSV.
            # - The metadata describing the schema of the input records (IMG_BYTES_METADATA).
            raw_dataset = (img_records, IMG_BYTES_METADATA)

            # The transformed_dataset is the result of applying the tft_preprocess function
            # to the raw_dataset. This function performs the actual image preprocessing steps:
            # - Decoding the image bytes into a float32 tensor.
            # - Rescaling the pixel values to the [0, 1] range.
            # - Resizing the images to the desired fixed size (IMG_HEIGHT, IMG_WIDTH).
            # The transform_fn is a SavedModel that contains the preprocessing logic.
            transformed_dataset, transform_fn = (
                raw_dataset | 'tft_img' >> tft_beam.AnalyzeAndTransformDataset(tft_preprocess)
            )
            # The transformed_dataset is a tuple containing:
            # - The transformed data (transformed_data) after applying the preprocessing steps.
            # - The metadata describing the schema of the transformed data (transformed_metadata).
            transformed_data, transformed_metadata = transformed_dataset
            # The transformed_data_coder is a utility that converts the transformed data
            # into a format suitable for writing to TFRecord files.
            transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)

            # The splits variable is a PCollection that contains the transformed data
            # assigned to the different splits ('train', 'valid', 'test').
            splits = (transformed_data
                      | 'create_tfr' >> beam.Map(transformed_data_coder.encode)
                      | 'assign_ds' >> beam.Map(assign_record_to_split)
                      )

            for split in ['train', 'valid', 'test']:
                write_records(OUTPUT_DIR, splits, split)

            # make sure to write out a SavedModel with the tf transforms that were carried out
            _ = (
                transform_fn | 'write_tft' >> tft_beam.WriteTransformFn(
                    os.path.join(OUTPUT_DIR, 'tft'))
            )

            if on_cloud:
                print("Submitting {} job: {}".format(RUNNER, JOBNAME))
                print("Monitor at https://console.cloud.google.com/dataflow/jobs")
            else:
                print("Running on DirectRunner. Please hold on ...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all_data',
        # pylint: disable=line-too-long
        help=
        'Path to input.  Each line of input has two fields  image-file-name and label separated by a comma',
        required=True)
    parser.add_argument(
        '--labels_file',
        help='Path to file containing list of labels, one per line',
        required=True)
    parser.add_argument(
        '--project_id',
        help='ID (not name) of your project. Ignored by DirectRunner',
        required=True)
    parser.add_argument(
        '--runner',
        help='If omitted, uses DataFlowRunner if output_dir starts with gs://',
        default=None)
    parser.add_argument(
        '--region',
        help='Cloud Region to run in. Ignored for DirectRunner',
        default='us-central1')
    parser.add_argument(
        '--resize',
        help='Specify the img_height,img_width that you want images resized.',
        default='448,448')
    parser.add_argument(
        '--output_dir', help='Top-level directory for TF Records', required=True)

    # Parse command-line arguments using argparse.
    # This returns a Namespace object, which is similar to a class instance
    # with attribute access for each argument (e.g., args.all_data, args.project_id, etc).
    args = parser.parse_args()

    # Convert the Namespace object to a regular dictionary so it can be
    # passed easily to run_main(arguments) for further use in the pipeline.
    arguments = args.__dict__
    
    run_main(arguments)
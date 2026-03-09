#!/usr/bin/env python3
"""
Inspect WOMD TFRecord structure to understand the schema.
"""

import tensorflow as tf
import os

data_dir = os.path.expanduser("~/projects/522_msis_hw1/data/womd")
tfrecord_file = os.path.join(data_dir, "validation.tfrecord-00000-of-00150")

print(f"Inspecting: {tfrecord_file}\n")

# Read first record
dataset = tf.data.TFRecordDataset(tfrecord_file)

for i, raw_record in enumerate(dataset.take(1)):
    # Parse as a generic Example proto
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    print("Available features:")
    print("=" * 80)

    features = example.features.feature

    for key in sorted(features.keys()):
        feature = features[key]

        # Determine feature type and shape
        if feature.HasField('bytes_list'):
            value_count = len(feature.bytes_list.value)
            sample = feature.bytes_list.value[0][:50] if value_count > 0 else b""
            print(f"{key}:")
            print(f"  Type: bytes_list")
            print(f"  Count: {value_count}")
            if value_count > 0:
                print(f"  Sample: {sample}...")
        elif feature.HasField('float_list'):
            values = feature.float_list.value
            print(f"{key}:")
            print(f"  Type: float_list")
            print(f"  Count: {len(values)}")
            if len(values) > 0:
                print(f"  Sample: {list(values[:5])}...")
        elif feature.HasField('int64_list'):
            values = feature.int64_list.value
            print(f"{key}:")
            print(f"  Type: int64_list")
            print(f"  Count: {len(values)}")
            if len(values) > 0:
                print(f"  Sample: {list(values[:5])}...")

        print()

print("\nTotal features:", len(features))

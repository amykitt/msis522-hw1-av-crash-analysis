#!/usr/bin/env python3
"""
Parse Waymo Open Motion Dataset (WOMD) TFRecord files into a pandas DataFrame.

This script reads WOMD v1.3.1 TFRecord files and extracts agent trajectory data
into a flat tabular format suitable for analysis.
"""

import os
import sys
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

# Add the waymo proto path to Python path
sys.path.insert(0, os.path.expanduser("~/projects/522_msis_hw1/waymo_repo/src"))

from waymo_open_dataset.protos import scenario_pb2


def parse_scenario(serialized_data):
    """Parse a serialized Scenario protobuf from WOMD."""
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(serialized_data)
    return scenario


def extract_trajectories(scenario):
    """
    Extract trajectory data from parsed scenario into flat records.

    Each agent at each timestep becomes a row in the output DataFrame.
    """
    records = []
    scenario_id = scenario.scenario_id
    timestamps = list(scenario.timestamps_seconds)

    # Iterate through all tracks (agents)
    for track in scenario.tracks:
        agent_id = track.id
        agent_type = track.object_type

        # Map agent type enum to string
        agent_type_name = scenario_pb2.Track.ObjectType.Name(agent_type)

        # Iterate through all timesteps for this agent
        for time_idx, state in enumerate(track.states):
            # Get the timestamp for this state
            timestamp = timestamps[time_idx] if time_idx < len(timestamps) else np.nan

            record = {
                'scenario_id': scenario_id,
                'agent_id': agent_id,
                'agent_type': agent_type,
                'agent_type_name': agent_type_name,
                'timestep': time_idx,
                'timestamp_seconds': timestamp,
                'x': state.center_x,
                'y': state.center_y,
                'z': state.center_z,
                'length': state.length,
                'width': state.width,
                'height': state.height,
                'heading': state.heading,
                'velocity_x': state.velocity_x,
                'velocity_y': state.velocity_y,
                'valid': state.valid,
            }
            records.append(record)

    return records


def process_tfrecord_files(data_dir, output_csv):
    """
    Process all TFRecord files in the data directory and save to CSV.
    """
    # Find all TFRecord files
    tfrecord_pattern = os.path.join(data_dir, "*.tfrecord*")
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))

    print(f"Found {len(tfrecord_files)} TFRecord files:")
    for f in tfrecord_files:
        print(f"  - {os.path.basename(f)}")

    all_records = []

    for file_idx, tfrecord_file in enumerate(tfrecord_files):
        print(f"\nProcessing file {file_idx + 1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_file)}")

        # Create a TFRecordDataset
        dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')

        scenario_count = 0
        for raw_record in dataset:
            try:
                scenario = parse_scenario(raw_record.numpy())
                records = extract_trajectories(scenario)
                all_records.extend(records)
                scenario_count += 1

                if scenario_count % 10 == 0:
                    print(f"  Processed {scenario_count} scenarios, {len(all_records)} total records...", end='\r')
            except Exception as e:
                print(f"\n  Warning: Failed to parse scenario: {e}")
                continue

        print(f"\n  Completed: {scenario_count} scenarios processed from this file")

    # Convert to DataFrame
    print(f"\nConverting {len(all_records)} records to DataFrame...")
    df = pd.DataFrame(all_records)

    # Add computed field for total velocity magnitude
    if 'velocity_x' in df.columns and 'velocity_y' in df.columns:
        df['velocity'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # Save to CSV
    print(f"Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)

    return df


def main():
    """Main entry point."""
    # Set paths
    data_dir = os.path.expanduser("~/projects/522_msis_hw1/data/womd")
    output_csv = os.path.expanduser("~/projects/522_msis_hw1/data/womd_tabular.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Process files
    print("=" * 60)
    print("WOMD TFRecord Parser")
    print("=" * 60)

    df = process_tfrecord_files(data_dir, output_csv)

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_csv}")
    print(f"DataFrame shape: {df.shape}")
    print(f"  Rows: {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")

    print(f"\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\nBasic statistics:")
    print(df[['x', 'y', 'velocity_x', 'velocity_y', 'velocity', 'heading']].describe())

    print(f"\nAgent type distribution:")
    print(df.groupby('agent_type_name').size())

    print(f"\nNumber of unique scenarios: {df['scenario_id'].nunique()}")
    print(f"Number of unique agents: {df['agent_id'].nunique()}")
    print(f"Number of valid observations: {df['valid'].sum():,} ({100*df['valid'].mean():.1f}%)")

    print(f"\nFirst few rows:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()

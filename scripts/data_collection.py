#!/usr/bin/env python3
"""
Script for collecting and processing robotics data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loaders.robomimic_loader import RoboMimicLoader
from src.loaders.d4rl_loader import D4RLLoader
from src.loaders.rosbag_loader import ROSBagLoader
from src.loaders.ieeedataport_loader import IEEEDataPortLoader
from src.preprocessing.sequence_preprocessor import SequencePreprocessor
import numpy as np

def main():
    # Example usage

    # RoboMimic
    robomimic_loader = RoboMimicLoader("data/robomimic")
    tasks = robomimic_loader.get_task_list()
    if tasks:
        data = robomimic_loader.load_dataset(tasks[0])
        print(f"Loaded RoboMimic data: {data['observations'].shape}")

    # D4RL
    d4rl_loader = D4RLLoader()
    envs = d4rl_loader.get_env_list()
    if envs:
        data = d4rl_loader.load_dataset(envs[0])
        print(f"Loaded D4RL data: {data['observations'].shape}")

    # ROS Bag
    ros_loader = ROSBagLoader("data/ros_bags/sample.bag")
    try:
        bag_data = ros_loader.load_bag()
        joint_data = ros_loader.extract_joint_states(bag_data)
        if joint_data:
            print(f"Loaded ROS joint data: {joint_data['positions'].shape}")
    except FileNotFoundError:
        print("ROS bag file not found, skipping.")

    # IEEE DataPort - manual download required
    ieeedp_loader = IEEEDataPortLoader("data/ieee_dataport")
    # ieeedp_loader.download_dataset("https://example.com/dataset.zip", "sample_dataset")

    # Preprocessing example
    preprocessor = SequencePreprocessor()
    if 'data' in locals():
        # Assume data is from one of the loaders
        sequences = preprocessor.create_sequences(data['observations'], seq_length=10)
        train_data, val_data, test_data = preprocessor.split_train_val_test(sequences)
        print(f"Created sequences: train {train_data.shape}, val {val_data.shape}, test {test_data.shape}")

if __name__ == "__main__":
    main()
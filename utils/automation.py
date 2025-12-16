import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../../')   # Add parent directory to Python path
from utils.preprocessing import *
from utils.segmentation import *

__author__ = "Ziyi Liu"
__copyright__ = """Copyright (C) 2024-2025 Ziyi Liu"""

"""
Utility functions for analyzing model predictions and matching false negatives to video data.

Author: Ziyi Liu
Project: Bachelor's Thesis - Cyclist Curb Detection under Varying Road Roughness
Institution: Universität Kassel
License: GNU General Public License v3.0

Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Original contact: cit27@cl.cam.ac.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

def preprocess_and_segment_curb(esp1_path, esp2_path, combined_output_path,freq_list, window_sizes, overlap, channels, scene_col='curb_scene'):
    """
    Preprocesses and segments accelerometer data from two ESP devices for curb detection.
    
    This function performs several steps:
    1. Handles missing values in data from both ESP devices
    2. Combines the data from both devices
    3. Downsamples the combined data to specified frequencies
    4. Segments the data into windows for each scene type (curb vs non-curb)
    
    Parameters:
    -----------
    esp1_path : str
        Path to the CSV file containing data from first ESP device
    esp2_path : str
        Path to the CSV file containing data from second ESP device
    combined_output_path : str
        Path where the combined data will be saved
    freq_list : list
        List of target frequencies for downsampling (e.g., [100, 30])
    window_sizes : list
        List of window sizes for segmentation, matching freq_list
    overlap : float
        Overlap ratio between consecutive windows (e.g., 0.5 for 50% overlap)
    channels : list
        List of accelerometer channels to process
    scene_col : str, optional
        Name of the column indicating scene type (default: 'curb_scene')
    
    Returns:
    --------
    None
        Saves processed files to disk:
        - Filled missing values: *_filled_missing_values.csv
        - Combined data: specified by combined_output_path
        - Downsampled data: *_<freq>hz.csv
        - Segmented data: *_scene<0/1>_segments.npz
    """
     
    # 1. Handle missing values for ESP1
    df_one = pd.read_csv(esp1_path)
    output1 = esp1_path.replace('.csv', '_filled_missing_values.csv')
    fill_missing_values_curb(df_one, output1)
    
    # 2. Handle missing values for ESP2
    df_two = pd.read_csv(esp2_path)
    output2 = esp2_path.replace('.csv', '_filled_missing_values.csv')
    fill_missing_values_curb(df_two, output2)
    
    # 3. Combine the two ESP dataframes into one
    df_combined = combine_activities_curb(df_one, df_two, combined_output_path)
    
    # 4. For each frequency and window size combination
    for freq, win_size in zip(freq_list, window_sizes):
        # 4a. Downsample the combined dataframe to the target frequency
        downsampled_path = combined_output_path.replace('.csv', f'_{freq}hz.csv')
        df_down = downsample_to_frequency(df_combined, target_hz=freq, timestamp_col='NTP',output_path=downsampled_path, categorical_attributes=[scene_col])
        
        # 4b. For each scene (e.g., curb_scene == 0 or 1)
        for scene in [0, 1]:
            # Filter the dataframe for the current scene
            df_scene = df_down[df_down[scene_col] == scene]
             # Segment the data into overlapping windows
            segments = segment_acceleration_data_overlapping_numpy(df_scene, window_size=win_size, overlap=overlap, channels=channels)
            # Save the segmented data as a .npz file
            np.savez(downsampled_path.replace('.csv', f'_scene{scene}_segments.npz'),segments=segments)


def preprocess_and_segment_road(esp_path,freq_list, window_sizes, overlap, channels):
    """
    Preprocesses and segments accelerometer data from road surface measurements.
    
    This function performs several steps:
    1. Loads and preprocesses accelerometer data from CSV file
    2. Downsamples data to specified frequencies
    3. Creates overlapping segments for each frequency
    4. Saves processed data in various formats
    
    Parameters:
    -----------
    esp_path : str
        Path to the CSV file containing accelerometer data
    freq_list : list
        List of target frequencies for downsampling (e.g., [100, 30])
    window_sizes : list
        List of window sizes for segmentation, matching freq_list
    overlap : float
        Overlap ratio between consecutive windows (e.g., 0.5 for 50% overlap)
    channels : list
        List of accelerometer channels to process (e.g., ['Acc-X', 'Acc-Y', 'Acc-Z'])
    
    Returns:
    --------
    None
        Saves processed files to disk:
        - Downsampled data: *_<freq>hz.csv
        - Segmented data: *_segments_<freq>hz_<window>s_<overlap>overlap.npz
    """
    ## for all road surfaces
    # 1. Handle missing values for ESP1
    df_one = pd.read_csv(esp_path)

    # Make sure NTP is datetime and set as index
    df_one['NTP'] = pd.to_datetime(df_one['NTP'])

    df_selected = df_one[['NTP', 'Acc-X', 'Acc-Y', 'Acc-Z']].copy()

    # 2. For each frequency and window size combination
    for freq, win_size in zip(freq_list, window_sizes):
        # 2. Downsample the combined dataframe to the target frequency
        downsampled_path = esp_path.replace('.csv', f'_{freq}hz.csv')
        df_down = downsample_to_frequency(df_selected, target_hz=freq, timestamp_col='NTP',output_path=downsampled_path, categorical_attributes=None)
        # 3.Segment the data into overlapping windows
        segments = segment_acceleration_data_overlapping_numpy(df_down, window_size=win_size, overlap=overlap, channels=channels)

        # 4. Save the segmented data as a .npz file
        # Calculate window size in seconds for better naming
        seconds = win_size / freq
        
        # Extract directory and base filename
        directory = os.path.dirname(esp_path)
        base_filename = os.path.splitext(os.path.basename(esp_path))[0]
                
        # Create the segments filename with proper format, including the original filename
        segment_filename = f'{base_filename}_segments_{freq}hz_{seconds}s_{overlap}overlap.npz'
        segment_path = os.path.join(directory, segment_filename)
        
        # Save the segmented data as a .npz file
        np.savez(segment_path, segments=segments)



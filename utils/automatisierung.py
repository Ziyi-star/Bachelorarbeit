import pandas as pd
import numpy as np
import sys
import os

sys.path.append('../../')   # Add parent directory to Python path
from utils.preprocessing import *
from utils.segmentation import *

def preprocess_and_segment_curb(esp1_path, esp2_path, combined_output_path,freq_list, window_sizes, overlap, channels, scene_col='curb_scene'):
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
    # 1. Handle missing values for ESP1
    df_one = pd.read_csv(esp_path)

    # Make sure NTP is datetime and set as index
    df_one['NTP'] = pd.to_datetime(df_one['NTP'])

    df_selected = df_one[['NTP', 'Acc-X', 'Acc-Y', 'Acc-Z']].copy()

    # 2. For each frequency and window size combination
    for freq, win_size in zip(freq_list, window_sizes):
        # 4a. Downsample the combined dataframe to the target frequency
        downsampled_path = esp_path.replace('.csv', f'_{freq}hz.csv')
        df_down = downsample_to_frequency(df_selected, target_hz=freq, timestamp_col='NTP',output_path=downsampled_path, categorical_attributes=None)
        # Segment the data into overlapping windows
        segments = segment_acceleration_data_overlapping_numpy(df_down, window_size=win_size, overlap=overlap, channels=channels)

        # Calculate window size in seconds for better naming
        seconds = win_size / freq
        
        # Extract directory for proper path construction
        directory = os.path.dirname(esp_path)
        
        # Create the segments filename with proper format
        segment_filename = f'segments_{freq}hz_{seconds}s_{overlap}overlap.npz'
        segment_path = os.path.join(directory, segment_filename)
        
        # Save the segmented data as a .npz file
        np.savez(segment_path, segments=segments)
import pandas as pd  
import numpy as np
import random
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from utils.visualization import plot_accelerometer_data

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


def combine_activities_curb(df_one, df_two, output_path):
    """
    Combines and processes specific curb crossing activities from two dataframes.
    
    Args:
        df_one (pd.DataFrame): First dataframe containing curb crossing data
        df_two (pd.DataFrame): Second dataframe containing curb crossing data
        output_path (str): Path where the combined CSV file will be saved
    
    Activity Types:
        - curb_activity: 1.0 = crossing down, 0.0 = crossing up
        - curb_type: 1.0 = curb, 2.0 = ramp, 3.0 = transition stone
    
    Returns:
        pd.DataFrame: Combined and sorted dataframe containing selected crossing activities
    """
    # Extract curb-down activities (crossing curb downwards)
    activity_one = df_one[(df_one['curb_activity'] == 1.0) & 
                         (df_one['curb_type_down'] == 1.0)]
    
    # Extract ramp-down activities (crossing ramp downwards)
    activity_three = df_two[(df_two['curb_activity'] == 1.0) & 
                          (df_two['curb_type_down'] == 2.0)]
    
    # Note: curb-up activities are currently commented out because they are not very correct
    # activity_two = df_two[(df_two['curb_activity'] == 0.0) & 
    #                      (df_two['curb_type_up'] == 1.0)]
    
    # Combine selected activities and reset the index
    df_combined = pd.concat([activity_one, activity_three], ignore_index=True)
    
    # Sort the combined data by timestamp (NTP)
    df_combined = df_combined.sort_values('NTP').reset_index(drop=True)
    
    # Save the processed data to CSV
    df_combined.to_csv(output_path, index=False)
    
    return df_combined


def fill_missing_values_curb(df, output_path):
    """
    Fill missing values in Acc-X,Y, Z column using temporal interpolation strategy.
    
    This function handles missing accelerometer X,Y,Z-axis values by:
    1. Using the previous value if within the same curb scene
    2. Using the next available value if at a scene boundary
    Also updates timestamps (NTP) to maintain temporal consistency.
    
    Args:
        df (pd.DataFrame): DataFrame containing all columns
        output_path (str): Path where the processed DataFrame will be saved as CSV
        
    Side Effects:
        - Modifies the input DataFrame in-place
        - Saves the processed DataFrame to a CSV file
        
    Note:
        NTP timestamps are adjusted by ±1 millisecond to maintain sequence order
    """
    # Convert NTP column to datetime format for temporal operations
    df['NTP'] = pd.to_datetime(df['NTP'])
    
    # Iterate through the DataFrame (starting from index 1)
    for index in range(1, len(df)):
        for col in ['Acc-X', 'Acc-Y', 'Acc-Z']:
            if pd.isnull(df[col].iloc[index]):
                # Case 1: Missing value within same curb scene
                if df['curb_scene'].iloc[index - 1] == df['curb_scene'].iloc[index]:
                    # Use previous value and increment timestamp
                    df.at[index, col] = df[col].iloc[index - 1]
                    df.at[index, 'NTP'] = df['NTP'].iloc[index - 1] + pd.Timedelta(milliseconds=1)
                else:
                    # Case 2: Missing value at scene boundary
                    # Search forward for next valid value
                    for j in range(index + 1, len(df)):
                        if not pd.isnull(df[col].iloc[j]):
                            df.at[index, col] = df[col].iloc[j]
                            df.at[index, 'NTP'] = df['NTP'].iloc[j] - pd.Timedelta(milliseconds=1)
                            break
    
    # Save processed DataFrame to CSV
    df.to_csv(output_path, index=False)

def downsample_to_frequency(df, target_hz, timestamp_col='NTP', output_path=None, categorical_attributes=None):
    """
    Downsamples the DataFrame to the specified frequency (Hz).
    For categorical attributes, takes the majority value in each interval.
    For numeric attributes, takes the mean.

    Parameters:
        df (pd.DataFrame): DataFrame with timestamp column.
        target_hz (int): Target frequency in Hz (e.g., 100 for 100Hz).
        timestamp_col (str): Name of the timestamp column.
        output_path (str): Path to save the downsampled CSV.
        categorical_attributes (list): List of categorical attribute names.

    Returns:
        pd.DataFrame: Downsampled DataFrame at the target frequency.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    interval_ms = int(1000 / target_hz)

    # Separate columns
    if categorical_attributes is None:
        categorical_attributes = []
    numeric_cols = [col for col in df.columns if col not in categorical_attributes]
    agg_dict = {col: 'mean' for col in numeric_cols}
    for cat in categorical_attributes:
        agg_dict[cat] = lambda x: x.mode().iloc[0] if not x.mode().empty else (x.iloc[0] if len(x) > 0 else np.nan)
    # Downsample the DataFrame by grouping data into intervals of 'interval_ms' milliseconds, then aggregate each column in these intervals using the functions specified in agg_dict
    df_downsampled = df.resample(f'{interval_ms}ms').agg(agg_dict)
    # Fills in missing values (NaNs) in the numeric columns of the downsampled DataFrame by interpolation
    df_downsampled[numeric_cols] = df_downsampled[numeric_cols].interpolate()
    # Fill missing values in categorical columns using forward fill, backward fill
    # Handle categorical columns - use forward fill, then backward fill
    for cat in categorical_attributes:
        # First forward fill (carry last valid observation forward)
        df_downsampled[cat] = df_downsampled[cat].fillna(method='ffill')
        # Then backward fill for any remaining NaNs at the beginning
        df_downsampled[cat] = df_downsampled[cat].fillna(method='bfill')
        # If still NaNs, fill with most common value
        if df_downsampled[cat].isna().any():
            most_common = df[cat].mode().iloc[0] if not df[cat].mode().empty else 0
            df_downsampled[cat] = df_downsampled[cat].fillna(most_common)
    df_downsampled = df_downsampled.reset_index()
    df_downsampled.to_csv(output_path, index=False)
    return df_downsampled

def trim_by_std(df, threshold=0.5, window_size=100):
    """
    Automatically trims the initial quiet period in accelerometer data where no significant movement occurs.
    
    Args:
        df: DataFrame with accelerometer data
        threshold: Standard deviation threshold to detect activity
        window_size: Size of the rolling window for standard deviation calculation
        
    Returns:
        Trimmed DataFrame starting from where activity begins
    """

    # Calculate rolling standard deviation for all axes
    # Computes the standard deviation within each window = window_size
    roll_std_x = df['Acc-X'].rolling(window=window_size).std()
    roll_std_y = df['Acc-Y'].rolling(window=window_size).std()
    roll_std_z = df['Acc-Z'].rolling(window=window_size).std()
    
    # Combine all axes to detect activity in any direction
    combined_std = roll_std_x + roll_std_y + roll_std_z
    
    # Find the first point where the combined standard deviation exceeds the threshold
    # (We use a buffer of window_size to ensure we have enough data before the activity starts)
    activity_starts = combined_std[window_size:].gt(threshold).idxmax()
    
    # If no activity is detected, return the original dataframe
    if activity_starts == 0:
        print("No significant activity detected in the dataset.")
        return df
    
    # Trim the dataframe to start from the detected activity start point
    # We can optionally include a small buffer before the activity starts
    buffer = int(window_size/2)  # Half window size as buffer
    start_idx = max(0, activity_starts - buffer)
    #start_idx = max(0, activity_starts)
    
    trimmed_df = df.iloc[start_idx:].copy()
    
    # Print info about the trimming
    start_time = df.iloc[activity_starts]['NTP']
    original_len = len(df)
    trimmed_len = len(trimmed_df)
    removed_percentage = ((original_len - trimmed_len) / original_len) * 100
    
    print(f"Activity detected starting at index {activity_starts}")
    print(f"Trimmed {original_len - trimmed_len} datapoints ({removed_percentage:.1f}% of the dataset)")
    print(f"Activity start time (NTP): {start_time}")
    
    return trimmed_df

def trim_by_start_time(df, start_time, time_column='NTP'):
    """
    Trim DataFrame to only include data after a specific start time.
    
    Args:
        df: DataFrame containing the data
        start_time: Start time as string (format: "YYYY-MM-DD HH:MM:SS")
        time_column: Name of the time column (default: 'NTP')
    
    Returns:
        Filtered DataFrame
    """
    # Convert time column to datetime
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert start time to datetime
    start = pd.to_datetime(start_time)
    
    # Filter data
    filtered_df = df[df[time_column] >= start].copy()
    
    return filtered_df

def trim_by_time(df, start_time, end_time=None, time_column='NTP'):
    """
    Trim DataFrame to include data between start_time and end_time (inclusive).
    If end_time is None, keeps all data from start_time to the end of DataFrame.
    
    Args:
        df: DataFrame containing the data
        start_time: Start time as string (format: "YYYY-MM-DD HH:MM:SS")
        end_time: End time as string, optional (default=None to keep until end)
        time_column: Name of the time column (default: 'NTP')
    
    Returns:
        Filtered DataFrame
    """
    # Convert time column to datetime
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert start time to datetime
    start = pd.to_datetime(start_time)
    
    # Create initial mask for start time
    mask = (df[time_column] >= start)
    
    # Add end time condition if provided
    if end_time is not None:
        end = pd.to_datetime(end_time)
        mask &= (df[time_column] <= end)
    
    # Apply filter in one operation
    filtered_df = df.loc[mask].copy()
    
    return filtered_df

def select_random_samples(data, num_samples):
    """
    Randomly selects a specified number of samples from the input data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array from which to select samples. The first dimension is assumed to be the sample dimension.
    num_samples : int
        Number of samples to select. Should be less than or equal to the number of samples in data.
        
    Returns:
    --------
    selected_data : numpy.ndarray
        Randomly selected samples from the input data. Has the same shape as input data except for
        the first dimension, which will be equal to num_samples.
    """
    
    # Generate random indices without replacement
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    
    # Select the samples using the random indices
    selected_data = data[indices]
    
    return selected_data

def normalize_3d_data(data):
    """
    Normalize 3D sensor data (samples, timesteps, features) using StandardScaler.

    Parameters:
    -----------
    data : list or numpy.ndarray
        Input data. If a list of arrays, it will be stacked first.
        Shape should be (N, L, C) where:
        N = number of samples, L = segment length (timesteps), C = number of channels/features

    Returns:
    --------
    numpy.ndarray
        Normalized data with the same shape as input
    """
    # Stack data if it's a list
    if isinstance(data, list):
        data_array = np.stack(data)  # shape: (N, L, C)
    else:
        data_array = data

    # Get dimensions
    N, L, C = data_array.shape

    # Reshape to 2D for scaling: (N*L, C)
    data_reshaped = data_array.reshape(-1, C)

    # Apply standard scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)

    # Reshape back to original 3D shape
    data_normalized = data_scaled.reshape(N, L, C)

    return data_normalized

def label_curb_scenes_real_world(df_data, df_curb, save_path=None):
    """
    Labels curb scenes in the accelerometer data using timestamp ranges.
    
    Args:
        df_data (pd.DataFrame): Accelerometer data with NTP timestamps
        df_curb (pd.DataFrame): Label data with timestamps and labels
        save_path (str, optional): Path to save labeled data
        
    Returns:
        pd.DataFrame: Labeled accelerometer data
    """
    # Create copy with selected columns
    count = 0
    df_selected = df_data[['NTP', 'Acc-X', 'Acc-Y', 'Acc-Z']].copy()
    
    # Initialize curb_activity column with 0
    df_selected['curb_activity'] = 0
    df_selected['curb_scene'] = 0
    
    # Convert timestamps to datetime
    df_selected['NTP'] = pd.to_datetime(df_selected['NTP'])
    df_curb['Timestamp'] = pd.to_datetime(df_curb['Timestamp'])
    
    valid_labels = df_curb.copy()
    print(f"Total rows in labels: {len(valid_labels)}")
    
    count = 0
    # Process events
    for idx in range(len(valid_labels)-1):
        current_label = valid_labels.iloc[idx]['Label']
        next_label = valid_labels.iloc[idx+1]['Label']
        
        if current_label not in [0] and next_label == 0:
            start_time = valid_labels.iloc[idx]['Timestamp']
            end_time = valid_labels.iloc[idx+1]['Timestamp']
            count += 1
            
            # Label data points between start and end time
            mask = (df_selected['NTP'] >= start_time) & (df_selected['NTP'] <= end_time)
            df_selected.loc[mask, 'curb_activity'] = current_label
    
    # Create curb_scene column based on condition
    df_selected['curb_scene'] = (~df_selected['curb_activity'].isin([0, 8, 9])).astype(int)
    
    print(f"Total label pairs found: {count}")

    if save_path:
        df_selected.to_csv(save_path, index=False)
    return df_selected






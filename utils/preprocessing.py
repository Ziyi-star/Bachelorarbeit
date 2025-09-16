import pandas as pd  
import numpy as np
import random
import matplotlib.dates as mdates

from utils.visualization import plot_accelerometer_data

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
    import numpy as np

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



import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# segement with time
def segment_acceleration_data_no_overlapping_time_diff(df,output):
    """
    Segments acceleration data into non-overlapping windows based on time intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing 'NTP' (timestamp), 'Acc-Z' (acceleration), and 'curb_scene' columns
    output : str
        Path to save the segmented data as CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Segmented data with 100 acceleration points per segment and metadata
    
    Notes:
    ------
    - Segments are created using 0.5s time windows (resampling)
    - Each segment must have at least 80 data points
    - Segments are standardized to exactly 100 points
    - Data is processed separately for each curb_scene group (0 and 1)
    """
    # Convert NTP column to datetime format
    df.loc[:, 'NTP'] = pd.to_datetime(df['NTP'])
    processed_segments = []

    # Group data by curb_scene (0 for normal, 1 for curb)
    grouped = df.groupby('curb_scene')
    for name, group in grouped:
        # Sort data by timestamp within each group
        group = group.sort_values(by='NTP')
        group.set_index('NTP', inplace=True)
        resampled = group.resample('0.5s')
        for index, segment in resampled:
            if len(segment) < 80:
                continue
            # Extract segment data and metadata
            acc_z_values = segment['Acc-Z'].values
            curb_scene_value = segment['curb_scene'].iloc[0]
            start_time = segment.index[0]
            end_time = segment.index[-1]
            data = {'curb_scene': curb_scene_value, 'start_time': start_time, 'end_time': end_time}
            
            # Standardize to exactly 100 data points
            if len(acc_z_values) > 100:
                acc_z_values = acc_z_values[:100]
            # Pad with last value if too short
            elif len(acc_z_values) < 100:
                acc_z_values = list(acc_z_values) + [acc_z_values[-1]] * (100 - len(acc_z_values))

            # Create columns for each acceleration value
            for j, value in enumerate(acc_z_values):
                data[f'Acc-Z_{j+1}'] = value
            new_df = pd.DataFrame([data])
            processed_segments.append(new_df)
        # Combine all segments into final DataFrame
        final_df = pd.concat(processed_segments, ignore_index=True)
        # Save the final DataFrame to a CSV file
        final_df.to_csv(output, index=False)
    return final_df



def segment_acceleration_data_overlapping_count_index(df, overlap, output):
    """
    Segments acceleration data into overlapping windows with fixed sample count.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing 'NTP' (timestamp), 'Acc-Z' (acceleration),
        and 'curb_scene' columns
    overlap : int
        Number of samples to overlap between consecutive segments (0-99)
    output : str
        Path to save the segmented data as CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Segmented data with the following columns:
        - curb_scene: Binary indicator (0 for normal, 1 for curb scene)
        - start_time: Start timestamp of the segment
        - end_time: End timestamp of the segment
        - Acc-Z_1 through Acc-Z_100: Acceleration values
    
    Notes:
    ------
    - Each segment contains exactly 100 samples
    - Step size between segments is (100 - overlap) samples
    - Data is processed separately for each curb_scene group
    """
    # Initialize list to store processed segments
    processed_segments = []
    step_size = 100 - overlap  # Calculate step size based on overlap
    
    # Group data by curb_scene (0 for normal, 1 for curb)
    grouped = df.groupby('curb_scene')
    
    # Process each group separately
    for name, group in grouped:
        # Sort data by timestamp within group
        group = group.sort_values(by='NTP')
        
        # Create overlapping segments of 100 samples
        for i in range(0, len(group), step_size):
            segment = group.iloc[i:i+100]
            
            # Skip incomplete segments
            if len(segment) < 100:
                break
                
            # Extract segment data and metadata
            acc_z_values = segment['Acc-Z'].values
            curb_scene_value = segment['curb_scene'].iloc[0]
            start_time = segment['NTP'].iloc[0]
            end_time = segment['NTP'].iloc[-1]
            
            # Create data dictionary for the segment
            data = {
                'curb_scene': curb_scene_value,
                'start_time': start_time,
                'end_time': end_time
            }
            
            # Add acceleration values to data dictionary
            for j, value in enumerate(acc_z_values):
                data[f'Acc-Z_{j+1}'] = value
                
            # Convert to DataFrame and append to results
            new_df = pd.DataFrame([data])
            processed_segments.append(new_df)
    
    # Combine all segments into final DataFrame
    final_df = pd.concat(processed_segments)
    
    # Save the processed data to CSV
    final_df.to_csv(output, index=False)
    
    return final_df
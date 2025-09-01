import pandas as pd  
import numpy as np
import matplotlib.dates as mdates

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

    df_downsampled = df.resample(f'{interval_ms}ms').agg(agg_dict)
    # Interpolate only numeric columns
    df_downsampled[numeric_cols] = df_downsampled[numeric_cols].interpolate()
    df_downsampled = df_downsampled.reset_index()
    df_downsampled.to_csv(output_path, index=False)
    return df_downsampled





import pandas as pd  
import numpy as np
import matplotlib.dates as mdates

def combine_activities(df_one, df_two, output_path):
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


def fill_missing_values(df, output_path):
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





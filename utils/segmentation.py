import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def segment_acceleration_data_overlapping_numpy(df,window_size=100, overlap=50, channels=['Acc-X', 'Acc-Y', 'Acc-Z']):
    """
    Segments acceleration data into overlapping windows and returns a list of 2D numpy arrays.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing timestamp, acceleration channels, and curb_scene columns.
    window_size : int
        Number of samples per segment.
    overlap : int
        Percentage overlap (0-99).
    channels : list
        List of column names for the channels (default: ['Acc-X', 'Acc-Y', 'Acc-Z']).
        
    Returns:
    --------
    segments : list of np.ndarray
        Each element is a 2D numpy array of shape (window_size, len(channels)).
    """
    step = int(window_size * (1 - overlap / 100))
    segments = []
    df_sorted = df.sort_values(by='NTP')
    for start in range(0, len(df_sorted) - window_size + 1, step):
        end = start + window_size
        segment = df_sorted.iloc[start:end][channels].values
        segments.append(segment)
    # Convert list of arrays to a single 3D numpy array if all segments have the same shape
    segments_array = np.array(segments)  # shape: (num_segments, window_size, 3)
    return segments_array
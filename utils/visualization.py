import pandas as pd  
import matplotlib.pyplot as plt


def print_sampling_frequency(df, timestamp_col='NTP'):
    """
    Calculates and prints the mean sampling frequency of a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing timestamp column.
        timestamp_col (str): Name of the timestamp column.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    time_diffs = df[timestamp_col].diff().dt.total_seconds()
    mean_freq = 1 / time_diffs.mean()
    print(f"Sampling frequency: {mean_freq:.2f} Hz")


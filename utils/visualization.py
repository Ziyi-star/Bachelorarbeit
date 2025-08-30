import pandas as pd  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.dates as mdates



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


def plot_accelerometer_data(df, name):
    """
    Plot Acc-X, Acc-Y, and Acc-Z for accelerometer data over time using Plotly.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Acc-X', 'Acc-Y', 'Acc-Z' columns with a time-based index or 'NTP' column.
    """
    if 'NTP' in df.columns:
        df['NTP'] = pd.to_datetime(df['NTP'])
        df = df.set_index('NTP')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Acc-Z'], mode='lines', name='Acc-Z'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Acc-Y'], mode='lines', name='Acc-Y'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Acc-X'], mode='lines', name='Acc-X'))

    fig.update_layout(
        title=name,
        xaxis_title='Time',
        yaxis_title='Acceleration (m/s^2)',
        legend_title='Axis',
        template='plotly_white',
        autosize=True,
    )
    fig.show()

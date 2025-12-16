import pandas as pd  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.dates as mdates
import numpy as np

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


def plot_accelerometer_data_bachelorarbeit(df):
    """
    Plot Acc-X, Acc-Y, and Acc-Z for handlebar accelerometer data over time.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Acc-X', 'Acc-Y', 'Acc-Z' columns with a time-based index.
    """
    df['NTP'] = pd.to_datetime(df['NTP'])
    df.set_index('NTP', inplace=True)
    plt.figure(figsize=(14, 7), dpi=300)
    plt.plot(df.index, df['Acc-X'], label='Acc-X', zorder=3)
    plt.plot(df.index, df['Acc-Y'], label='Acc-Y', zorder=2)
    plt.plot(df.index, df['Acc-Z'], label='Acc-Z', zorder=1)
    plt.legend(fontsize=14)
    plt.grid()
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    # Get the current axes and set major ticks every 200 seconds
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=200))
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Acceleration (m/s^2)', fontsize=20)
    plt.show()


def plot_sample_data_numpy_bachelorarbeit(sample_data, title="Acceleration Data"):
    """
    Plot 3-axis accelerometer data from a segmented numpy array for publication-quality visualization.
    
    This function creates a high-resolution matplotlib plot showing all three acceleration axes
    on a single figure, suitable for academic publications and presentations.
    
    Parameters:
    -----------
    sample_data : np.ndarray
        2D numpy array of shape (n_timesteps, 3) containing acceleration data
        where columns represent X, Y, Z acceleration values (m/s²)
    title : str, default="Acceleration Data"
        Title text to display at the top of the plot
        
    Returns:
    --------
    None
        Displays the plot using plt.show()
        
    Example:
    --------
    >>> segment = segments[0]  # Shape: (100, 3) for 1s window at 100Hz
    >>> plot_sample_data_numpy_bachelorarbeit(segment, "Curb Crossing Event")
    """
    # Create a high-resolution figure (300 DPI for publication quality)
    plt.figure(figsize=(14, 7), dpi=300)
    
    # Get number of time steps
    time_steps = np.arange(len(sample_data))
    
    # Plot X, Y, Z acceleration
    plt.plot(time_steps, sample_data[:, 0], label='X-axis')
    plt.plot(time_steps, sample_data[:, 1], label='Y-axis')
    plt.plot(time_steps, sample_data[:, 2], label='Z-axis')
    
    # Set plot title with provided text
    plt.title(title)
    plt.xlabel('Time Steps', fontsize = 20)
    plt.ylabel('Acceleration(m/s^2)', fontsize =20)
    plt.legend()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

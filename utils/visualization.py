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

    # Get axis limits to place arrows appropriately
    y_min = min(df['Acc-X'].min(), df['Acc-Y'].min(), df['Acc-Z'].min())
    y_max = max(df['Acc-X'].max(), df['Acc-Y'].max(), df['Acc-Z'].max())
    x_min = df.index.min()
    x_max = df.index.max()
    
    # Add padding for arrows
    y_range = y_max - y_min
    x_range = (x_max - x_min).total_seconds()
    
    arrow_x_end = x_max + pd.Timedelta(seconds=x_range * 0.05)
    arrow_y_end = y_max + y_range * 0.05

    fig.update_layout(
        title=name,
        xaxis_title='Time',
        yaxis_title='Acceleration (m/s^2)',
        legend_title='Axis',
        template='plotly_white',
        autosize=True,
        # Make axes thicker/more prominent
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            showgrid=True,
            range=[x_min, arrow_x_end]  # Extend range to show arrow
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            showgrid=True,
            range=[y_min, arrow_y_end]  # Extend range to show arrow
        ),
        # Add arrows for x and y axes
        shapes=[
            # X-axis arrow
            dict(
                type="line",
                x0=x_max, 
                y0=y_min,
                x1=arrow_x_end,
                y1=y_min,
                line=dict(color="black", width=2),
                layer="below"
            ),
            # X-axis arrowhead
            dict(
                type="line",
                x0=arrow_x_end - pd.Timedelta(seconds=x_range * 0.01),
                y0=y_min - y_range * 0.01,
                x1=arrow_x_end,
                y1=y_min,
                line=dict(color="black", width=2),
                layer="below"
            ),
            dict(
                type="line",
                x0=arrow_x_end - pd.Timedelta(seconds=x_range * 0.01),
                y0=y_min + y_range * 0.01,
                x1=arrow_x_end,
                y1=y_min,
                line=dict(color="black", width=2),
                layer="below"
            ),
            # Y-axis arrow
            dict(
                type="line",
                x0=x_min,
                y0=y_max,
                x1=x_min,
                y1=arrow_y_end,
                line=dict(color="black", width=2),
                layer="below"
            ),
            # Y-axis arrowhead
            dict(
                type="line",
                x0=x_min - pd.Timedelta(seconds=x_range * 0.01),
                y0=arrow_y_end - y_range * 0.01,
                x1=x_min,
                y1=arrow_y_end,
                line=dict(color="black", width=2),
                layer="below"
            ),
            dict(
                type="line",
                x0=x_min + pd.Timedelta(seconds=x_range * 0.01),
                y0=arrow_y_end - y_range * 0.01,
                x1=x_min,
                y1=arrow_y_end,
                line=dict(color="black", width=2),
                layer="below"
            ),
        ]
    )
    fig.show()
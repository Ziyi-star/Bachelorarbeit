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


# def plot_accelerometer_data(df, name):
#     """
#     Plot Acc-X, Acc-Y, and Acc-Z for accelerometer data over time.
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing 'Acc-X', 'Acc-Y', 'Acc-Z' columns with a time-based index.
#     """

#     if 'NTP' in df.columns:
#         df['NTP'] = pd.to_datetime(df['NTP'])
#         df = df.set_index('NTP')
#     plt.figure(figsize=(14, 7), dpi=300)
#     plt.title(name)
#     plt.plot(df.index, df['Acc-X'], label='Acc-X', zorder=3)
#     plt.plot(df.index, df['Acc-Y'], label='Acc-Y', zorder=2)
#     plt.plot(df.index, df['Acc-Z'], label='Acc-Z', zorder=1)
#     plt.legend()
#     plt.grid()
#     # Rotate date labels
#     plt.gcf().autofmt_xdate()
#     plt.xticks(rotation=45)
#     # Get the current axes and set major ticks every 120 seconds
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(mdates.SecondLocator(interval=120))
#     plt.xlabel('Time')
#     plt.ylabel('Acceleration (m/s^2)')
#     plt.show()

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

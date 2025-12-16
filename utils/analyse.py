import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

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


def match_false_negatives_to_video(df_fn, df_video, buffer_seconds=0.1):
    """
    Match false negative segments to nearest video timestamps within a time buffer.
    
    This function finds the closest video frame timestamp for each false negative (FN)
    prediction segment, enabling visual inspection and validation of model errors.
    
    Parameters:
    -----------
    df_fn : pd.DataFrame
        DataFrame containing false negative segments with 'Start Time' column
    df_video : pd.DataFrame
        DataFrame containing video metadata with 'NTP' (timestamp) and 'Video' columns
    buffer_seconds : float, default=0.1
        Maximum time difference (in seconds) to consider a video frame as matching
        
    Returns:
    --------
    pd.DataFrame
        Enhanced df_fn with three new columns:
        - 'video': Name/path of the matched video file
        - 'matched_ntp': Exact timestamp of the matched video frame
        - 'time_diff': Time difference between FN segment and matched video frame
    """
    # Create a time delta object for the buffer window (e.g., ±0.1 seconds)
    td = pd.Timedelta(seconds=buffer_seconds)

    # Create copies to avoid modifying original dataframes
    df_fn = df_fn.copy()
    df_video = df_video.copy()
    
    # Convert timestamp columns to datetime objects for accurate time comparisons
    df_fn['Start Time'] = pd.to_datetime(df_fn['Start Time'], errors='coerce')
    df_video['NTP'] = pd.to_datetime(df_video['NTP'], errors='coerce')

    # Initialize new columns to store matching results
    # NaN values indicate no match was found within the buffer
    df_fn['video'] = np.nan           # Video file name/path
    df_fn['matched_ntp'] = np.nan     # Matched video frame timestamp
    df_fn['time_diff'] = pd.NaT       # Time difference between FN and video frame

    # Iterate through each false negative segment
    for idx, row in df_fn.iterrows():
        start_time = row['Start Time']
        
        # Find all video frames within the time buffer window
        # Search range: [start_time - buffer, start_time + buffer]
        candidates = df_video[
            (df_video['NTP'] >= start_time - td) &
            (df_video['NTP'] <= start_time + td)
        ]
        
        # If matching video frames were found
        if len(candidates):
            candidates = candidates.copy()
            
            # Calculate absolute time difference for each candidate
            candidates['time_diff'] = (candidates['NTP'] - start_time).abs()
            
            # Select the video frame with the smallest time difference (closest match)
            best = candidates.loc[candidates['time_diff'].idxmin()]
            
            # Store the matching information in the false negative dataframe
            df_fn.at[idx, 'video'] = best['Video']              # Video file name
            df_fn.at[idx, 'matched_ntp'] = best['NTP']          # Exact timestamp
            df_fn.at[idx, 'time_diff'] = best['time_diff']      # Time difference

    return df_fn
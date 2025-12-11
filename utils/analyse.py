import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import sys
import pandas as pd


def match_false_negatives_to_video(df_fn, df_video, buffer_seconds=0.1):
    """Match FN segments to nearest video NTP within a time buffer."""
    td = pd.Timedelta(seconds=buffer_seconds)

    # Ensure datetime types
    df_fn = df_fn.copy()
    df_video = df_video.copy()
    df_fn['Start Time'] = pd.to_datetime(df_fn['Start Time'], errors='coerce')
    df_video['NTP'] = pd.to_datetime(df_video['NTP'], errors='coerce')

    # Prepare columns
    df_fn['video'] = np.nan
    df_fn['matched_ntp'] = np.nan
    df_fn['time_diff'] = pd.NaT

    for idx, row in df_fn.iterrows():
        start_time = row['Start Time']
        candidates = df_video[
            (df_video['NTP'] >= start_time - td) &
            (df_video['NTP'] <= start_time + td)
        ]
        if len(candidates):
            candidates = candidates.copy()
            candidates['time_diff'] = (candidates['NTP'] - start_time).abs()
            best = candidates.loc[candidates['time_diff'].idxmin()]
            df_fn.at[idx, 'video'] = best['Video']
            df_fn.at[idx, 'matched_ntp'] = best['NTP']
            df_fn.at[idx, 'time_diff'] = best['time_diff']

    return df_fn
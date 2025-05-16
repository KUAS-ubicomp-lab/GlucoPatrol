import numpy as np
import pandas as pd
from scipy.stats import skew


class DataSegmenter:
    def __init__(self, subject_id):
        self.subject_id = subject_id

    def __segment_df(self, signal_df, glucose_timestamp, fs, window_size):
        """Segments a given signal dataframe based on glucose timestamps."""
        start_time = glucose_timestamp - pd.Timedelta(seconds=window_size//fs)
        end_time = glucose_timestamp
        segment = signal_df[(signal_df.index > start_time)
                            & (signal_df.index <= end_time)]
        sig_period = 1 / fs * 1000  # in ms

        if len(segment) < window_size:
            segment = segment.reindex(
                pd.date_range(start=start_time, end=end_time, freq=f'{sig_period}ms'), fill_value=np.nan
            ).rename_axis("timestamp")

        return segment.tail(window_size)

    def __segment_all_signals(self, e4_dfs_dict, glucose_df, epoch_size_dict, fs_dict):
        """Segments all signals for each glucose timestamp and returns a new dataframe."""
        segmented_glucose_df = glucose_df.copy()

        for signal_name, signal_df in e4_dfs_dict.items():
            window_size = epoch_size_dict[signal_name]
            fs = fs_dict[signal_name]

            # segmented_glucose_df[f'{signal_name}_segments'] = segmented_glucose_df['timestamp'].apply(
            #     lambda ts: self.__segment_df(signal_df, ts, fs, window_size)
            # )
            segmented_glucose_df[f'{signal_name}_segments'] = segmented_glucose_df['timestamp'].apply(
                lambda ts, sdf=signal_df, f=fs, ws=window_size: self.__segment_df(
                    sdf, ts, f, ws)
            )

        return segmented_glucose_df

    def __drop_missing_epochs(self, segmented_glucose_df, e4_dfs_dict, epoch_size_dict):
        """Drops rows where more than 50% of an epoch is missing."""
        rows_to_drop = []

        for signal_name in e4_dfs_dict.keys():
            window_size = epoch_size_dict[signal_name]

            for idx, row in segmented_glucose_df.iterrows():
                segment = row[f'{signal_name}_segments']
                if segment.isnull().sum().sum() > window_size // 2:
                    rows_to_drop.append(idx)

        return segmented_glucose_df.drop(index=set(rows_to_drop))

    def __impute_all_subjects(self, segmented_glucose_df, e4_dfs_dict):
        for signal_name in e4_dfs_dict.keys():
            segmented_glucose_df[f'{signal_name}_segments'] = segmented_glucose_df[f'{signal_name}_segments'].apply(
                lambda df: df.apply(lambda column: column.fillna(
                    column.dropna().mean() if column.dropna().nunique() <= 1 or abs(skew(column.dropna())) < 0.5
                    else column.dropna().median()
                ))
            )
        return segmented_glucose_df

    def process_all(self, e4_dfs_dict, glucose_df, epoch_size_dict, fs_dict):
        """Runs segmentation, drops missing epochs, and imputes missing values in one function."""
        segmented_glucose_df = self.__segment_all_signals(
            e4_dfs_dict, glucose_df, epoch_size_dict, fs_dict)
        segmented_glucose_df = self.__drop_missing_epochs(
            segmented_glucose_df, e4_dfs_dict, epoch_size_dict)
        segmented_glucose_df = self.__impute_all_subjects(
            segmented_glucose_df, e4_dfs_dict)
        return segmented_glucose_df

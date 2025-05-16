import os

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_glucose_data(self, subject_id):
        """Load glucose data for a given subject."""
        path = os.path.join(
            self.data_path, f"{subject_id:03d}", f"Dexcom_{subject_id:03d}.csv")
        df = pd.read_csv(path)

        df['timestamp'] = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)'])
        df = df[['timestamp', 'Glucose Value (mg/dL)']].rename(
            columns={'Glucose Value (mg/dL)': 'glucose'})
        return df.dropna(subset=['timestamp']).reset_index(drop=True)

    def load_E4_data(self, subject_id, signal_type, timestamp_col, retain_cols):
        """Load E4 signals (HR, ACC, TEMP, etc.)"""
        path = os.path.join(
            self.data_path, f"{subject_id:03d}", f"{signal_type}_{subject_id:03d}.csv")
        df = pd.read_csv(path)

        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df = df[['timestamp'] +
                retain_cols].sort_values('timestamp').dropna().reset_index(drop=True)
        # Drop rows and reset index if timestamp is not available
        df = df.dropna(subset=['timestamp']).reset_index(drop=True)

        # Remove leading and trailing spaces
        df.columns = df.columns.str.strip()

        # Special processing
        if signal_type == "ACC":
            df["acc"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]
                                ** 2 + df["acc_z"]**2)
            df = df[["timestamp", "acc"]]

        if signal_type == "HR":
            df["timestamp"] = df["timestamp"] + \
                pd.to_timedelta(df.groupby("timestamp").cumcount(), unit="s")

        df = df.set_index("timestamp")

        # Remove duplicate indices, only retain first value
        df = df.loc[~df.index.duplicated(keep='first')]

        return df

    def load_segmented_dfs(self, subject_id, epoch_duration):
        path = os.path.join(
            self.data_path, f"segmented_df_{subject_id}_{epoch_duration}.pkl")
        return pd.read_pickle(path)

    def load_feature_dfs(self, subject_id, epoch_duration):
        path = os.path.join(
            self.data_path, f"feature_df_{subject_id}_{epoch_duration}.pkl")
        return pd.read_pickle(path)

    def load_cleaned_feature_dfs(self, subject_id, epoch_duration):
        path = os.path.join(
            self.data_path, f"clean_feature_df_{subject_id}_{epoch_duration}.pkl")
        return pd.read_pickle(path)

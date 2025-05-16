import numpy as np
from scipy.signal import butter, filtfilt


class SignalProcessor:

    @staticmethod
    def TD_cutoff_filter(df, signal_name, lower_cut, upper_cut):
        df.loc[(df[signal_name] < lower_cut) | (
            df[signal_name] > upper_cut), signal_name] = np.nan
        df = df.dropna()
        return df

    @staticmethod
    def bandpass_filter(data, fs, lowcut, highcut, order=4):
        """Apply bandpass filter."""
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        return filtfilt(b, a, data)

    @staticmethod
    def lowpass_filter(data, fs, cutoff, order=4):
        """Apply low-pass filter."""
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype="low")
        return filtfilt(b, a, data)

    def process_hr(self, df, signal_name):
        """Filter HR signal."""
        valid_data = self.TD_cutoff_filter(df, signal_name, 25, 240)
        return valid_data

    def process_temp(self, df, signal_name):
        """Filter TEMP signal."""
        valid_data = self.TD_cutoff_filter(df, signal_name, 30, 40)
        return valid_data

    def process_bvp(self, df, signal_name):
        """Filter BVP signal."""
        df.loc[:, f'{signal_name}'] = self.bandpass_filter(
            df[f'{signal_name}'], fs=64, lowcut=0.5, highcut=5)
        valid_data_df = self.TD_cutoff_filter(df, signal_name, -500, 500)
        return valid_data_df

    def process_acc(self, df, signal_name):
        """Filter ACC signal."""
        df.loc[:, f'{signal_name}'] = self.lowpass_filter(
            df[f'{signal_name}'], fs=32, cutoff=10)
        valid_data_df = self.TD_cutoff_filter(df, signal_name, 0, 221)
        return valid_data_df

    def process_eda(self, df, signal_name):
        """Filter EDA signal."""
        df.loc[:, f'{signal_name}'] = self.lowpass_filter(
            df[f'{signal_name}'], fs=4, cutoff=0.5)
        valid_data_df = self.TD_cutoff_filter(df, signal_name, 0.01, 100)
        return valid_data_df

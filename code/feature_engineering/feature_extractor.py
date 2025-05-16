import gc
import warnings

import antropy as ant
import EntropyHub as EH
import flirt
import heartpy as hp
import nolds
import numpy as np
import ordpy
import pandas as pd
import scipy
from pyentrp import entropy as ent
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from tqdm.notebook import tqdm
# Custom functions
from utils import utils

# NOTES
"""
    FD and NL features were not extracted
"""


class FeatureExtractor:
    def __init__(self, subject_id, epoch_size_dict, e4_signal_list, fs_dict):
        self.subject_id = subject_id
        self.epoch_size_dict = epoch_size_dict
        self.e4_dfs_dict_keys = e4_signal_list
        self.fs_dict = fs_dict

    def get_circadianFeatures(self, df):
        circadian_features_df = pd.DataFrame()
        circadian_features_df['TS_mins_midnight'] = df['timestamp'].dt.hour * \
            60 + df['timestamp'].dt.minute
        circadian_features_df['TS_mins_midnight_sin'] = np.sin(
            2*np.pi*circadian_features_df['TS_mins_midnight'] / 1440)
        circadian_features_df['TS_mins_midnight_cos'] = np.cos(
            2*np.pi*circadian_features_df['TS_mins_midnight'] / 1440)
        return circadian_features_df

    def get_demographicFeatures(self, df):
        demographic_features_df = df.copy()
        if self.subject_id in [1, 3, 4, 5, 6, 7, 8, 10, 15]:
            # Use binary encoding
            demographic_features_df['biological_sex'] = 0  # female
        else:
            demographic_features_df['biological_sex'] = 1  # male
        return demographic_features_df['biological_sex']

    def get_SubjectID(self, df):
        subject_id_df = df.copy()
        subject_id_df['subject_id'] = self.subject_id
        return subject_id_df['subject_id']

    # For ACC: TODO: Filter separately axis-wise

    # For BVP: HeartPy: TODO: check PPG signal vs BVP signal waveform, units
    def get_hrv_features(self, df):
        column_list = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad',
                       'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']

        try:
            bvp = pd.concat(df['bvp_segments'].tolist())
        except KeyError as e:
            print(f'Error - {e} for {self.subject_id}')

        m_concat_df = pd.DataFrame()
        n_epochs = len(bvp) // self.epoch_size_dict['bvp']

        for epoch in tqdm(range(n_epochs), desc=f"Subject {self.subject_id}", leave=True):
            try:
                wd, m = hp.process(
                    bvp['bvp'].values[epoch*self.epoch_size_dict['bvp']:(epoch + 1)*self.epoch_size_dict['bvp']],
                    sample_rate=64.0
                )
                m_df = pd.DataFrame([m])
            except Exception:
                m_df = pd.DataFrame(np.nan, index=[0], columns=column_list)
                # print(f"HeartPy failed to process the signal for subject {subject_id}, epoch {epoch}.")
            m_concat_df = pd.concat([m_concat_df, m_df])

        hrv_features = m_concat_df.reset_index(drop=True)
        # Convert sdsd column to float
        hrv_features['sdsd'] = hrv_features['sdsd'].astype(float)
        hrv_features = hrv_features.rename(columns={'sd1/sd2': 'sd1_sd2'})
        hrv_features = hrv_features.add_prefix("BVP_")
        return hrv_features

    # For EDA: FLIRT features are ok
    def get_eda_features(self, df):
        try:
            eda = pd.concat(df['eda_segments'].tolist())
        except Exception as e:
            print(f'Error - {e} for {self.subject_id}')
        eda_features = flirt.get_eda_features(
            eda['eda'], window_length=300, window_step_size=300, data_frequency=4, num_cores=1)

        try:
            eda_features = eda_features.drop(
                columns=['phasic_entropy', 'tonic_entropy'])
        except Exception as e:
            print("Columns not found - ", e)
        eda_features = eda_features.reset_index(drop=True)
        eda_features = eda_features.add_prefix("EDA_")
        return eda_features

    # For Temp

    def get_time_domain_features(self, signal, signal_name=""):

        maxp = np.nanmax(signal, axis=1)
        minp = np.nanmin(signal, axis=1)
        mean = np.mean(signal, axis=1)
        median = np.median(signal, axis=1)
        std = np.std(signal, axis=1)
        cv = std / mean
        n5 = np.nanpercentile(signal, 5, axis=1)
        n25 = np.nanpercentile(signal, 25, axis=1)
        n75 = np.nanpercentile(signal, 75, axis=1)
        n95 = np.nanpercentile(signal, 95, axis=1)
        mav = np.mean(np.abs(signal), axis=1)
        iav = np.sum(np.abs(signal), axis=1)
        rms = np.sqrt(np.nanmean(signal ** 2, axis=1))
        p2p = np.ptp(signal, axis=1)
        skew = scipy.stats.skew(signal, axis=1)
        kurt = scipy.stats.kurtosis(signal, axis=1)
        mmt_5th = scipy.stats.moment(signal, moment=5, axis=1)
        gm = scipy.stats.gmean(signal, axis=1)
        m10 = scipy.stats.trim_mean(signal, 0.10, axis=1)
        m25 = scipy.stats.trim_mean(signal, 0.25, axis=1)
        hjm, hjc = ant.hjorth_params(signal)
        timeDomainsignal_names = np.concatenate((
            np.matrix(maxp).T, np.matrix(minp).T, np.matrix(mean).T, np.matrix(
                median).T, np.matrix(std).T, np.matrix(cv).T,
            np.matrix(n5).T, np.matrix(n25).T, np.matrix(n75).T, np.matrix(
                n95).T, np.matrix(mav).T, np.matrix(iav).T,
            np.matrix(rms).T, np.matrix(p2p).T,
            np.matrix(skew).T, np.matrix(kurt).T, np.matrix(mmt_5th).T,
            np.matrix(gm).T, np.matrix(m10).T, np.matrix(m25).T,
            np.matrix(hjm).T, np.matrix(hjc).T
        ), axis=1)

        feature_list = [
            "maxp", "minp", "mean", "median", "std", "cv", "n5", "n25", "n75", "n95", "mav", "iav",
                    "rms", "p2p",
                    "skew", "kurt", "mmt_5th", "gm", "m10", "m25",
                    "hjm", "hjc"
        ]

        for i in range(len(feature_list)):
            feature_list[i] = signal_name.upper() + "_" + feature_list[i]

        dataframe = pd.DataFrame(timeDomainsignal_names, columns=feature_list)
        return dataframe

    def get_freq_domain_features(self, signal, frame_size, fs, signal_name=""):
        warnings.filterwarnings("ignore")

        # Notes
        # frame_size = length of each epoch (no. of columns)
        # len(signal): no. of epochs
        timestep = 1 / fs
        feature_list = ["mean", "var", "third", "forth", "grand", "std", "cfactor", "dfactor",
                        "efactor", "gfactor", "third1", "forth1", "hfactor", "jfactor"]

        for i in range(len(feature_list)):
            feature_list[i] = signal_name.upper() + "_" + feature_list[i]

        stationary_features = []
        mean = []
        var = []
        third = []
        forth = []
        grand = []
        std = []
        cfactor = []
        dfactor = []
        efactor = []
        gfactor = []
        third1 = []
        forth1 = []
        hfactor = []
        jfactor = []

        for i in range(0, len(signal)):
            # pbar_freq.update(1)

            # 1. FFT Computation
            # 2. Taking the magnitude of the complex FFT result
            # 3. Select the first half (postitive freq) (FFT is symmetric about midpoint)

            # ===> y = magnitudes of frequency components after FFT
            y = abs(np.fft.fft(signal[i] / frame_size))[:int(frame_size / 2)]
            current_mean = np.sum(y) / frame_size
            mean.append(current_mean)

            current_var = (
                np.sum((y - (np.sum(y) / frame_size)) ** 2)) / (frame_size - 1)
            var.append(current_var)

            current_third = (np.sum((y - (np.sum(y) / frame_size)) ** 3)) / (frame_size * (
                np.sqrt((np.sum((y - (np.sum(y) / frame_size)) ** 2)) / (frame_size - 1))) ** 3)
            third.append(current_third)

            current_forth = (np.sum((y - (np.sum(y) / frame_size)) ** 4)) / (frame_size * (
                (np.sum((y - (np.sum(y) / frame_size)) ** 2)) / (frame_size - 1)) ** 2)
            forth.append(current_forth)

            # Generates frequency bins corresponding to the positive frequencies in the FFT
            f = np.fft.fftfreq(frame_size, timestep)[:int(frame_size / 2)]
            current_grand = np.sum(f * y) / np.sum(y)
            grand.append(current_grand)

            current_std = np.sqrt(
                np.sum((f - (np.sum(f * y) / np.sum(y))) ** 2 * y) / frame_size)
            std.append(current_std)

            current_cfactor = np.sqrt(np.sum(f ** 2 * y) / np.sum(y))
            cfactor.append(current_cfactor)

            current_dfactor = np.sqrt(np.sum(f ** 4 * y) / np.sum(f ** 2 * y))
            dfactor.append(current_dfactor)

            current_efactor = np.sqrt(
                np.sum(f ** 2 * y) / np.sqrt(np.sum(y) * np.sum(f ** 4 * y)))
            efactor.append(current_efactor)

            current_gfactor = (np.sqrt(np.sum(
                (f - (np.sum(f * y) / np.sum(y))) ** 2 * y) / frame_size)) / (np.sum(f * y) / np.sum(y))
            gfactor.append(current_gfactor)

            current_third1 = np.sum((f - (np.sum(f * y) / np.sum(y))) ** 3 * y) / (frame_size * (
                np.sqrt(np.sum((f - (np.sum(f * y) / np.sum(y))) ** 2 * y) / frame_size)) ** 3)
            third1.append(current_third1)

            current_forth1 = np.sum((f - (np.sum(f * y) / np.sum(y))) ** 4 * y) / (frame_size * (
                np.sqrt(np.sum((f - (np.sum(f * y) / np.sum(y))) ** 2 * y) / frame_size)) ** 4)
            forth1.append(current_forth1)

            current_hfactor = np.sum(np.sqrt(abs(f - (np.sum(f * y) / np.sum(y)))) * y) / (
                frame_size * np.sqrt(np.sqrt(np.sum((f - (np.sum(f * y) / np.sum(y))) ** 2 * y) / frame_size)))
            hfactor.append(current_hfactor)

            current_jfactor = ((np.sqrt(np.sum(f ** 2 * y) / np.sum(y))) + (
                np.sqrt(np.sum(f ** 4 * y) / np.sum(f ** 2 * y)))) / (np.sum(y) / frame_size)
            jfactor.append(current_jfactor)

        stationary_features.append(mean)
        stationary_features.append(var)
        stationary_features.append(third)
        stationary_features.append(forth)
        stationary_features.append(grand)
        stationary_features.append(std)
        stationary_features.append(cfactor)
        stationary_features.append(dfactor)
        stationary_features.append(efactor)
        stationary_features.append(gfactor)
        stationary_features.append(third1)
        stationary_features.append(forth1)
        stationary_features.append(hfactor)
        stationary_features.append(jfactor)

        stationary_features = np.array(stationary_features).T

        dataframe = pd.DataFrame(stationary_features, columns=feature_list)

        return dataframe

    def get_nonlinear_features(self, signal, signal_name=""):
        warnings.filterwarnings("ignore")

        feature_list = [
            "RQAresult_recurrence_rate", "RQAresult_determinism", "RQAresult_average_diagonal_line", "RQAresult_longest_diagonal_line", "RQAresult_divergence",
            "RQAresult_entropy_diagonal_lines", "RQAresult_laminarity", "RQAresult_trapping_time", "RQAresult_longest_vertical_line", "RQAresult_entropy_vertical_lines",
            "RQAresult_ratio_determinism_recurrence_rate", "RQAresult_ratio_laminarity_determinism",
            "corDim", "alpha", "alphaOverlap",
            "hurstExpK", "ApEn[0]", "ApEn[1]", "ApEn[2]", "CoSiEn",
            "Bm", "CondEn[0]", "CondEn[1]", "DispEn", "rDispEn",
            "DistEn", "FuzzEn[0]", "FuzzEn[1]", "GridEn", "GDR",
            "IncrEn", "K2En[0]", "K2En[1]", "PermEn[0]", "PermEn[1]",
            "PhasEn", "SampEn[0]", "SampEn[1]", "SampEn[2]", "SpecEn",
            "ShannonEn", "ComplexEn"
        ]

        for i in range(len(feature_list)):
            feature_list[i] = signal_name.upper() + \
                "_" + feature_list[i]

        aRQAresult_recurrence_rate = np.ones(len(signal)) * np.nan
        aRQAresult_determinism = np.ones(len(signal)) * np.nan
        aRQAresult_average_diagonal_line = np.ones(len(signal)) * np.nan
        aRQAresult_longest_diagonal_line = np.ones(len(signal)) * np.nan
        aRQAresult_divergence = np.ones(len(signal)) * np.nan
        aRQAresult_entropy_diagonal_lines = np.ones(len(signal)) * np.nan
        aRQAresult_laminarity = np.ones(len(signal)) * np.nan
        aRQAresult_trapping_time = np.ones(len(signal)) * np.nan
        aRQAresult_longest_vertical_line = np.ones(len(signal)) * np.nan
        aRQAresult_entropy_vertical_lines = np.ones(len(signal)) * np.nan
        aRQAresult_ratio_determinism_recurrence_rate = np.ones(
            len(signal)) * np.nan
        aRQAresult_ratio_laminarity_determinism = np.ones(len(signal)) * np.nan
        acorDim = np.ones(len(signal)) * np.nan
        aalpha = np.ones(len(signal)) * np.nan
        aalphaOverlap = np.ones(len(signal)) * np.nan
        ahurstExpK = np.ones(len(signal)) * np.nan
        aApEn0 = np.ones(len(signal)) * np.nan
        aApEn1 = np.ones(len(signal)) * np.nan
        aApEn2 = np.ones(len(signal)) * np.nan
        aCoSiEn = np.ones(len(signal)) * np.nan
        aBm = np.ones(len(signal)) * np.nan
        aCondEn0 = np.ones(len(signal)) * np.nan
        aCondEn1 = np.ones(len(signal)) * np.nan
        aDispEn = np.ones(len(signal)) * np.nan
        arDispEn = np.ones(len(signal)) * np.nan
        aDistEn = np.ones(len(signal)) * np.nan
        aFuzzEn0 = np.ones(len(signal)) * np.nan
        aFuzzEn1 = np.ones(len(signal)) * np.nan
        aGridEn = np.ones(len(signal)) * np.nan
        aGDR = np.ones(len(signal)) * np.nan
        aIncrEn = np.ones(len(signal)) * np.nan
        aK2En0 = np.ones(len(signal)) * np.nan
        aK2En1 = np.ones(len(signal)) * np.nan
        aPermEn0 = np.ones(len(signal)) * np.nan
        aPermEn1 = np.ones(len(signal)) * np.nan
        aPhasEn = np.ones(len(signal)) * np.nan
        aSampEn0 = np.ones(len(signal)) * np.nan
        aSampEn1 = np.ones(len(signal)) * np.nan
        aSampEn2 = np.ones(len(signal)) * np.nan
        aSpecEn = np.ones(len(signal)) * np.nan
        aShannonEn = np.ones(len(signal)) * np.nan
        aComplexEn = np.ones(len(signal)) * np.nan

        for i in range(len(signal)):
            data_points = signal[i]
            if ~np.isnan(signal[i]).any():
                if len(np.unique(signal[i])) != 1:
                    time_series = TimeSeries(
                        data_points, embedding_dimension=2, time_delay=2)
                    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(
                        0.65), similarity_measure=EuclideanMetric, theiler_corrector=1)
                    computation = RQAComputation.create(settings, verbose=True)
                    RQAresult = computation.run()
                    RQAresult.min_diagonal_line_length = 2
                    RQAresult.min_vertical_line_length = 2
                    RQAresult.min_white_vertical_line_length = 2

                    aRQAresult_recurrence_rate[i] = RQAresult.recurrence_rate
                    aRQAresult_determinism[i] = RQAresult.determinism
                    aRQAresult_average_diagonal_line[i] = RQAresult.average_diagonal_line
                    aRQAresult_longest_diagonal_line[i] = RQAresult.longest_diagonal_line
                    aRQAresult_divergence[i] = RQAresult.divergence
                    aRQAresult_entropy_diagonal_lines[i] = RQAresult.entropy_diagonal_lines
                    aRQAresult_laminarity[i] = RQAresult.laminarity
                    aRQAresult_trapping_time[i] = RQAresult.trapping_time
                    aRQAresult_longest_vertical_line[i] = RQAresult.longest_vertical_line
                    aRQAresult_entropy_vertical_lines[i] = RQAresult.entropy_vertical_lines
                    aRQAresult_ratio_determinism_recurrence_rate[
                        i] = RQAresult.ratio_determinism_recurrence_rate
                    aRQAresult_ratio_laminarity_determinism[i] = RQAresult.ratio_laminarity_determinism

                    corDim = nolds.corr_dim(signal[i], emb_dim=2)
                    alpha = nolds.dfa(signal[i], overlap=False)
                    alphaOverlap = nolds.dfa(signal[i], overlap=True)
                    hurstExpK = nolds.hurst_rs(signal[i])

                    ApEn, _ = EH.ApEn(signal[i], m=2, tau=1)
                    CoSiEn, Bm = EH.CoSiEn(signal[i], m=2, tau=1)
                    CondEn, _, _ = EH.CondEn(signal[i], m=2, tau=1)
                    DispEn, rDispEn = EH.DispEn(signal[i], m=2, tau=1)
                    DistEn, _ = EH.DistEn(signal[i], m=2, tau=1)
                    FuzzEn, _, _ = EH.FuzzEn(signal[i], m=2, tau=1)
                    GridEn, GDR, _ = EH.GridEn(signal[i], m=2, tau=1)
                    IncrEn = EH.IncrEn(signal[i], m=2, tau=1)
                    K2En, _ = EH.K2En(signal[i], m=2, tau=1)
                    PermEn, _, _ = EH.PermEn(signal[i], m=2, tau=1)
                    PhasEn = EH.PhasEn(signal[i], tau=1)
                    SampEn, _, _ = EH.SampEn(signal[i], m=2, tau=1)
                    SpecEn, _ = EH.SpecEn(signal[i])

                    ShannonEn = ent.shannon_entropy(signal[i])
                    _, ComplexEn = ordpy.complexity_entropy(signal[i], dx=2)

                    acorDim[i] = corDim
                    aalpha[i] = alpha
                    aalphaOverlap[i] = alphaOverlap
                    ahurstExpK[i] = hurstExpK
                    aApEn0[i] = ApEn[0]
                    aApEn1[i] = ApEn[1]
                    aApEn2[i] = ApEn[2]
                    aCoSiEn[i] = CoSiEn
                    aBm[i] = Bm
                    aCondEn0[i] = CondEn[0]
                    aCondEn1[i] = CondEn[1]
                    aDispEn[i] = DispEn
                    arDispEn[i] = rDispEn
                    aDistEn[i] = DistEn
                    aFuzzEn0[i] = FuzzEn[0]
                    aFuzzEn1[i] = FuzzEn[1]
                    aGridEn[i] = GridEn
                    aGDR[i] = GDR
                    aIncrEn[i] = IncrEn
                    aK2En0[i] = K2En[0]
                    aK2En1[i] = K2En[1]
                    aPermEn0[i] = PermEn[0]
                    aPermEn1[i] = PermEn[1]
                    aPhasEn[i] = PhasEn
                    aSampEn0[i] = SampEn[0]
                    aSampEn1[i] = SampEn[1]
                    aSampEn2[i] = SampEn[2]
                    aSpecEn[i] = SpecEn
                    aShannonEn[i] = ShannonEn
                    aComplexEn[i] = ComplexEn

            # Invoke garbage collection periodically to free memory
            if i % 100 == 0:
                gc.collect()

        results = [aRQAresult_recurrence_rate, aRQAresult_determinism, aRQAresult_average_diagonal_line, aRQAresult_longest_diagonal_line, aRQAresult_divergence,
                   aRQAresult_entropy_diagonal_lines, aRQAresult_laminarity, aRQAresult_trapping_time, aRQAresult_longest_vertical_line, aRQAresult_entropy_vertical_lines,
                   aRQAresult_ratio_determinism_recurrence_rate, aRQAresult_ratio_laminarity_determinism, acorDim, aalpha, aalphaOverlap, ahurstExpK, aApEn0, aApEn1, aApEn2,
                   aCoSiEn, aBm, aCondEn0, aCondEn1, aDispEn, arDispEn, aDistEn, aFuzzEn0, aFuzzEn1, aGridEn, aGDR, aIncrEn, aK2En0, aK2En1, aPermEn0, aPermEn1,
                   aPhasEn, aSampEn0, aSampEn1, aSampEn2, aSpecEn, aShannonEn, aComplexEn]
        return pd.DataFrame(np.array(results).T, columns=feature_list)

    def get_data_driven_features(self, segmented_df):
        TD_features = {}
        TDFDNL_features = {}

        TD_signal_list = ['temp', 'acc']
        # for signal_name in self.e4_signal_list:
        for signal_name in TD_signal_list:
            signal_reshaped = utils.reshape_nested_column(
                segmented_df, f'{signal_name}_segments')
            TD_features[signal_name] = self.get_time_domain_features(
                signal_reshaped, signal_name)

            TDFDNL_features[signal_name] = pd.concat(
                [TD_features[signal_name]], axis=1)

        all_TDFDNL_features = pd.concat(TDFDNL_features.values(), axis=1)
        return all_TDFDNL_features

    def combine_all_features(self, segmented_df):
        circadian_features = self.get_circadianFeatures(segmented_df)
        demographic_features = self.get_demographicFeatures(segmented_df)
        subject_id_col = self.get_SubjectID(segmented_df)
        eda_features = self.get_eda_features(segmented_df)
        hrv_features = self.get_hrv_features(segmented_df)
        data_driven_features = self.get_data_driven_features(segmented_df)

        return pd.concat([segmented_df['glucose'], subject_id_col, circadian_features, demographic_features, eda_features, hrv_features, data_driven_features], axis=1)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import RFE, mutual_info_regression
from xgboost import XGBRegressor


class FeatureSelector:
    def __init__(self, subject_id, method='shap', k=10, random_seed=42, output_dir=""):
        self.subject_id = subject_id
        self.method = method
        self.k = k
        self.output_dir = output_dir
        self.random_seed = random_seed

        self.selected_features = None
        self.shap_values_ = None
        self.explainer_ = None
        self.model_ = None
        self.X_ = None

    @staticmethod
    def _ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def fit(self, X, y):
        if self.method == 'shap':
            self._embedded_shap(X, y)
        elif self.method == 'rfe':
            self._wrapper_rfe(X, y)
        elif self.method == 'mi':
            self._filter_mutual_info(X, y)
        else:
            raise ValueError(f"Method '{self.method}' not recognized")

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError(
                "Feature selection has not been fitted yet. Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        return X[self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _filter_mutual_info(self, X, y):
        """ Filter-based: Mutual Information (MI) """

        mi = mutual_info_regression(X, y)
        selected_feature_indices = np.argsort(mi)[::-1][:self.k]
        self.selected_features = X.columns[selected_feature_indices]

    def _wrapper_rfe(self, X, y):
        """ Wrapper-based: Recursive Feature Elimination (RFE) """
        wrapper_rfe_xgb = RFE(XGBRegressor(
            n_jobs=25, random_state=self.random_seed), n_features_to_select=self.k)
        wrapper_rfe_xgb.fit(X, y)

        selected_feature_indices = wrapper_rfe_xgb.get_support(indices=True)
        self.selected_features = X.columns[selected_feature_indices]

    def _embedded_shap(self, X, y):
        """ Embedded: SHAP (SHapley Additive exPlanations) """
        xgb = XGBRegressor(n_jobs=25, random_state=self.random_seed)
        xgb.fit(X, y)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X)

        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_indices = np.argsort(feature_importance)[::-1][:self.k]
        self.selected_features = X.columns[feature_indices]

        # Save for manual plotting
        self.shap_values_ = shap_values
        self.explainer_ = explainer
        self.model_ = xgb
        self.X_ = X

        self._ensure_dir(self.output_dir)

        # Save SHAP values per feature
        shap_df = pd.DataFrame(self.shap_values_, columns=self.X_.columns)
        shap_df['subject_id'] = self.subject_id
        shap_df.to_csv(f"{self.output_dir}/shap_values.csv", index=False)

        # Save corresponding feature values
        X_copy = self.X_.copy()
        X_copy['subject_id'] = self.subject_id
        X_copy.to_csv(f"{self.output_dir}/feature_values.csv", index=False)

    def plot_shap_summary(self, plot_type='dot', max_display=10, save=True, show=False):
        """
        Plot a SHAP summary plot ('dot' or 'bar') after calling fit(method='shap').

        Parameters:
            plot_type (str): 'dot' (beeswarm) or 'bar' summary plot type.
            max_display (int): Max number of top features to show.
            save (bool): Whether to save the plot as a PNG file.
            show (bool): Whether to display the plot interactively.
        """
        if not hasattr(self, 'shap_values_'):
            raise ValueError(
                "SHAP values not computed. Run fit() with method='shap' first.")

        shap.initjs()

        file_name = f"{self.output_dir}/shap_summary_{'beeswarm' if plot_type == 'dot' else 'bar'}.png"

        plt.clf()
        shap.summary_plot(self.shap_values_, self.X_,
                          plot_type=plot_type, max_display=max_display, show=show)

        if save:
            fig = plt.gcf()
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Prevent figure overlap on next plot

    def get_selected_features(self):
        if self.selected_features is None:
            raise ValueError(
                "Feature selection has not been fitted yet. Call fit() first.")
        return self.selected_features

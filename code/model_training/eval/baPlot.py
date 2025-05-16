import matplotlib.pyplot as plt
import numpy as np


class BlandAltmanPlotter:
    def __init__(self, model_name: str):
        """
        Initializes the BlandAltmanPlotter.

        Parameters:
        - model_name (str): The name of the model for labeling the plot.
        """
        self.model_name = model_name
        self.ref_values = None
        self.pred_values = None
        self.differences = None
        self.mean_diff = None
        self.std_diff = None
        self.loa_upper = None
        self.loa_lower = None
        self.percent_outside_loa = None

    def compute(self, ref_values, pred_values):
        """
        Computes the statistics for the Bland–Altman plot.

        Parameters:
        - ref_values (array-like): Reference (true) values.
        - pred_values (array-like): Predicted values.
        """
        self.ref_values = np.asarray(ref_values)
        self.pred_values = np.asarray(pred_values)
        self.differences = self.ref_values - self.pred_values
        self.mean_diff = np.mean(self.differences)
        self.std_diff = np.std(self.differences)
        self.loa_upper = self.mean_diff + 1.96 * self.std_diff
        self.loa_lower = self.mean_diff - 1.96 * self.std_diff
        self.percent_outside_loa = self._calculate_outside_loa_percent()

    def _calculate_outside_loa_percent(self):
        num_outside = np.sum((self.differences > self.loa_upper) | (
            self.differences < self.loa_lower))
        return (num_outside / len(self.differences)) * 100

    def plot(self, label_fontsize=12):
        """
        Plots the Bland–Altman plot in the currently active axes context.

        Parameters:
        - label_fontsize (int): Font size for the labels.
        """
        if self.ref_values is None or self.pred_values is None:
            raise ValueError("Must call `.compute()` before plotting.")

        plt.scatter(self.ref_values, self.differences,
                    c='#31C0CE', marker='.', s=80, alpha=0.7)
        plt.axhline(self.mean_diff, color='#278A97', linestyle='-',
                    linewidth=2, label=f'Mean: {self.mean_diff:.2f}')
        plt.axhline(self.loa_upper, color='#E3995E', linestyle='--',
                    linewidth=2, label=f'+1.96 SD: {self.loa_upper:.2f}')
        plt.axhline(self.loa_lower, color='#E3995E', linestyle='--',
                    linewidth=2, label=f'-1.96 SD: {self.loa_lower:.2f}')

        xmax = np.max(self.ref_values) + 60
        plt.text(xmax, self.mean_diff - 15, f'Mean:\n {self.mean_diff:.2f}',
                 color='k', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
        plt.text(xmax, self.loa_upper - 12, f'+1.96 SD:\n {self.loa_upper:.2f}',
                 color='k', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
        plt.text(xmax, self.loa_lower - 20, f'-1.96 SD:\n {self.loa_lower:.2f}',
                 color='k', verticalalignment='bottom', horizontalalignment='right', fontsize=10)

        plt.title(
            f'BA Plot - {self.model_name}', fontsize=label_fontsize + 2)
        plt.xlabel('Reference concentration (mg/dL)', fontsize=label_fontsize)
        plt.ylabel('Difference (mg/dL)', fontsize=label_fontsize)
        plt.xlim([0, np.max(self.ref_values)])
        plt.ylim([-120, 120])
        plt.tight_layout()

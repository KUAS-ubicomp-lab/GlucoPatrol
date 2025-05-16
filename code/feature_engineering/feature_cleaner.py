import numpy as np
from sklearn.feature_selection import VarianceThreshold

class FeatureCleaner:
    def __init__(self,
                 col_nan_thresh=0.2,
                 row_nan_thresh=0.2,
                 low_variance_thresh=1e-5,
                 corr_thresh=0.95,
                 verbose=True):
        """
        Initializes the FeatureCleaner with configurable thresholds.

        Parameters:
        - col_nan_thresh: float, max fraction of NaNs allowed in a column
        - row_nan_thresh: float, max fraction of NaNs allowed in a row
        - low_variance_thresh: float, threshold for removing low-variance features
        - corr_thresh: float, threshold for dropping highly correlated features
        - verbose: bool, whether to print log information
        """
        self.col_nan_thresh = col_nan_thresh
        self.row_nan_thresh = row_nan_thresh
        self.low_variance_thresh = low_variance_thresh
        self.corr_thresh = corr_thresh
        self.verbose = verbose
        self.log = {}

    def clean(self, df_input):
        """
        Cleans a DataFrame of extracted features before modeling.

        Parameters:
        - df_input: pd.DataFrame, raw features to be cleaned

        Returns:
        - cleaned_df: pd.DataFrame, cleaned version of input
        - log: dict, details of dropped columns/rows
        """
        df = df_input.copy()
        initial_shape = df.shape

        self.log = {
            "columns_dropped": {
                "non_numeric": [],
                "nan_or_overflow": [],
                "constant": [],
                "duplicates": [],
                "low_variance": [],
                "high_corr": []
            },
            "rows_dropped": {
                "infinite_values": [],
                "too_many_nans": []
            },
            "initial_shape": initial_shape,
            "final_shape": None
        }

        # Remove non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        df = df.select_dtypes(include=[np.number])
        self.log["columns_dropped"]["non_numeric"] = non_numeric_cols

        # Drop columns with too many NaNs or overflow values
        too_many_nans = df.columns[df.isna().sum() > self.col_nan_thresh * len(df)].tolist()
        overflow_columns = df.columns[(df > np.finfo(np.float32).max).any()].tolist()
        nan_or_overflow_cols = list(set(too_many_nans + overflow_columns))
        df.drop(columns=nan_or_overflow_cols, inplace=True)
        self.log["columns_dropped"]["nan_or_overflow"] = nan_or_overflow_cols

        # Drop constant columns
        constant_cols = df.columns[df.nunique() <= 1].tolist()
        df.drop(columns=constant_cols, inplace=True)
        self.log["columns_dropped"]["constant"] = constant_cols

        # Drop duplicated columns
        duplicated_map = []
        cols = df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    duplicated_map.append({'dropped': cols[j], 'duplicate_of': cols[i]})
        duplicated_cols = [entry['dropped'] for entry in duplicated_map]
        df.drop(columns=duplicated_cols, inplace=True)
        self.log["columns_dropped"]["duplicates"] = duplicated_map

        # Drop rows with Inf/-Inf
        inf_rows = df.index[df.isin([np.inf, -np.inf]).any(axis=1)].tolist()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.drop(index=inf_rows, inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.log["rows_dropped"]["infinite_values"] = inf_rows

        # Drop rows with too many NaNs
        too_many_nan_rows = df.index[df.isnull().mean(axis=1) >= self.row_nan_thresh].tolist()
        df = df.drop(index=too_many_nan_rows).reset_index(drop=True)
        self.log["rows_dropped"]["too_many_nans"] = too_many_nan_rows

        # Drop low-variance features
        try:
            selector = VarianceThreshold(threshold=self.low_variance_thresh)
            selector.fit(df.fillna(0))  # Temporarily fill NaNs for variance calc
            low_var_cols = df.columns[~selector.get_support()].tolist()
            df.drop(columns=low_var_cols, inplace=True)
            self.log["columns_dropped"]["low_variance"] = low_var_cols
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Variance thresholding skipped due to error: {e}")

        # Drop highly correlated features
        if df.shape[1] > 1:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_cols = [col for col in upper.columns if any(upper[col] > self.corr_thresh)]
            df.drop(columns=high_corr_cols, inplace=True)
            self.log["columns_dropped"]["high_corr"] = high_corr_cols

        self.log["final_shape"] = df.shape

        if self.verbose:
            self._print_summary()

        return df

    def _print_summary(self):
        print("=== Cleaning Summary ===")
        total_cols = sum(len(cols) for cols in self.log["columns_dropped"].values())
        total_rows = sum(len(rows) for rows in self.log["rows_dropped"].values())

        for step, cols in self.log["columns_dropped"].items():
            print(f"  Dropped {len(cols):>3} columns ({step}): {cols}")
        for step, rows in self.log["rows_dropped"].items():
            print(f"  Dropped {len(rows):>3} rows due to ({step})")

        print(f"\n  ➤ Total dropped columns: {total_cols}")
        print(f"  ➤ Total dropped rows: {total_rows}")
        print(f"  ➤ Final shape: {self.log['final_shape']} (was {self.log['initial_shape']})")

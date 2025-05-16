import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def debug_array(arr: np.ndarray) -> None:
    """
    Print detailed debugging information about a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Input NumPy array to inspect.
    """
    nan_count = np.isnan(arr).sum()
    array_has_nan = nan_count > 0

    logger.debug("[DEBUG] Array Debugging Information:")
    logger.debug("Shape: %s", arr.shape)
    logger.debug("Data Type: %s", arr.dtype)
    logger.debug("Contains NaN: %s", array_has_nan)
    logger.debug("Total NaN count: %s", nan_count)
    logger.debug("Array preview:\n%s", arr)


def has_nan(arr: np.ndarray) -> bool:
    """
    Check whether a NumPy array contains any NaN values.

    Parameters
    ----------
    arr : np.ndarray
        Input NumPy array.

    Returns
    -------
    bool
        True if the array contains any NaNs, False otherwise.
    """
    return np.isnan(arr).any()


def reshape_nested_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Reshape a column of nested DataFrames into a NumPy array.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name containing nested DataFrames.

    Returns
    -------
    np.ndarray
        Flattened 2D NumPy array.
    """
    return np.array([nested_df.values.flatten() for nested_df in df[column]])


def nested_df_has_nan(subject_id: str, df: pd.DataFrame, column_list: list[str], raise_error: bool = False) -> bool:
    """
    Check if reshaped columns of nested DataFrames contain NaNs.

    Parameters
    ----------
    subject_id : str
        ID of the subject (used for error messages).
    df : pd.DataFrame
        DataFrame to check.
    column_list : list of str
        List of base column names (without "_segments").

    raise_error : bool, optional
        If True, raises a ValueError when NaNs are found. Default is False.

    Returns
    -------
    bool
        True if NaNs are found in any reshaped column, False otherwise.

    Raises
    ------
    ValueError
        If NaNs are detected and raise_error is True.
    """
    for column in column_list:
        reshaped = reshape_nested_column(df, f'{column}_segments')
        if has_nan(reshaped):
            if raise_error:
                raise ValueError(
                    f"NaNs detected in {column} for Subject {subject_id}.")
            return True
    return False

def compare_dataframes_with_tolerance(df1, df2, rtol=1e-05, atol=1e-08):
    """
    Compare two DataFrames with numeric tolerance for floating-point values.

    Parameters:
    df1 (pd.DataFrame)
    df2 (pd.DataFrame)
    rtol (float): Relative tolerance
    atol (float): Absolute tolerance

    Returns:
    bool: True if DataFrames are effectively equal, False otherwise
    """
    if df1.shape != df2.shape or not all(df1.columns == df2.columns):
        print("DataFrames differ in shape or columns.")
        return False

    unequal_cells = []
    for col in df1.columns:
        s1 = df1[col]
        s2 = df2[col]

        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            comparison = np.isclose(s1, s2, rtol=rtol, atol=atol, equal_nan=True)
        else:
            comparison = s1 == s2

        if not comparison.all():
            for idx in comparison[~comparison].index:
                unequal_cells.append((idx, col, s1.loc[idx], s2.loc[idx]))

    if unequal_cells:
        print("Differences found (within tolerance limits):")
        for idx, col, v1, v2 in unequal_cells:
            print(f"Row {idx}, Column '{col}': df1={v1}, df2={v2}")
        return False
    else:
        print("DataFrames are effectively equal within tolerance.")
        return True


# ============================
# TODOs for future utilities
# ============================

# - Check if a DataFrame has duplicated indices
# - Check for NaNs across multiple nested columns
# - Validate segment lengths across nested arrays
# - Add exportable summary of missing data and shapes

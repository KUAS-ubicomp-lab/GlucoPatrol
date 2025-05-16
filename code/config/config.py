import os

# File Paths

BASE_PATH = "/home/thilsk/Documents/12_Projects/12.02_BIL_project/6_BIL_FL"
DATA_PATH = os.path.join(BASE_PATH, "data/0_dataset")
SEGDATA_PATH = os.path.join(BASE_PATH, "data/1_segmented_data")
RAW_FEATURE_PATH = os.path.join(BASE_PATH, "features/0_raw")
CLEAN_FEATURE_PATH = os.path.join(BASE_PATH, "features/1_clean")
RESULTS_PATH = os.path.join(BASE_PATH, "results")


# Sampling Rates
FS = {
    "bvp": 64,
    "acc": 32,
    "temp": 4,
    "eda": 4,
    "hr": 1
}

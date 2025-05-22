import os

from dotenv import load_dotenv

load_dotenv()

# Base path
BASE_PATH = os.getenv("BASE_PATH")

# Paths built from environment variables
DATA_PATH = os.path.join(BASE_PATH, os.getenv("DATA_SUBDIR"))
SEGDATA_PATH = os.path.join(BASE_PATH, os.getenv("SEGDATA_SUBDIR"))
RAW_FEATURE_PATH = os.path.join(BASE_PATH, os.getenv("RAW_FEATURE_SUBDIR"))
CLEAN_FEATURE_PATH = os.path.join(BASE_PATH, os.getenv("CLEAN_FEATURE_SUBDIR"))
RESULTS_PATH = os.path.join(BASE_PATH, os.getenv("RESULTS_SUBDIR"))

# Sampling Rates
FS = {
    "bvp": 64,
    "acc": 32,
    "temp": 4,
    "eda": 4,
    "hr": 1
}

import os
import sys

from config.config import CLEAN_FEATURE_PATH, RAW_FEATURE_PATH
from utils.import_classes import load_class

DataLoader = load_class("data_loader")
FeatureCleaner = load_class("feature_cleaner")

def main(subject_id, epoch_duration):

    loader = DataLoader(RAW_FEATURE_PATH)
    feature_cleaner = FeatureCleaner(
       corr_thresh=1.0, verbose=True) # TODO: change default corr_thresh

    # Load and clean data
    feature_df = loader.load_feature_dfs(subject_id, epoch_duration)
    cleaned_feature_df = feature_cleaner.clean(feature_df)

    # Save cleaned features
    cleaned_feature_df.to_pickle(os.path.join(
        CLEAN_FEATURE_PATH, f"clean_feature_df_{subject_id}_{epoch_duration}.pkl"))
    print(f"Saved cleaned features for subject {subject_id}.\n")


if __name__ == "__main__":
    subject_id_ = int(sys.argv[1])
    epoch_duration_minutes_ = int(sys.argv[2])
    epoch_duration_seconds_ = epoch_duration_minutes_ * 60
    main(subject_id=subject_id_, epoch_duration=epoch_duration_seconds_)

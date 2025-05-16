import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from config.config import FS, RAW_FEATURE_PATH, SEGDATA_PATH
from utils.import_classes import load_class

DataLoader = load_class("data_loader")
FeatureExtractor = load_class("feature_extractor")

def process_batch(subject_id, epoch_duration, start_idx, batch_segmented_df, signal_list, epoch_size_dict):

    feature_extractor = FeatureExtractor(
        subject_id, epoch_size_dict, signal_list, FS)
    feature_df = feature_extractor.combine_all_features(batch_segmented_df)

    batch_feature_save_path = f"{RAW_FEATURE_PATH}/feature_batch_{subject_id}_{epoch_duration}_{start_idx}.pkl"
    feature_df.to_pickle(batch_feature_save_path)

    del batch_segmented_df, feature_df
    gc.collect()

    return batch_feature_save_path


def batchwise_featureConstruction(subject_id, segmented_df, signal_list, epoch_size_dict, epoch_duration, batch_size):

    batch_feature_files = []
    futures = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        for start_idx in range(0, segmented_df.shape[0], batch_size):
            # SUbmit each task as a separate proces
            future = executor.submit(
                process_batch,
                subject_id,
                epoch_duration,
                start_idx,
                # Only pass the batch
                segmented_df[start_idx:start_idx +
                             batch_size].reset_index(drop=True),
                signal_list,
                epoch_size_dict
            )
            futures.append(future)

        for future in as_completed(futures):
            result_path = future.result()
            batch_feature_files.append(result_path)
            print(f"Saved batch: {result_path}")

    # Sort files based on the start_idx
    batch_feature_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Merge all batches
    all_feature_dfs = [pd.read_pickle(f) for f in batch_feature_files]
    full_feature_df = pd.concat(all_feature_dfs, ignore_index=True)
    feature_df_save_path = f"{RAW_FEATURE_PATH}/feature_df_{subject_id}_{epoch_duration}.pkl"
    full_feature_df.to_pickle(feature_df_save_path)

    print(f"Completed Feature Construction for Subject {subject_id}.")

    # Clean up batch files
    for f in batch_feature_files:
        os.remove(f)


def main(subject_id, epoch_duration, batch_size):
    loader = DataLoader(SEGDATA_PATH)

    epoch_size_dict = {
        "bvp": FS["bvp"] * epoch_duration,
        "acc": FS["acc"] * epoch_duration,
        "eda": FS["eda"] * epoch_duration,
        "hr": FS["hr"] * epoch_duration,
        "temp": FS["temp"] * epoch_duration,
    }

    e4_signal_list = list(epoch_size_dict.keys())

    # Load the segmented data for the entire subject
    segmented_df = loader.load_segmented_dfs(subject_id, epoch_duration)

    # Process the glucose data in batches and save a single file
    batchwise_featureConstruction(
        subject_id, segmented_df, e4_signal_list, epoch_size_dict, epoch_duration, batch_size)


if __name__ == "__main__":
    subject_id_ = int(sys.argv[1])
    epoch_duration_minutes_ = int(sys.argv[2])
    epoch_duration_seconds_ = epoch_duration_minutes_ * 60
    batch_size_ = int(sys.argv[3])
    main(subject_id=subject_id_, epoch_duration=epoch_duration_seconds_, batch_size=batch_size_)

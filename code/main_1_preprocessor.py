import gc
import os
import sys

import pandas as pd
from config.config import DATA_PATH, FS, SEGDATA_PATH
from utils.import_classes import load_class

DataLoader = load_class("data_loader")
DataSegmenter = load_class("data_segmenter")
SignalProcessor = load_class("signal_processor")

def process_in_batches(subject_id, glucose_df, epoch_size_dict, epoch_duration, batch_size):
    loader = DataLoader(DATA_PATH)
    processor = SignalProcessor()

    # Load and process signals
    e4_dfs_dict = {
        'hr': processor.process_hr(loader.load_E4_data(subject_id, "HR", "datetime", [" hr"]), "hr"),
        'temp': processor.process_temp(loader.load_E4_data(subject_id, "TEMP", "datetime", [" temp"]), "temp"),
        'eda': processor.process_eda(loader.load_E4_data(subject_id, "EDA", "datetime", [" eda"]), "eda"),
        'acc': processor.process_acc(loader.load_E4_data(subject_id, "ACC", "datetime", [" acc_x", " acc_y", " acc_z"]), "acc"),
        'bvp': processor.process_bvp(loader.load_E4_data(subject_id, "BVP", "datetime", [" bvp"]), "bvp")
    }

    segmenter = DataSegmenter(subject_id=subject_id)
    batch_files = []  # Track saved batch files


    for start_idx in range(0, glucose_df.shape[0], batch_size):
        batch_glucose_df = glucose_df[start_idx:start_idx + batch_size]
        segmented_df = segmenter.process_all(
            e4_dfs_dict, batch_glucose_df, epoch_size_dict, FS)

        # Save batch separately
        batch_save_path = f"{SEGDATA_PATH}/batch_{subject_id}_{epoch_duration}_{start_idx}.pkl"
        segmented_df.to_pickle(batch_save_path)
        batch_files.append(batch_save_path)

        print(
            f"Processed and saved batch {start_idx} to {start_idx + batch_size} for Subject {subject_id}.")

        del segmented_df, batch_glucose_df  # Free memory
        gc.collect()

    # Merge all batches
    all_dfs = [pd.read_pickle(f) for f in batch_files]
    full_segmented_df = pd.concat(all_dfs, ignore_index=True)
    final_save_path = f"{SEGDATA_PATH}/segmented_df_{subject_id}_{epoch_duration}.pkl"
    full_segmented_df.to_pickle(final_save_path)

    print(
        f"Saved full segmented data for Subject {subject_id} at {final_save_path}.")

    # Delete batch files
    for f in batch_files:
        os.remove(f)


def main(subject_id, epoch_duration, batch_size):
    loader = DataLoader(DATA_PATH)

    epoch_size_dict = {
        "bvp": FS["bvp"] * epoch_duration,
        "acc": FS["acc"] * epoch_duration,
        "eda": FS["eda"] * epoch_duration,
        "hr": FS["hr"] * epoch_duration,
        "temp": FS["temp"] * epoch_duration,
    }

    # Load the glucose data for the entire subject
    glucose_df = loader.load_glucose_data(subject_id)

    # Process the glucose data in batches and save a single file
    process_in_batches(subject_id, glucose_df,
                       epoch_size_dict, epoch_duration, batch_size)


if __name__ == "__main__":
    subject_id_ = int(sys.argv[1])
    epoch_duration_minutes_ = int(sys.argv[2])
    epoch_duration_seconds_ = epoch_duration_minutes_ * 60
    batch_size_ = int(sys.argv[3])
    main(subject_id=subject_id_, epoch_duration=epoch_duration_seconds_, batch_size=batch_size_)

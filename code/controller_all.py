import subprocess
import sys

from config.config import BASE_PATH

base_path = BASE_PATH

subject_ids = range(1, 17)
batch_size_PP = 100
batch_size_FE = 200


# for epoch_duration in ["5", "120"]:
for epoch_duration in ["5"]:
    for subject_id in subject_ids:
        print("==============================================================")
        print(f"Preprocessing Subject {subject_id}...")
        subprocess.run([sys.executable, f"{BASE_PATH}/code/main_1_preprocessor.py", str(
            subject_id), epoch_duration, str(batch_size_PP)], check=True)
        print(f"Finished Preprocessing Subject {subject_id}.")

        print("==============================================================")
        print(f"Extracting Features for Subject {subject_id}...")
        subprocess.run([sys.executable, f"{base_path}/code/main_2_featureExtractor.py", str(
        subject_id), epoch_duration, str(batch_size_FE)], check=True)
        print(f"Finished Feature Extraction for Subject {subject_id}.")

        print("==============================================================")
        print(f"Cleaning Features for Subject {subject_id}...")
        subprocess.run([sys.executable, f"{base_path}/code/main_3_featureCleaner.py", str(
        subject_id), epoch_duration], check=True)
        print(f"Finished Feature Cleaning for Subject {subject_id}.")

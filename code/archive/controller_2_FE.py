import subprocess
import sys

from config.config import BASE_PATH

base_path = BASE_PATH

subject_ids = range(1, 2)
batch_size = 200

# for epoch_duration in ["5", "120"]:
for epoch_duration in ["5"]:
    for subject_id in subject_ids:
        print(f"Processing Subject {subject_id}...")
        subprocess.run([sys.executable, f"{base_path}/code/main_2_featureExtractor.py", str(
            subject_id), str(epoch_duration), str(batch_size)], check=True)
        print(f"Finished processing Subject {subject_id}.")

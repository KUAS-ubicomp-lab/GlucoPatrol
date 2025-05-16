import subprocess
import sys

from config.config import BASE_PATH

base_path = BASE_PATH

subject_ids = range(1, 17)
batch_size = 200

# for epoch_duration in ["5MIN", "2H"]:
for epoch_duration in ["5MIN"]:
    for subject_id in subject_ids:
        print(f"Cleaning features for subject {subject_id}...")
        subprocess.run([sys.executable, f"{base_path}/code/main_3_featureCleaner.py", str(
            subject_id), epoch_duration], check=True)

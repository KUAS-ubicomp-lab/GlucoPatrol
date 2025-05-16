import os
from config.config import (
    SEGDATA_PATH, RAW_FEATURE_PATH, CLEAN_FEATURE_PATH, RESULTS_PATH
)

ALL_PATHS = [
    SEGDATA_PATH,
    RAW_FEATURE_PATH,
    CLEAN_FEATURE_PATH,
    RESULTS_PATH
]

def create_dirs(paths, overwrite=False, remove_only=False):
    for path in paths:
        if remove_only:
            if os.path.exists(path):
                print(f"Removing: {path}")
                os.system(f"rm -rf {path}")
            else:
                print(f"Folder does not exist: {path}")
        else:
            if os.path.exists(path):
                if overwrite:
                    print(f"Overwriting: {path}")
                    os.system(f"rm -rf {path}")
                    os.makedirs(path)
                    print(f"Recreated: {path}")
                else:
                    print(f"Exists OK: {path}")
            else:
                os.makedirs(path)
                print(f"Created: {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Delete existing folders before creating")
    parser.add_argument("--remove-only", action="store_true", help="Remove existing folders without recreating them")
    args = parser.parse_args()

    create_dirs(ALL_PATHS, overwrite=args.overwrite, remove_only=args.remove_only)

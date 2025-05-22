import os

from config.config import CLEAN_FEATURE_PATH, RESULTS_PATH
from utils.import_classes import load_class

DataLoader = load_class("data_loader")
ModelTrainer = load_class("model_trainer")

def main():
    for subject_id in range(1, 17):
        print(f"Processing subject {subject_id}...")

        loader = DataLoader(CLEAN_FEATURE_PATH)
        feature_df = loader.load_cleaned_feature_dfs(subject_id, '300') # TODO: refactor

        subject_results_dir = os.path.join(RESULTS_PATH, str(subject_id))
        model_trainer = ModelTrainer(
            results_dir=subject_results_dir, subject_id=subject_id, folder_name="XGBoost", random_seed=42)

        # Train and evaluate models
        summary = model_trainer.train_all(
            feature_df, target_column='glucose')
        print(f"Saved FL results for subject {subject_id}.\n")


if __name__ == "__main__":
    main()

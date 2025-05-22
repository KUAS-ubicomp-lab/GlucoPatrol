import json
import os
import pickle
from datetime import datetime

import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .eval.eval import ModelEvaluator

matplotlib.use('Agg')


class ModelTrainer:
    def __init__(self, results_dir, subject_id, folder_name, random_seed):
        self.results_dir = results_dir
        self.folder_name  = folder_name
        self.random_seed = random_seed
        os.makedirs(f"{self.results_dir}/{folder_name}/", exist_ok=True)

        self.subject_id = subject_id

        self.models = {
            # "LinearRegression": {
            #     "model": LinearRegression(),
            #     "params": {}
            # },
            # "Ridge": {
            #     "model": Ridge(),
            #     "params": {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]}
            # },
            # "Lasso": {
            #     "model": Lasso(max_iter=10000),
            #     "params": {"regressor__alpha": [0.001, 0.01, 0.1, 1.0]}
            # },
            # "RandomForest": {
            #     "model": RandomForestRegressor(random_state=42),
            #     "params": {
            #         "regressor__n_estimators": [10, 25, 50, 100],
            #         "regressor__max_depth": [2, 4, 6, 8, 10]
            #     }
            # },
            # "XGBoost": {
            #     "model": XGBRegressor(random_state=42),
            #     "params": {
            #         "regressor__n_estimators": [10, 15, 25],
            #         "regressor__max_depth": [2, 4, 6],
            #         "regressor__learning_rate": [0.01, 0.1, 0.15]
            #     }
            # }

            # SAMPLE - XGBoost
            "XGBoost": {
                "model": XGBRegressor(random_state=self.random_seed),
                "params": {
                    "regressor__n_estimators": [10, 25],
                    "regressor__max_depth": [2, 4],
                    "regressor__learning_rate": [0.1, 0.3]
                }
            }
        }

    def train_all(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        summary_log = {}

        for name, spec in self.models.items():
            print(f"Training model: {name}")
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler()),
                ("regressor", spec["model"])
            ])

            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_seed)

            grid = GridSearchCV(
                pipe, spec["params"], cv=cv, scoring="r2", n_jobs=25, return_train_score=True)
            grid.fit(X, y)

            best_model = grid.best_estimator_
            best_params = grid.best_params_

            y_pred = best_model.predict(X)

            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(y, y_pred)

            # Save predictions
            pred_df = pd.DataFrame({"actual": y, "predicted": y_pred})
            pred_path = f"{self.results_dir}/{self.folder_name}/{name}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            # Save model
            model_path = f"{self.results_dir}/{self.folder_name}/{name}_best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)

            summary_log[name] = {
                "best_params": best_params,
                "cv_metrics": {
                    "cv_R2_train": grid.best_score_,
                    "cv_R2_test": grid.cv_results_['mean_train_score'][grid.best_index_],
                },
                "prediction_metrics": metrics
            }

        # Save global summary
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.results_dir}/{self.folder_name}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_log, f, indent=4)

        # Save CV results
        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_df.to_csv(f"{self.results_dir}/{self.folder_name}/cv_results.csv", index=False)

        return summary_log
    

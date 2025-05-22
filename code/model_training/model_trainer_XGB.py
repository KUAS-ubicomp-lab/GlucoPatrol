import json
import os
import pickle

import matplotlib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .eval.eval import ModelEvaluator

matplotlib.use('Agg')


class ModelTrainer:
    def __init__(self, results_dir, subject_id, folder_name, random_seed, n_jobs=28):
        self.results_dir = results_dir
        self.folder_name  = folder_name
        self.random_seed = random_seed
        self.n_jobs_ = n_jobs
        os.makedirs(f"{self.results_dir}/{folder_name}/", exist_ok=True)

        self.subject_id = subject_id


    def train_all(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        summary_log = {}

        print(f"Training model: {self.folder_name}")
        pipe1 = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(objective='reg:squarederror', verbosity=1, random_state=self.random_seed))
        ])

        pipe2 = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(objective='reg:squarederror', verbosity=1, random_state=self.random_seed))
        ])

        pipe3 = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(objective='reg:squarederror', verbosity=1, random_state=self.random_seed))
        ])

        pipe4 = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(objective='reg:squarederror', verbosity=1, random_state=self.random_seed))
        ])

        pipe5 = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(objective='reg:squarederror', verbosity=1, random_state=self.random_seed))
        ])
        
        # Setup cross-validation
        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=self.random_seed)

        # Step 1 - Broad Search
        # param_grid1 = {
        #     'regressor__n_estimators': np.arange(10, 200, 20),
        #     'regressor__learning_rate': [0.025, 0.05, 0.1, 0.3],
        #     'regressor__max_depth': [4, 6, 8, 10],
        #     'regressor__gamma': [0],  # not important yet
        #     'regressor__subsample': [1.0],
        #     'regressor__colsample_bytree': [1.0],
        #     'regressor__min_child_weight': [1]
        # }

        param_grid1 = {
            'regressor__n_estimators': list(range(10, 100, 50)),
            'regressor__learning_rate': [0.05, 0.1, 0.3],
            'regressor__max_depth': [4, 6, 8],
            'regressor__gamma': [0],  # not important yet
            'regressor__subsample': [1.0],
            'regressor__colsample_bytree': [1.0],
            'regressor__min_child_weight': [1]
        }

        grid1 = GridSearchCV(pipe1, param_grid1, cv=cv, scoring='r2', n_jobs=self.n_jobs_, verbose=1)
        grid1.fit(X, y)
        print("Finished fitting Grid 1")

        best_depth = grid1.best_params_['regressor__max_depth']
        depth_range = list(range(max(2, best_depth - 1), best_depth + 2))

        param_grid2 = {
            'regressor__n_estimators': list(range(10, 100, 50)),
            'regressor__learning_rate': [grid1.best_params_['regressor__learning_rate']],
            'regressor__max_depth': depth_range,
            'regressor__min_child_weight': [1, 2, 3],
            'regressor__gamma': [0],
            'regressor__subsample': [1.0],
            'regressor__colsample_bytree': [1.0],
        }

        grid2 = GridSearchCV(pipe2, param_grid2, cv=cv, scoring='r2', n_jobs=self.n_jobs_, verbose=1)
        grid2.fit(X, y)
        print("Finished fitting Grid 2")

        # Step 3 - Tune colsample_bytree and subsample
        # param_grid3 = {
        #     'regressor__n_estimators': np.arange(10, 100, 20),
        #     'regressor__learning_rate': [grid1.best_params_['learning_rate']],
        #     'regressor__max_depth': [grid2.best_params_['max_depth']],
        #     'regressor__min_child_weight': [grid2.best_params_['min_child_weight']],
        #     'regressor__gamma': [0],
        #     'regressor__colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        #     'regressor__subsample': [0.5, 0.75, 1.0],
        # }

        param_grid3 = {
            'regressor__n_estimators': list(range(10, 100, 50)),
            'regressor__learning_rate': [grid1.best_params_['regressor__learning_rate']],
            'regressor__max_depth': [grid2.best_params_['regressor__max_depth']],
            'regressor__min_child_weight': [grid2.best_params_['regressor__min_child_weight']],
            'regressor__gamma': [0],
            'regressor__colsample_bytree': [0.8, 1.0],
            'regressor__subsample': [0.5, 1.0],
        }

        grid3 = GridSearchCV(pipe3, param_grid3, cv=cv, scoring='r2', n_jobs=self.n_jobs_, verbose=1)
        grid3.fit(X, y)
        print("Finished fitting Grid 3")

        # Step 4 - Tune gamma
        # param_grid4 = {
        #     'regressor__n_estimators': np.arange(10, 100, 20),
        #     'regressor__learning_rate': [grid1.best_params_['learning_rate']],
        #     'regressor__max_depth': [grid2.best_params_['max_depth']],
        #     'regressor__min_child_weight': [grid2.best_params_['min_child_weight']],
        #     'regressor__gamma': [0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0],
        #     'regressor__colsample_bytree': [grid3.best_params_['colsample_bytree']],
        #     'regressor__subsample': [grid3.best_params_['subsample']],
        # }

        param_grid4 = {
            'regressor__n_estimators': list(range(10, 100, 50)),
            'regressor__learning_rate': [grid1.best_params_['regressor__learning_rate']],
            'regressor__max_depth': [grid2.best_params_['regressor__max_depth']],
            'regressor__min_child_weight': [grid2.best_params_['regressor__min_child_weight']],
            'regressor__gamma': [0.1, 0.5, 1.0],
            'regressor__colsample_bytree': [grid3.best_params_['regressor__colsample_bytree']],
            'regressor__subsample': [grid3.best_params_['regressor__subsample']],
        }

        grid4 = GridSearchCV(pipe4, param_grid4, cv=cv, scoring='r2', n_jobs=self.n_jobs_, verbose=1)
        grid4.fit(X, y)
        print("Finished fitting Grid 4")

        # Step 5 - Final tuning: long n_estimators + smaller learning rate
        # param_grid5 = {
        #     'regressor__n_estimators': np.arange(10, 100, 20),
        #     'regressor__learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
        #     'regressor__max_depth': [grid2.best_params_['max_depth']],
        #     'regressor__min_child_weight': [grid2.best_params_['min_child_weight']],
        #     'regressor__gamma': [grid4.best_params_['gamma']],
        #     'regressor__colsample_bytree': [grid3.best_params_['colsample_bytree']],
        #     'regressor__subsample': [grid3.best_params_['subsample']],
        # }

        param_grid5 = {
            'regressor__n_estimators': list(range(10, 100, 50)),
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [grid2.best_params_['regressor__max_depth']],
            'regressor__min_child_weight': [grid2.best_params_['regressor__min_child_weight']],
            'regressor__gamma': [grid4.best_params_['regressor__gamma']],
            'regressor__colsample_bytree': [grid3.best_params_['regressor__colsample_bytree']],
            'regressor__subsample': [grid3.best_params_['regressor__subsample']],
        }

        grid_last = GridSearchCV(pipe5, param_grid5, cv=cv, scoring='r2', n_jobs=self.n_jobs_, verbose=1, return_train_score=True)
        grid_last.fit(X, y)
        print("Finished fitting Grid Last")

        pipe_last = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(objective='reg:squarederror', verbosity=1, random_state=self.random_seed))
        ])

        pipe_last.set_params(**grid_last.best_params_)

        # Final model with best parameters
        final_model = pipe_last
        final_model.fit(X, y)

        # Predict on test set
        y_pred = final_model.predict(X)

        best_model = grid_last.best_estimator_
        best_params = grid_last.best_params_

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y, y_pred)

        # Save predictions
        pred_df = pd.DataFrame({"actual": y, "predicted": y_pred})
        pred_path = f"{self.results_dir}/{self.folder_name}/predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # Save model
        model_path = f"{self.results_dir}/{self.folder_name}/best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        summary_log[self.folder_name] = {
            "best_params": best_params,
            "cv_metrics": {
                "cv_R2_train": grid_last.best_score_,
                "cv_R2_test": grid_last.cv_results_['mean_train_score'][grid_last.best_index_],
            },
            "prediction_metrics": metrics
        }

        # Save global summary
        with open(f"{self.results_dir}/{self.folder_name}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_log, f, indent=4)

        # Save CV results
        full_cv_results_df = pd.DataFrame(grid_last.cv_results_)
        main_cv_results_df = full_cv_results_df.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'])

        main_cv_results_df.to_csv(f"{self.results_dir}/{self.folder_name}/cv_results.csv", index=False)
        full_cv_results_df.to_csv(f"{self.results_dir}/{self.folder_name}/full_cv_results.csv", index=False)

        return summary_log
    

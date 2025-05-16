import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, model_name=""):
        self.model_name = model_name
        self.metrics = {}
        self.y_true = None
        self.y_pred = None

    def evaluate(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        nrmse = rmse / (np.std(y_true))
        mard = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        self.metrics = {
            "Model": self.model_name,
            "R2": r2,
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "MARD": mard,
            "MAE": mae
        }

        return self.metrics

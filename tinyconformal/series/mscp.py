import numpy as np


class MultiStepSplitConformal:
    def __init__(self, base_model, horizon: int, alpha: float = 0.1):
        """
        MSCP: Calcula quantis calibrados específicos para cada passo h.
        """
        self.base_model = base_model
        self.horizon = horizon
        self.alpha = alpha
        self.quantiles_ = None

    def fit_calibration(self, X_cal, Y_cal):
        """
        Y_cal deve ter formato (N_amostras, horizon)
        """
        preds = self.base_model.predict(X_cal)
        residuals = np.abs(Y_cal - preds)

        n = len(Y_cal)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))

        # Calcula o quantil (1 - alpha) de erro para CADA h
        self.quantiles_ = np.quantile(residuals, q_level, axis=0)

    def predict_interval(self, X_test):
        preds = self.base_model.predict(X_test)
        lower = preds - self.quantiles_
        upper = preds + self.quantiles_
        return preds, lower, upper

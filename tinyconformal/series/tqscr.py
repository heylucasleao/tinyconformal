# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List
from sklearn.base import RegressorMixin, BaseEstimator
from .base import BaseTimeSeriesConformalRegressor


class ConformalQuantileTimeSeriesRegressor(
    RegressorMixin, BaseEstimator, BaseTimeSeriesConformalRegressor
):
    """
    Time Series Conformal Quantile Regression (TSCQR) wrapping Nixtla estimators.

    Parameters
    ----------
    quantile_cols : Tuple[str, str]
        Obrigatório. Tupla contendo o nome exato das colunas correspondentes
        aos quantis inferior e superior gerados pelo learner.
        Exemplo: ('LightGBM-lo-90', 'LightGBM-hi-90')
    point_col : str, optional
        Nome da coluna da predição pontual (mediana ou média), se houver.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        quantile_cols: Tuple[str, str],
        point_col: Optional[str] = None,
        n_windows: int = 3,
        alpha: float = 0.05,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        if not isinstance(quantile_cols, (tuple, list)) or len(quantile_cols) != 2:
            raise ValueError(
                "quantile_cols deve ser uma tupla/lista com 2 nomes de colunas: (low, high)."
            )

        self.quantile_cols = quantile_cols
        self.point_col = point_col

    def _sample_correction(self, alpha: float) -> float:
        """Calcula o nível do quantil com correção de amostra finita."""
        q_level = np.ceil((self.n + 1) * (1.0 - alpha)) / self.n
        return float(np.clip(q_level, 0.0, 1.0))

    def _generate_residuals(
        self, cv_df: pd.DataFrame, y_val_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Calcula o CQR Nonconformity Score: E_t = max(q_low - y, y - q_high)
        """
        low_col, high_col = self.quantile_cols

        if low_col not in cv_df.columns or high_col not in cv_df.columns:
            raise KeyError(
                f"As colunas {self.quantile_cols} não foram encontradas no resultado do backtesting. "
                f"Colunas disponíveis: {list(cv_df.columns)}"
            )

        q_low = cv_df[low_col].to_numpy()
        q_high = cv_df[high_col].to_numpy()
        y_true = y_val_df[self.target_col].to_numpy()

        # CQR Score: E_t = max(q_low - y, y - q_high)
        scores = np.maximum(q_low - y_true, y_true - q_high)
        return scores

    def _compute_bounds(
        self,
        q_low: np.ndarray,
        q_high: np.ndarray,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes lower and upper conformalized quantile bounds:
        [q_low - q_hat, q_high + q_hat]
        """
        eff_alpha = self._get_alpha(alpha)
        q_level = self._sample_correction(eff_alpha)
        q_hat = self._compute_qhat(self.residuals_, q_level)

        lower_bound = q_low - q_hat
        upper_bound = q_high + q_hat

        return lower_bound, upper_bound

    def predict_interval(
        self,
        X_df: Optional[pd.DataFrame] = None,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Aplica a correção conformal aos quantis nativos e retorna um DataFrame
        estruturado padrão Nixtla.
        """
        alpha = self._get_alpha(alpha)
        h = self._get_horizon(h)

        pred_df = (
            self._invoke(
                self.learner.predict,
                h=h,
                X_df=X_df,
            )
            .sort_values(by=[self.id_col, self.time_col])
            .reset_index(drop=True)
        )

        low_col, high_col = self.quantile_cols

        q_low = pred_df[low_col].to_numpy()
        q_high = pred_df[high_col].to_numpy()
        n_series = pred_df[self.id_col].nunique()

        # Infere o nome base do modelo eliminando os sufixos de quantil se houver
        model_name = low_col.split("-")[0]

        lower_bound, upper_bound = self._compute_bounds(
            q_low=q_low,
            q_high=q_high,
            model_name=model_name,
            h=h,
            n_series=n_series,
            alpha=alpha,
        )

        eff_alpha = self._get_alpha(alpha)
        level = int(round((1 - eff_alpha) * 100))

        pred_df[f"{model_name}-lo-{level}"] = lower_bound
        pred_df[f"{model_name}-hi-{level}"] = upper_bound

        return pred_df

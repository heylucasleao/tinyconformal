# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import re


class ConformalMetalogNewsvendor(BaseEstimator):
    """
    Decisor do Novo-Jornaleiro via Metalog sobre Regressor Conformalizado Fitado.

    Parameters
    ----------
    conformal_regressor : BaseEstimator
        Instância do regressor conformal JÁ FITADA.
    interval_pairs : Tuple[str, str] ou List[Tuple[str, str]]
        Tupla (ou lista de tuplas) com os nomes das colunas dos quantis extremos (lower_col, upper_col).
    cu_col : str
        Nome da coluna contendo o custo de falta no X_df.
    co_col : str
        Nome da coluna contendo o custo de sobra no X_df.
    median_cols : str, List[str], optional
        Nome da coluna (ou lista de colunas) da mediana/ponto central.
        Se informada, ativa Metalog de 3 termos. Deve ter o mesmo comprimento de interval_pairs.
    level : float, default=90.0
        Nível de cobertura nominal dos intervalos em porcentagem (ex: 90.0 para [P5, P95], 80.0 para [P10, P90]).
    id_col : str, default="unique_id"
        Coluna identificadora do SKU/Série.
    """

    def __init__(
        self,
        conformal_regressor: BaseEstimator,
        interval_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        cu_col: str,
        co_col: str,
        median_cols: Union[str, List[str], None] = None,
        level: float = 90.0,
        id_col: str = "unique_id",
    ):
        self.conformal_regressor = conformal_regressor
        self.cu_col = cu_col
        self.co_col = co_col
        self.level = level
        self.id_col = id_col

        self.interval_pairs_ = self._validate_interval_pairs(interval_pairs)
        self.median_cols_ = self._validate_median_cols(
            median_cols, num_pairs=len(self.interval_pairs_)
        )
        self.pair_mappings_ = self._build_pair_mappings(
            self.interval_pairs_, self.median_cols_
        )

        if not (0.0 < self.level < 100.0):
            raise ValueError("O nível (level) deve estar entre 0 e 100.")

        self.p_low_ = (100.0 - self.level) / 200.0
        self.p_high_ = 1.0 - self.p_low_
        self._logit_level = float(np.log(self.p_high_ / self.p_low_))

    @staticmethod
    def _validate_interval_pairs(
        interval_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        pattern = re.compile(r"^.+-(lo|hi)-\d+.*$")

        if (
            isinstance(interval_pairs, tuple)
            and len(interval_pairs) == 2
            and isinstance(interval_pairs[0], str)
            and isinstance(interval_pairs[1], str)
        ):
            pairs = [interval_pairs]
        elif isinstance(interval_pairs, list) and all(
            isinstance(pair, (tuple, list))
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
            for pair in interval_pairs
        ):
            pairs = [tuple(pair) for pair in interval_pairs]
        else:
            raise ValueError(
                "interval_pairs deve ser uma tupla de 2 strings (low, high) ou uma lista dessas tuplas."
            )

        for low, high in pairs:
            if not pattern.match(low) or not pattern.match(high):
                raise ValueError(
                    f"As colunas do intervalo ('{low}', '{high}') devem seguir o padrão '<model>-(lo|hi)-<level>'."
                )

        return pairs

    @staticmethod
    def _validate_interval_pairs(
        interval_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        if (
            isinstance(interval_pairs, tuple)
            and len(interval_pairs) == 2
            and isinstance(interval_pairs[0], str)
            and isinstance(interval_pairs[1], str)
        ):
            return [interval_pairs]
        elif isinstance(interval_pairs, list) and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
            for pair in interval_pairs
        ):
            return interval_pairs

        raise ValueError(
            "interval_pairs deve ser uma tupla de 2 strings (low, high) ou uma lista dessas tuplas."
        )

    @staticmethod
    def _validate_median_cols(
        median_cols: Union[str, List[str], None], num_pairs: int
    ) -> List[Optional[str]]:
        if isinstance(median_cols, str):
            normalized_medians = [median_cols]
        elif isinstance(median_cols, list) and all(
            isinstance(col, str) for col in median_cols
        ):
            normalized_medians = median_cols
        elif median_cols is None:
            normalized_medians = [None] * num_pairs
        else:
            raise TypeError("median_cols deve ser str, lista de str ou None.")

        if len(normalized_medians) != num_pairs:
            raise ValueError(
                f"O número de median_cols ({len(normalized_medians)}) deve ser igual "
                f"ao número de interval_pairs ({num_pairs})."
            )

        return normalized_medians

    @staticmethod
    def _build_pair_mappings(
        interval_pairs: List[Tuple[str, str]], median_cols: List[Optional[str]]
    ) -> Dict[int, Dict[str, Optional[str]]]:
        return {
            idx: {
                "low": q_pair[0],
                "high": q_pair[1],
                "median": med,
            }
            for idx, (q_pair, med) in enumerate(zip(interval_pairs, median_cols))
        }

    def _eval_metalog_spt(
        self,
        p_star: np.ndarray,
        p_low_cqr: np.ndarray,
        p_high_cqr: np.ndarray,
        p_low_base: Optional[np.ndarray] = None,
        p_high_base: Optional[np.ndarray] = None,
        p50_base: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        p_clipped = np.clip(p_star, 1e-5, 1.0 - 1e-5)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))

        has_3_terms = (
            p50_base is not None and p_low_base is not None and p_high_base is not None
        )

        if has_3_terms:
            base_spread = np.maximum(p_high_base - p_low_base, 1e-6)
            skew_ratio = np.clip((p50_base - p_low_base) / base_spread, 0.01, 0.99)
            p50_adj = p_low_cqr + skew_ratio * (p_high_cqr - p_low_cqr)

            a1 = p50_adj
            a2 = (p_high_cqr - p_low_cqr) / (2.0 * self._logit_level)

            a3_denom = (self.p_high_ - 0.5) * self._logit_level
            a3 = (p_high_cqr + p_low_cqr - 2.0 * p50_adj) / a3_denom

            y_star = a1 + a2 * logit_p + a3 * (p_clipped - 0.5) * logit_p
            return y_star, p50_adj
        else:
            a1 = 0.5 * (p_low_cqr + p_high_cqr)
            a2 = (p_high_cqr - p_low_cqr) / (2.0 * self._logit_level)

            y_star = a1 + a2 * logit_p
            return y_star, None

    def predict_optimal_quantity(
        self,
        X_df: pd.DataFrame,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        df_pred = self.conformal_regressor.predict_interval(h=h, alpha=alpha, X_df=X_df)

        c_u = X_df[self.cu_col].to_numpy(dtype=float)
        c_o = X_df[self.co_col].to_numpy(dtype=float)

        if np.any(c_u <= 0) or np.any(c_o <= 0):
            raise ValueError(
                "Os valores de Cu e Co no X_df devem ser estritamente positivos."
            )

        p_star = c_u / (c_u + c_o)
        df_pred["p_star"] = p_star

        for idx, mapping in self.pair_mappings_.items():
            low_col = mapping["low"]
            high_col = mapping["high"]
            med_col = mapping["median"]

            p_low_cqr = df_pred[low_col].to_numpy(dtype=float)
            p_high_cqr = df_pred[high_col].to_numpy(dtype=float)

            if med_col is not None:
                p_low_base = p_low_cqr
                p_high_base = p_high_cqr
                p50_base = df_pred[med_col].to_numpy(dtype=float)
            else:
                p_low_base = p_high_base = p50_base = None

            y_star, p50_adj = self._eval_metalog_spt(
                p_star=p_star,
                p_low_cqr=p_low_cqr,
                p_high_cqr=p_high_cqr,
                p_low_base=p_low_base,
                p_high_base=p_high_base,
                p50_base=p50_base,
            )

            suffix = f"_{idx}" if len(self.pair_mappings_) > 1 else ""

            if p50_adj is not None:
                df_pred[f"p50_conformal_adj{suffix}"] = p50_adj

            df_pred[f"y_optimal{suffix}"] = np.maximum(0.0, y_star)

        return df_pred

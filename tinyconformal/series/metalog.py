# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ConformalMetalogNewsvendor(BaseEstimator):
    """
    Decisor do Novo-Jornaleiro via Metalog sobre Regressor Conformalizado Fitado.

    Parameters
    ----------
    conformal_regressor : BaseEstimator
        Instância do regressor conformal JÁ FITADA.
    quantile_pairs : Tuple[str, str] ou List[Tuple[str, str]]
        Tupla (ou lista de tuplas) com os nomes das colunas dos quantis extremos (lower_col, upper_col).
    cu_col : str
        Nome da coluna contendo o custo de falta no X_df.
    co_col : str
        Nome da coluna contendo o custo de sobra no X_df.
    median_cols : str, List[str], optional
        Nome da coluna (ou lista de colunas) da mediana/ponto central.
        Se informada, ativa Metalog de 3 termos. Deve ter o mesmo comprimento de quantile_pairs.
    id_col : str, default="unique_id"
        Coluna identificadora do SKU/Série.
    time_col : str, default="ds"
        Coluna de data/tempo.
    """

    def __init__(
        self,
        conformal_regressor: BaseEstimator,
        quantile_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        cu_col: str,
        co_col: str,
        median_cols: Union[str, List[str], None] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ):
        self.conformal_regressor = conformal_regressor
        self.cu_col = cu_col
        self.co_col = co_col
        self.id_col = id_col
        self.time_col = time_col

        self.quantile_pairs_ = self._validate_quantile_pairs(quantile_pairs)
        self.median_cols_ = self._validate_median_cols(
            median_cols, num_pairs=len(self.quantile_pairs_)
        )
        self.pair_mappings_ = self._build_pair_mappings(
            self.quantile_pairs_, self.median_cols_
        )

        self._logit_95 = np.log(0.95 / 0.05)  # ~ 2.9444

    @staticmethod
    def _validate_quantile_pairs(
        quantile_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        """Valida e normaliza o parâmetro quantile_pairs para uma lista de tuplas."""
        if (
            isinstance(quantile_pairs, tuple)
            and len(quantile_pairs) == 2
            and isinstance(quantile_pairs[0], str)
            and isinstance(quantile_pairs[1], str)
        ):
            return [quantile_pairs]
        elif isinstance(quantile_pairs, list) and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
            for pair in quantile_pairs
        ):
            return quantile_pairs

        raise ValueError(
            "quantile_pairs deve ser uma tupla de 2 strings (low, high) ou uma lista dessas tuplas."
        )

    @staticmethod
    def _validate_median_cols(
        median_cols: Union[str, List[str], None], num_pairs: int
    ) -> List[Optional[str]]:
        """Valida, normaliza e verifica a cardinalidade de median_cols."""
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
                f"ao número de quantile_pairs ({num_pairs})."
            )

        return normalized_medians

    @staticmethod
    def _build_pair_mappings(
        quantile_pairs: List[Tuple[str, str]], median_cols: List[Optional[str]]
    ) -> Dict[int, Dict[str, Optional[str]]]:
        """Constrói o dicionário de mapeamento das colunas por índice."""
        return {
            idx: {
                "low": q_pair[0],
                "high": q_pair[1],
                "median": med,
            }
            for idx, (q_pair, med) in enumerate(zip(quantile_pairs, median_cols))
        }

    def _eval_metalog_spt(
        self,
        p_star: np.ndarray,
        p5_cqr: np.ndarray,
        p95_cqr: np.ndarray,
        p5_base: Optional[np.ndarray] = None,
        p95_base: Optional[np.ndarray] = None,
        p50_base: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Avalia o quantil Q(p_star) da Metalog de 2 ou 3 termos."""
        p_clipped = np.clip(p_star, 1e-5, 1.0 - 1e-5)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))

        has_3_terms = (
            p50_base is not None and p5_base is not None and p95_base is not None
        )

        if has_3_terms:
            base_spread = np.maximum(p95_base - p5_base, 1e-6)
            skew_ratio = np.clip((p50_base - p5_base) / base_spread, 0.01, 0.99)
            p50_adj = p5_cqr + skew_ratio * (p95_cqr - p5_cqr)

            a1 = p50_adj
            a2 = (p95_cqr - p5_cqr) / (2.0 * self._logit_95)
            a3 = (p95_cqr + p5_cqr - 2.0 * p50_adj) / (0.45 * self._logit_95)

            y_star = a1 + a2 * logit_p + a3 * (p_clipped - 0.5) * logit_p
            return y_star, p50_adj
        else:
            a1 = 0.5 * (p5_cqr + p95_cqr)
            a2 = (p95_cqr - p5_cqr) / (2.0 * self._logit_95)

            y_star = a1 + a2 * logit_p
            return y_star, None

    def predict_optimal_quantity(
        self,
        X_df: pd.DataFrame,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Calcula a quantidade ótima y* para cada conjunto de par configurado."""
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

            p5_cqr = df_pred[low_col].to_numpy(dtype=float)
            p95_cqr = df_pred[high_col].to_numpy(dtype=float)

            if med_col is not None:
                p5_base = p5_cqr
                p95_base = p95_cqr
                p50_base = df_pred[med_col].to_numpy(dtype=float)
            else:
                p5_base = p95_base = p50_base = None

            y_star, p50_adj = self._eval_metalog_spt(
                p_star=p_star,
                p5_cqr=p5_cqr,
                p95_cqr=p95_cqr,
                p5_base=p5_base,
                p95_base=p95_base,
                p50_base=p50_base,
            )

            suffix = f"_{idx}" if len(self.pair_mappings_) > 1 else ""

            if p50_adj is not None:
                df_pred[f"p50_conformal_adj{suffix}"] = p50_adj

            df_pred[f"y_optimal{suffix}"] = np.maximum(0.0, y_star)

        return df_pred

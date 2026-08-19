import numpy as np


class ConformalSeasonalPoolsAdaptive:
    def __init__(
        self, season_period: int, decay_lambda: float = 0.05, alpha: float = 0.1
    ):
        """
        CSP-Adaptive: Gerador probabilístico sem parâmetros treináveis.
        """
        self.season_period = season_period
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.history_ = None
        self.signed_residuals_ = None

    def fit(self, y_history: np.ndarray):
        self.history_ = np.array(y_history)
        m = self.season_period
        # Resíduos sinalizados em relação ao naive sazonal: y_t - y_{t-m}
        if len(self.history_) > m:
            self.signed_residuals_ = self.history_[m:] - self.history_[:-m]
        else:
            self.signed_residuals_ = np.zeros_like(self.history_)

    def predict_samples(
        self, horizon: int, n_samples: int = 1000, initial_weight: float = 0.5
    ):
        T = len(self.history_)
        m = self.season_period
        samples = np.zeros((n_samples, horizon))

        for h in range(1, horizon + 1):
            target_phase = (T + h - 1) % m

            # 1. Pool Sazonal Empírico (mesma fase com decaimento por recência)
            indices = [i for i in range(target_phase, T, m)]
            same_season_vals = self.history_[indices]
            recency = np.arange(len(same_season_vals))
            weights = np.exp(self.decay_lambda * recency)
            weights /= np.sum(weights)

            draws_seasonal = np.random.choice(
                same_season_vals, size=n_samples, p=weights
            )

            # 2. Resíduos Sinalizados em torno do Naive Sazonal
            naive_point = self.history_[T - m + ((h - 1) % m)]
            res_draws = np.random.choice(
                self.signed_residuals_, size=n_samples, replace=True
            )
            draws_conformal = naive_point + res_draws

            # 3. Mistura Adaptativa (ponderação entre sazonalidade pura e resíduos)
            w_h = initial_weight
            mix_mask = np.random.binomial(1, w_h, size=n_samples)
            samples[:, h - 1] = (
                mix_mask * draws_seasonal + (1 - mix_mask) * draws_conformal
            )

        return samples

    def predict_interval(self, horizon: int, n_samples: int = 1000):
        samples = self.predict_samples(horizon, n_samples=n_samples)
        lower = np.percentile(samples, (self.alpha / 2) * 100, axis=0)
        upper = np.percentile(samples, (1 - self.alpha / 2) * 100, axis=0)
        median = np.median(samples, axis=0)
        return median, lower, upper

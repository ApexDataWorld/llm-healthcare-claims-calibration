from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

try:
    from scipy.stats import beta  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    beta = None


@dataclass
class BetaBinomialPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, mass: float = 0.95) -> tuple[float, float]:
        if beta is not None:
            tail = (1.0 - mass) / 2.0
            return float(beta.ppf(tail, self.alpha, self.beta)), float(beta.ppf(1.0 - tail, self.alpha, self.beta))

        variance = (self.alpha * self.beta) / (((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1.0))
        std_dev = variance**0.5
        z_value = NormalDist().inv_cdf(0.5 + mass / 2.0)
        lower = max(0.0, self.mean - z_value * std_dev)
        upper = min(1.0, self.mean + z_value * std_dev)
        return float(lower), float(upper)

    def probability_above(self, threshold: float) -> float:
        if beta is not None:
            return float(1.0 - beta.cdf(threshold, self.alpha, self.beta))

        variance = (self.alpha * self.beta) / (((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1.0))
        std_dev = max(variance**0.5, 1e-8)
        return float(1.0 - NormalDist(mu=self.mean, sigma=std_dev).cdf(threshold))


def fit_beta_binomial(successes: int, trials: int, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> BetaBinomialPosterior:
    return BetaBinomialPosterior(alpha=alpha_prior + successes, beta=beta_prior + trials - successes)

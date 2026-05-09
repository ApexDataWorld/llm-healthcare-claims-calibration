from llm_claims_calibration.bayesian.beta_binomial import fit_beta_binomial


def test_beta_binomial_posterior_properties() -> None:
    posterior = fit_beta_binomial(successes=8, trials=10)
    lower, upper = posterior.credible_interval()
    assert 0.0 < lower < upper < 1.0
    assert posterior.mean > 0.5
    assert posterior.probability_above(0.5) > 0.5

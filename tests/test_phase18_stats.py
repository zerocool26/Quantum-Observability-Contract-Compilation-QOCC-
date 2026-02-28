import pytest
import numpy as np
import sys

has_scipy = False
try:
    import scipy # noqa
    has_scipy = True
except ImportError:
    pass

from qocc.contracts.stats import (
    kolmogorov_smirnov_test,
    jensen_shannon_divergence,
    permutation_test,
    fdr_correction,
    tvd_bootstrap_ci
)
from qocc.contracts.calibration import calibrate_ci_coverage

@pytest.mark.skipif(not has_scipy, reason="scipy not installed")
def test_kolmogorov_smirnov():
    # identical
    c_a = {"00": 100, "11": 100}
    c_b = {"00": 100, "11": 100}
    res = kolmogorov_smirnov_test(c_a, c_b)
    # p-value should be high for identical means we don't reject null -> passed = True
    assert res["passed"] is True
    assert res["statistic"] == 0.0

    # vastly different
    c_c = {"00": 200, "11": 0}
    res2 = kolmogorov_smirnov_test(c_a, c_c)
    assert res2["statistic"] > 0
    assert res2["passed"] is False

@pytest.mark.skipif(not has_scipy, reason="scipy not installed")
def test_jensen_shannon():
    c_a = {"00": 100}
    c_b = {"00": 100}
    assert jensen_shannon_divergence(c_a, c_b) == 0.0

    c_c = {"11": 100}
    assert jensen_shannon_divergence(c_a, c_c) > 0.0
    
def test_permutation_test():
    c_a = {"00": 50, "11": 50}
    c_b = {"00": 52, "11": 48}
    res = permutation_test(c_a, c_b, n_permutations=100)
    assert res["passed"] is True # null not rejected

    c_c = {"00": 100, "11": 0}
    res2 = permutation_test(c_a, c_c, n_permutations=100)
    assert res2["passed"] is False

def test_fdr_correction():
    p_values = [0.001, 0.01, 0.04, 0.1, 0.2]
    # alpha = 0.05
    # k=1: P_(1) = 0.001 <= (1/5)*0.05 = 0.01 -> Reject (False)
    # k=2: P_(2) = 0.01 <= (2/5)*0.05 = 0.02 -> Reject (False)
    # k=3: P_(3) = 0.04 > (3/5)*0.05 = 0.03 -> Retain? Wait, largest k
    results = fdr_correction(p_values, alpha=0.05)
    
    assert results[0] is False # rejected
    assert results[1] is False # rejected
    assert results[2] is True # retained
    assert results[3] is True # retained
    assert results[4] is True # retained

def test_calibration():
    # True underlying TVD = 0 (distributions are exactly equal)
    def generator(seed):
        rng = np.random.default_rng(seed)
        ca = {"0": rng.binomial(100, 0.5), "1": rng.binomial(100, 0.5)}
        cb = {"0": rng.binomial(100, 0.5), "1": rng.binomial(100, 0.5)}
        return ca, cb
        
    def eval_wrapper(ca, cb):
        # use tiny n_bootstrap to make test fast
        return tvd_bootstrap_ci(ca, cb, confidence=0.95, n_bootstrap=20)
        
    report = calibrate_ci_coverage(
        eval_wrapper,
        true_param=0.0,
        distribution_generator=generator,
        n_trials=10, # Keep it small for test
        confidence_level=0.95
    )
    
    assert "actual_coverage" in report
    assert "passed_calibration" in report

import numpy as np
import pytest

from qpong.stage4_settlement.gfn import GFN, alpha_cut, gaussian_coverage, width
from qpong.stage4_settlement.governance import Action, policy
from qpong.stage4_settlement.shapley import coalition_efficiency_error, exact_shapley, permutation_shapley


def random_game(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(10, 5, 1 << n)
    s = rng.uniform(0.1, 2, 1 << n)
    m[0] = 0.0
    s[0] = 0.0

    def v(masks):
        masks = np.asarray(masks)
        return m[masks], s[masks]

    return v, m


@pytest.mark.parametrize("n", [3, 6, 8])
def test_exact_efficiency(n):
    """Proposition 2: sum of mean allocations equals the grand-coalition mean."""
    v, m = random_game(n)
    mphi, sphi, _ = exact_shapley(n, v)
    assert coalition_efficiency_error(mphi, m[-1]) < 1e-9
    assert np.all(sphi > 0)


def test_shared_permutation_efficiency():
    n = 12
    v, m = random_game(n, 1)
    mphi, _, info = permutation_shapley(n, v, M=200, rng=np.random.default_rng(0))
    assert coalition_efficiency_error(mphi, m[-1]) < 1e-8
    assert info["n_coalitions"] <= 200 * n + 1


def test_permutation_converges_to_exact():
    n = 6
    v, _ = random_game(n, 2)
    ex, _, _ = exact_shapley(n, v)
    ap, _, _ = permutation_shapley(n, v, M=20000, rng=np.random.default_rng(0))
    assert np.max(np.abs(ex - ap)) < 0.3


def test_alpha_cut_monotone_and_collapse():
    """Proposition 1 and the collapse at alpha = 1."""
    m, s = 5.0, 2.0
    prev = None
    for a in [0.1, 0.3, 0.5, 0.8, 1.0]:
        lo, hi = alpha_cut(m, s, a)
        if prev is not None:
            assert lo >= prev[0] - 1e-12 and hi <= prev[1] + 1e-12
        prev = (lo, hi)
    assert width(s, 1.0) == pytest.approx(0.0)
    assert GFN(m, s).membership(alpha_cut(m, s, 0.4)[1]) == pytest.approx(0.4)


def test_gfn_addition():
    c = GFN(1, 3) + GFN(2, 4)
    assert c.m == 3 and c.s == pytest.approx(5)


def test_gaussian_coverage_values():
    assert gaussian_coverage(0.2) == pytest.approx(0.927, abs=1e-3)
    assert gaussian_coverage(0.8) == pytest.approx(0.496, abs=1e-3)


def test_policy_branches():
    m = np.array([10.0, 10.0])
    assert policy(False, m, m * 0, m * 0, 0.5) == Action.FALLBACK
    assert policy(True, m, np.array([0.1, 0.1]), np.array([11.0, 0.0]), 0.5) == Action.REJECT
    assert policy(True, m, np.array([2.0, 0.1]), np.array([0.0, 0.0]), 0.5) == Action.ADJUST
    assert policy(True, m, np.array([0.1, 0.1]), np.array([0.0, 0.0]), 0.9) == Action.APPROVE

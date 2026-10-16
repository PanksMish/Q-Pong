import numpy as np
import pytest

from qpong.config import load_config
from qpong.stage2_quantum.ansatz import StatevectorAnsatz, n_params, qiskit_circuit
from qpong.stage2_quantum.classical import exact_enumeration, simulated_annealing
from qpong.stage2_quantum.instances import synthetic_uc
from qpong.stage2_quantum.qubo import build_qubo, ising_energy, qubo_to_ising, to_sparse_pauli_op


@pytest.fixture(scope="module")
def cfg():
    return load_config("ieee30")


def _bits(n):
    idx = np.arange(1 << n)
    return ((idx[:, None] >> np.arange(n)) & 1)


@pytest.mark.parametrize("n", [3, 6, 9])
def test_qubo_ising_equivalence(cfg, n):
    q = build_qubo(synthetic_uc(n), cfg["stage2"]["penalties"])
    E = q.energy_all()
    k, h, J = qubo_to_ising(q)
    Z = 1 - 2 * _bits(n)
    assert np.allclose(ising_energy(k, h, J, Z), E)
    op = to_sparse_pauli_op(k, h, J)
    assert np.allclose(np.real(op.to_matrix(sparse=True).diagonal()), E)


def test_parameter_count():
    for n in (4, 6, 19):
        assert n_params(n, 3) == 8 * n
        qc, p = qiskit_circuit(n, 3, measure=False)
        assert len(p) == 8 * n
        assert qc.count_ops()["cx"] == 3 * n          # circular chain, L = 3


def test_numpy_simulator_matches_qiskit():
    from qiskit.quantum_info import Statevector

    n = 5
    th = np.random.default_rng(3).uniform(0, 2 * np.pi, 8 * n)
    qc, par = qiskit_circuit(n, 3, measure=False)
    ref = Statevector(qc.assign_parameters(dict(zip(par, th)))).data
    assert np.allclose(StatevectorAnsatz(n).statevector(th), ref)


def test_variational_upper_bound(cfg):
    """Theorem 1: <H> >= E0 for any state."""
    n = 6
    q = build_qubo(synthetic_uc(n), cfg["stage2"]["penalties"])
    _, E0, E = exact_enumeration(q)
    A = StatevectorAnsatz(n)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert A.probabilities(rng.uniform(0, 6.3, 8 * n)) @ E >= E0[0] - 1e-12


def test_sa_finds_small_optimum(cfg):
    q = build_qubo(synthetic_uc(8), cfg["stage2"]["penalties"])
    _, E0, _ = exact_enumeration(q)
    _, e = simulated_annealing(q, sweeps=2000, restarts=10, rng=np.random.default_rng(0))
    assert e <= E0[0] + 1e-6 or abs(e - E0[0]) / abs(E0[0]) < 0.05

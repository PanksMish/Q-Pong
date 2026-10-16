"""Sampling back-ends for the variational circuit.

* ``NumpySampler``  - exact statevector + multinomial shot sampling. For a
  diagonal (Pauli-Z) Hamiltonian this is statistically identical to measuring
  the circuit S_shots times on a noiseless simulator.
* ``AerSampler``     - Qiskit Aer (CPU, or GPU/cuQuantum with device='GPU');
  optional noise model (Section V-C secondary mode).
* ``HardwareSampler`` lives in :mod:`qpong.stage2_quantum.hardware`.

All samplers return counts as {state index: count}, where bit k of the index
is the measured value of qubit k (= y_k).
"""
from __future__ import annotations

import numpy as np

from .ansatz import StatevectorAnsatz, qiskit_circuit


class NumpySampler:
    name = "numpy"

    def __init__(self, n: int, reps: int = 3):
        self.n = n
        self.ans = StatevectorAnsatz(n, reps)

    def probabilities(self, theta):
        return self.ans.probabilities(theta)

    def sample(self, theta, shots, rng):
        p = self.probabilities(theta)
        cnt = rng.multinomial(shots, p)
        nz = np.nonzero(cnt)[0]
        return dict(zip(nz.tolist(), cnt[nz].tolist()))

    def expectation(self, theta, shots, rng, energies_all):
        """Shot-based estimate of <H> (exact if shots is None)."""
        p = self.probabilities(theta)
        if shots is None:
            return float(p @ energies_all)
        cnt = rng.multinomial(shots, p)
        return float(cnt @ energies_all / shots)


class AerSampler:
    name = "aer"

    def __init__(self, n: int, reps: int = 3, noise_model=None, device: str = "CPU",
                 method: str = "automatic", seed: int | None = None, optimization_level: int = 1):
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        self.n = n
        opts = {"method": method, "device": device}
        if noise_model is not None:
            opts["noise_model"] = noise_model
        if seed is not None:
            opts["seed_simulator"] = seed
        self.sim = AerSimulator(**opts)
        qc, self.params = qiskit_circuit(n, reps, measure=True)
        basis = noise_model.basis_gates if noise_model is not None else None
        if basis is not None:
            self.tqc = transpile(qc, basis_gates=basis, optimization_level=optimization_level)
        else:
            self.tqc = transpile(qc, self.sim, optimization_level=optimization_level)

    def sample(self, theta, shots, rng=None):
        bound = self.tqc.assign_parameters(dict(zip(self.params, theta)))
        kw = {"seed_simulator": int(rng.integers(2 ** 31))} if rng is not None else {}
        res = self.sim.run(bound, shots=shots, **kw).result()
        counts = res.get_counts()
        return {int(k.replace(" ", ""), 2): v for k, v in counts.items()}

    def expectation(self, theta, shots, rng, qubo):
        counts = self.sample(theta, shots or 8192, rng)
        return energy_from_counts(counts, qubo)


def decode_indices(idx, n):
    idx = np.asarray(list(idx), dtype=np.int64)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


def energy_from_counts(counts: dict, qubo) -> float:
    keys = list(counts.keys())
    Y = decode_indices(keys, qubo.n)
    E = qubo.energy(Y)
    c = np.array([counts[k] for k in keys], float)
    return float(c @ E / c.sum())


def make_sampler(cfg: dict, n: int, seed: int | None = None, noise_model=None):
    s2 = cfg["stage2"]
    if s2["mode"] == "noise" and noise_model is None:
        from .noise import build_noise_model

        noise_model = build_noise_model(cfg)
    if s2["mode"] == "hardware":
        raise ValueError("use qpong.stage2_quantum.hardware for QPU execution")
    if s2["mode"] == "noise" or s2["simulator"] == "aer" or n > s2.get("numpy_max_qubits", 14):
        method = "matrix_product_state" if (n > 28 and s2.get("aer_device", "CPU") == "CPU") else "automatic"
        return AerSampler(n, s2["reps"], noise_model=noise_model, device=s2.get("aer_device", "CPU"),
                          method=method, seed=seed)
    return NumpySampler(n, s2["reps"])

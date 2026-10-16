"""EfficientSU2-type hardware-efficient ansatz (Section III-B).

Initial RY, RZ layer on every qubit, followed by L layers of
  * a circular CNOT chain (i, (i mod N) + 1), i = 1..N  (1-based), i.e.
    CX(0,1), CX(1,2), ..., CX(N-2,N-1), CX(N-1,0) in 0-based indices,
  * another RY, RZ layer.
K = 2 N (L + 1) = 8 N parameters for L = 3.

Two equivalent back-ends:
  * ``StatevectorAnsatz`` - fast NumPy statevector (N <= ~24),
  * ``qiskit_circuit``     - Qiskit circuit for Aer / IBM hardware.
Parameter order: for layer l = 0..L, all RY angles (qubit 0..N-1), then all RZ.
Qubit k carries bit y_k (Qiskit little-endian convention).
"""
from __future__ import annotations

import numpy as np


def n_params(n: int, reps: int = 3) -> int:
    return 2 * n * (reps + 1)


def cx_pairs(n: int):
    if n == 1:
        return []
    return [(i, (i + 1) % n) for i in range(n)]


def qiskit_circuit(n: int, reps: int = 3, measure: bool = True):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    th = ParameterVector("theta", n_params(n, reps))
    qc = QuantumCircuit(n)
    p = 0
    for layer in range(reps + 1):
        if layer > 0:
            for c, t in cx_pairs(n):
                qc.cx(c, t)
        for q in range(n):
            qc.ry(th[p + q], q)
        p += n
        for q in range(n):
            qc.rz(th[p + q], q)
        p += n
    if measure:
        qc.measure_all()
    return qc, th


class StatevectorAnsatz:
    """Exact statevector simulation of the ansatz with NumPy."""

    def __init__(self, n: int, reps: int = 3):
        self.n, self.reps = n, reps
        idx = np.arange(1 << n, dtype=np.int64)
        self._cx_perm = []
        for c, t in cx_pairs(n):
            self._cx_perm.append((idx ^ (((idx >> c) & 1) << t)).astype(np.int64))

    def _apply_1q(self, psi, U, q):
        n = self.n
        s = psi.reshape(1 << (n - q - 1), 2, 1 << q)
        return np.einsum("ab,ibj->iaj", U, s).reshape(-1)

    @staticmethod
    def _ry(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def _rz(t):
        return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)

    def statevector(self, theta: np.ndarray) -> np.ndarray:
        n = self.n
        psi = np.zeros(1 << n, dtype=complex)
        psi[0] = 1.0
        p = 0
        for layer in range(self.reps + 1):
            if layer > 0:
                for perm in self._cx_perm:
                    psi = psi[perm]
            for q in range(n):     # fused RZ(phi) RY(theta) on each qubit
                U = self._rz(theta[p + n + q]) @ self._ry(theta[p + q])
                psi = self._apply_1q(psi, U, q)
            p += 2 * n
        return psi

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        pr = np.abs(self.statevector(theta)) ** 2
        return pr / pr.sum()

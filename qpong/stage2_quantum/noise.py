"""Noise model of the secondary simulation mode (Section V-C).

Depolarizing errors p1 (1q gates) and p2 (CX), composed with thermal
relaxation (T1 = 50 us, T2 = 70 us) over the gate durations.
"""
from __future__ import annotations


def build_noise_model(cfg: dict):
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

    nz = cfg["stage2"]["noise"]
    T1, T2 = nz["T1_us"] * 1e3, nz["T2_us"] * 1e3         # ns
    t1q, t2q, tm = nz["t_1q_ns"], nz["t_2q_ns"], nz["t_meas_ns"]
    nm = NoiseModel(basis_gates=["rz", "sx", "x", "cx", "ry", "measure", "id"])
    e1 = depolarizing_error(nz["p1"], 1).compose(thermal_relaxation_error(T1, T2, t1q))
    e2 = depolarizing_error(nz["p2"], 2).compose(
        thermal_relaxation_error(T1, T2, t2q).expand(thermal_relaxation_error(T1, T2, t2q)))
    nm.add_all_qubit_quantum_error(e1, ["sx", "x", "ry"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    nm.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, tm), ["measure"])
    return nm

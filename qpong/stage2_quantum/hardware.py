"""Execution of trained circuits on IBM Quantum hardware (Section VI-E).

Parameters theta* are trained in noiseless simulation and transferred
unchanged; only the sampling step runs on the QPU.

* raw        : SamplerV2 / EstimatorV2 without any mitigation.
* mitigated  : SamplerV2 with XY4 dynamical decoupling and gate + measurement
               twirling (these counts give S_feas/S_shots and the candidate set);
               EstimatorV2 with TREX readout mitigation (measure_mitigation)
               for the energy only.

Everything needed to audit a run is written to ``hardware/<run_id>/``:
job identifiers, raw counts, transpiled 2Q-gate counts / depth, and a
calibration snapshot of the backend.

Credentials: set QISKIT_IBM_TOKEN (and optionally QISKIT_IBM_INSTANCE), or
save them once with QiskitRuntimeService.save_account(...). Use
``--dry-run`` (FakeKingston, local simulation with a Heron r2 noise snapshot)
to test the workflow without QPU time; dry-run results are NOT hardware results.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from .ansatz import qiskit_circuit


def get_backend(name: str = "ibm_kingston", dry_run: bool = False):
    if dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeKingston

        return FakeKingston(), None
    from qiskit_ibm_runtime import QiskitRuntimeService

    kw = {}
    if os.environ.get("QISKIT_IBM_TOKEN"):
        kw = {"channel": "ibm_quantum_platform", "token": os.environ["QISKIT_IBM_TOKEN"]}
        if os.environ.get("QISKIT_IBM_INSTANCE"):
            kw["instance"] = os.environ["QISKIT_IBM_INSTANCE"]
    service = QiskitRuntimeService(**kw)
    return service.backend(name), service


def calibration_snapshot(backend) -> dict:
    """Median/best 2Q error and median readout error from the backend target."""
    tgt = backend.target
    twoq = [g for g in ("cz", "ecr", "cx") if g in tgt.operation_names]
    e2 = []
    for g in twoq:
        for qargs, props in tgt[g].items():
            if props is not None and props.error is not None:
                e2.append(props.error)
    ro = [p.error for q, p in tgt["measure"].items() if p is not None and p.error is not None]
    snap = {
        "backend": backend.name,
        "num_qubits": backend.num_qubits,
        "twoq_gate": twoq,
        "median_2q_error": float(np.median(e2)) if e2 else None,
        "best_2q_error": float(np.min(e2)) if e2 else None,
        "median_readout_error": float(np.median(ro)) if ro else None,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:  # layered 2Q error (EPLG-based) is reported by some backends
        props = backend.properties()
        if props is not None:
            snap["properties_last_update"] = str(props.last_update_date)
    except Exception:
        pass
    return snap


def transpile_for(backend, n: int, reps: int, optimization_level: int = 3, seed: int = 0):
    from qiskit.transpiler import generate_preset_pass_manager

    qc, params = qiskit_circuit(n, reps, measure=True)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed)
    tqc = pm.run(qc)
    ops = tqc.count_ops()
    n2q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
    qc_nm, _ = qiskit_circuit(n, reps, measure=False)
    tqc_nm = pm.run(qc_nm)
    return tqc, tqc_nm, params, {"twoq_gates": int(n2q), "depth": int(tqc.depth()),
                                  "logical_cx": len(qc.get_instructions("cx"))}


def _sampler(mode, mitigated: bool, shots: int):
    from qiskit_ibm_runtime import SamplerV2

    s = SamplerV2(mode=mode)
    s.options.default_shots = shots
    if mitigated:
        s.options.dynamical_decoupling.enable = True
        s.options.dynamical_decoupling.sequence_type = "XY4"
        s.options.twirling.enable_gates = True
        s.options.twirling.enable_measure = True
    else:
        s.options.dynamical_decoupling.enable = False
        s.options.twirling.enable_gates = False
        s.options.twirling.enable_measure = False
    return s


def _estimator(mode, mitigated: bool, shots: int):
    from qiskit_ibm_runtime import EstimatorV2

    e = EstimatorV2(mode=mode)
    e.options.default_shots = shots
    if mitigated:
        e.options.resilience_level = 1
        e.options.resilience.measure_mitigation = True       # TREX
        e.options.resilience.zne_mitigation = False
    else:
        e.options.resilience_level = 0
    return e


def run_on_hardware(jobs: list[dict], backend, out_dir: Path, shots: int = 8192, reps: int = 3,
                    optimization_level: int = 3, service=None) -> list[dict]:
    """Execute a list of trained circuits.

    Each job dict: {"key": str, "n": int, "theta": array, "hamiltonian": SparsePauliOp}.
    Returns per-job dicts with raw / mitigated counts and energies.
    """
    from qiskit_ibm_runtime import Batch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = calibration_snapshot(backend)
    (out_dir / "calibration_snapshot.json").write_text(json.dumps(snap, indent=2))
    import qiskit
    import qiskit_ibm_runtime

    versions = {"qiskit": qiskit.__version__, "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__}

    cache = {}
    pubs_s, pubs_e, meta = [], [], []
    for j in jobs:
        n = j["n"]
        if n not in cache:
            cache[n] = transpile_for(backend, n, reps, optimization_level)
        tqc, tqc_nm, params, info = cache[n]
        theta = np.asarray(j["theta"], float)
        pubs_s.append((tqc, theta))
        obs = j["hamiltonian"].apply_layout(tqc_nm.layout)
        pubs_e.append((tqc_nm, obs, theta))
        meta.append({"key": j["key"], "n": n, **info})

    results = []
    mode = Batch(backend=backend) if service is not None else backend
    try:
        runs = {}
        for tag, mit in (("raw", False), ("mitigated", True)):
            js = _sampler(mode, mit, shots).run(pubs_s)
            je = _estimator(mode, mit, shots).run(pubs_e)
            runs[tag] = (js, je)
        for tag, (js, je) in runs.items():
            rs, re_ = js.result(), je.result()
            for i, m in enumerate(meta):
                if len(results) <= i:
                    results.append(dict(m))
                counts = rs[i].data.meas.get_counts()
                results[i][f"counts_{tag}"] = {int(k, 2): v for k, v in counts.items()}
                results[i][f"energy_{tag}"] = float(re_[i].data.evs)
                results[i][f"job_sampler_{tag}"] = js.job_id()
                results[i][f"job_estimator_{tag}"] = je.job_id()
    finally:
        if service is not None:
            mode.close()

    record = {"versions": versions, "calibration": snap, "shots": shots, "jobs": [
        {k: v for k, v in r.items() if not k.startswith("counts_")} for r in results]}
    (out_dir / "jobs.json").write_text(json.dumps(record, indent=2, default=str))
    (out_dir / "raw_counts.json").write_text(json.dumps(
        {r["key"]: {"raw": r["counts_raw"], "mitigated": r["counts_mitigated"]} for r in results}))
    return results

import numpy as np
import pytest

from qpong.config import load_config
from qpong.data import day_splits
from qpong.grid import GridModel, ac_mismatch, evaluate_dispatch, solve_acopf
from qpong.stage1_fbwm import FBWM, context_vector


@pytest.fixture(scope="module")
def grid30():
    return GridModel.from_config(load_config("ieee30"))


def test_case30_units(grid30):
    assert grid30.n_enc == 6
    assert sorted(grid30.units.bus1[grid30.encoded]) == [1, 2, 13, 22, 23, 27]


def test_case118_encoding():
    g = GridModel.from_config(load_config("ieee118"))
    assert g.n_units == 54 and g.n_enc == 19 and len(g.fixed) == 35


def test_opf_reproduces_matpower_case30():
    cfg = load_config("ieee30", overrides=["grid.pmin_fraction=0", "grid.noload_cost_fraction=0"])
    g = GridModel.from_config(cfg)
    r = solve_acopf(g, g.base_p, g.base_q, np.zeros(g.n_bus), np.ones(g.n_units))
    assert r.success and r.cost == pytest.approx(576.89, abs=0.1)     # MATPOWER case30 optimum


def test_gate_checker_accepts_opf_solution(grid30):
    z = np.zeros(grid30.n_bus)
    u = np.ones(grid30.n_units)
    r = solve_acopf(grid30, grid30.base_p, grid30.base_q, z, u)
    d = evaluate_dispatch(grid30, grid30.base_p, grid30.base_q, z, u, r.pg, r.vg)
    assert d.converged and d.feasible and d.n_viol == 0
    assert ac_mismatch(grid30, d.vm, d.va, d.pg, d.qg, grid30.base_p, grid30.base_q, z) < 1e-6


def test_gate_checker_flags_voltage_violation(grid30):
    z = np.zeros(grid30.n_bus)
    u = np.ones(grid30.n_units)
    r = solve_acopf(grid30, grid30.base_p, grid30.base_q, z, u)
    d = evaluate_dispatch(grid30, grid30.base_p, grid30.base_q, z, u, r.pg, np.full(grid30.n_units, 1.2))
    assert not d.feasible and d.n_v > 0


def test_fbwm_simplex_and_adaptation():
    cfg = load_config("ieee30")
    f = FBWM(cfg)
    w0 = f(np.zeros(4))
    assert w0.sum() == pytest.approx(1) and np.all(w0 >= 0)
    w_stress = f(np.array([0.5, 1.0, 0.0, 0.5]))        # high network stress -> more security weight
    assert w_stress[2] > w0[2]
    assert np.allclose(FBWM(cfg, adaptive=False)(np.array([0.5, 1.0, 0.0, 0.5])), w0, atol=1e-6)


def test_context_projection():
    cfg = load_config("ieee30")
    s = context_vector(cfg, np.ones(3) * 10, np.ones(3), 1000.0, 20.0, 61.0)
    assert np.all(s <= 1) and np.all(s >= -1)


def test_day_splits():
    sp = day_splits()
    assert len(sp["test"]) == 30 and len(sp["val"]) == 30 and len(sp["train"]) == 305
    assert sp["test"][0] == 11 and sp["test"][-1] == 359
    assert not set(sp["test"]) & set(sp["train"])

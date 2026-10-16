"""Network model built on the MATPOWER case files (via PYPOWER, the Python port
of MATPOWER). case30 / case57 / case118 are the MATPOWER files, not the
PGLib-OPF variants (Section V-A).

Units are the rows of the MATPOWER gen table, in table order.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from pypower import api as ppapi
from pypower.idx_brch import F_BUS, RATE_A, T_BUS
from pypower.idx_bus import BUS_TYPE, PD, QD, REF, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.makeYbus import makeYbus


@dataclass
class GridModel:
    cfg: dict
    ppc: dict = field(repr=False)                    # MATPOWER case (external = internal numbering)
    units: pd.DataFrame = field(repr=False)
    encoded: np.ndarray = field(repr=False)
    fixed: np.ndarray = field(repr=False)
    base_p: np.ndarray = field(repr=False)           # MW, bus order
    base_q: np.ndarray = field(repr=False)
    G: np.ndarray = field(repr=False)
    B: np.ndarray = field(repr=False)
    Yf: object = field(repr=False)
    Yt: object = field(repr=False)
    f_bus: np.ndarray = field(repr=False)
    t_bus: np.ndarray = field(repr=False)
    s_max: np.ndarray = field(repr=False)            # pu MVA, inf = unlimited
    vmin: np.ndarray = field(repr=False)
    vmax: np.ndarray = field(repr=False)
    base_mva: float = 100.0
    name: str = ""
    slack_bus: int = 0
    n_unrated: int = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: dict) -> "GridModel":
        sysc, g = cfg["system"], cfg["grid"]
        ppc = copy.deepcopy(getattr(ppapi, sysc["case"])())
        bus, gen, br, gc = ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
        nb = bus.shape[0]
        assert np.all(bus[:, 0] == np.arange(1, nb + 1)), "case must use consecutive bus numbers"

        # polynomial costs (MATPOWER model 2): last three coefficients c2, c1, c0
        coeffs = np.zeros((gen.shape[0], 3))
        for i, row in enumerate(gc):
            n = int(row[3])
            c = row[4:4 + n][::-1]                      # c0, c1, c2, ...
            coeffs[i, : min(3, n)] = c[:3]
        units = pd.DataFrame({
            "bus": gen[:, GEN_BUS].astype(int) - 1,
            "pmax": gen[:, PMAX], "qmin": gen[:, QMIN], "qmax": gen[:, QMAX],
            "a": coeffs[:, 2], "b": coeffs[:, 1], "c0": coeffs[:, 0],
        })
        units["bus1"] = units["bus"] + 1
        if sysc.get("encoded_buses") == "auto":
            anchors = set(sysc["zone_anchor_buses"])
            enc = units.index[units.bus1.isin(anchors)].to_numpy()
            assert len(enc) == len(anchors), "every zone anchor bus must hold one unit"
        else:
            enc = np.array([units.index[units.bus1 == b][0] for b in sysc["encoded_buses"]])
        enc = np.asarray(enc, int)
        fixed = np.array([i for i in units.index if i not in set(enc)], dtype=int)
        units["encoded"] = units.index.isin(enc)
        units["pmin"] = np.where(units.encoded, g["pmin_fraction"] * units.pmax, gen[:, PMIN])
        if sysc.get("fixed_units_mode", "dispatchable") == "condenser" and len(fixed):
            # units outside the encoding are synchronous condensers: P = 0, voltage set-point held
            units.loc[fixed, ["pmax", "pmin"]] = 0.0
            ppc["gen"][fixed, PMAX] = 0.0
            ppc["gen"][fixed, PMIN] = 0.0
            from pypower.idx_gen import PG

            ppc["gen"][fixed, PG] = 0.0
        units["ramp"] = g["ramp_fraction_per_interval"] * units.pmax
        units["c"] = units["c0"] + g["noload_cost_fraction"] * units["b"] * units["pmax"]
        units["startup"] = g["startup_cost_hours"] * units["b"] * units["pmax"]

        from pypower.ext2int import ext2int

        pint = ext2int(copy.deepcopy(ppc))
        Ybus, Yf, Yt = makeYbus(pint["baseMVA"], pint["bus"], pint["branch"])
        assert pint["branch"].shape[0] == br.shape[0], "out-of-service branches not supported"
        base_mva = float(ppc["baseMVA"])
        f_bus = br[:, F_BUS].astype(int) - 1
        t_bus = br[:, T_BUS].astype(int) - 1

        # thermal ratings
        s_max = br[:, RATE_A] / base_mva
        unrated = (s_max <= 0) | (br[:, RATE_A] >= 9000)   # MATPOWER uses 0 or 9900 for "no limit"
        mode = g.get("thermal_ratings", "case_or_derived")
        if mode == "case":
            s_max = np.where(unrated, np.inf, s_max)
        elif unrated.any():  # MODELING CHOICE: derive ratings from the base-case OPF flows
            opt = ppapi.ppoption(VERBOSE=0, OUT_ALL=0)
            base = ppapi.runopf(copy.deepcopy(ppc), opt)
            V = base["bus"][:, 7] * np.exp(1j * np.deg2rad(base["bus"][:, 8]))
            sf = np.abs(V[f_bus] * np.conj(Yf @ V))
            st = np.abs(V[t_bus] * np.conj(Yt @ V))
            flow = np.maximum(sf, st)
            factor = g.get("derived_rating_factor", 1.5)
            floor = g.get("derived_rating_floor_mva", 20.0) / base_mva
            s_max = np.where(unrated, np.maximum(factor * flow, floor), s_max)
        ppc["branch"][:, RATE_A] = np.where(np.isfinite(s_max), s_max * base_mva, 0.0)

        units["zone"] = cls._assign_zones(nb, f_bus, t_bus, units, enc)
        model = cls(cfg=cfg, ppc=ppc, units=units, encoded=enc, fixed=fixed,
                    base_p=bus[:, PD].copy(), base_q=bus[:, QD].copy(),
                    G=Ybus.toarray().real, B=Ybus.toarray().imag, Yf=Yf, Yt=Yt,
                    f_bus=f_bus, t_bus=t_bus, s_max=s_max,
                    vmin=bus[:, VMIN].copy(), vmax=bus[:, VMAX].copy(), base_mva=base_mva,
                    name=sysc["name"], slack_bus=int(np.nonzero(bus[:, BUS_TYPE] == REF)[0][0]),
                    n_unrated=int(unrated.sum()))
        return model

    @staticmethod
    def _assign_zones(nb, f_bus, t_bus, units, enc):
        Gx = nx.Graph()
        Gx.add_nodes_from(range(nb))
        Gx.add_edges_from(zip(f_bus, t_bus))
        anchor_bus = units.loc[enc, "bus"].to_numpy()
        zone = np.zeros(len(units), dtype=int)
        for i, b in enumerate(units.bus):
            hit = np.nonzero(enc == i)[0]
            if len(hit):
                zone[i] = int(hit[0])
                continue
            d = nx.single_source_shortest_path_length(Gx, int(b))
            zone[i] = int(np.argmin([d.get(int(a), 10 ** 9) for a in anchor_bus]))
        return zone

    # ------------------------------------------------------------------ PTDF
    def ptdf(self) -> np.ndarray:
        """DC power-transfer distribution factors (branches x buses), slack column zero."""
        if getattr(self, "_ptdf", None) is None:
            from pypower.ext2int import ext2int
            from pypower.makeBdc import makeBdc

            pint = ext2int(copy.deepcopy(self.ppc))
            Bbus, Bf, _, _ = makeBdc(pint["baseMVA"], pint["bus"], pint["branch"])
            Bbus, Bf = Bbus.toarray(), Bf.toarray()
            keep = np.array([b for b in range(self.n_bus) if b != self.slack_bus])
            H = np.zeros((Bf.shape[0], self.n_bus))
            H[:, keep] = Bf[:, keep] @ np.linalg.inv(Bbus[np.ix_(keep, keep)])
            self._ptdf = H
        return self._ptdf

    def security_sensitivity(self, loading_prev=None) -> np.ndarray:
        """rho_i for R_t(u, P) = sum_i rho_i P_i: |PTDF| of the unit bus weighted by the
        squared loading of each branch in the previous interval (uniform if unknown)."""
        H = np.abs(self.ptdf())
        if loading_prev is None or len(loading_prev) != H.shape[0]:
            w = np.ones(H.shape[0])
        else:
            w = np.nan_to_num(np.asarray(loading_prev, float)) ** 2 + 1e-6
        rho_bus = (w @ H) / w.sum()
        return rho_bus[self.units.bus.to_numpy()]

    # --------------------------------------------------------------- helpers
    @property
    def n_bus(self) -> int:
        return self.ppc["bus"].shape[0]

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_enc(self) -> int:
        return len(self.encoded)

    def unit_bus_matrix(self) -> np.ndarray:
        C = np.zeros((self.n_bus, self.n_units))
        C[self.units.bus.to_numpy(), np.arange(self.n_units)] = 1.0
        return C

    def full_commitment(self, u_enc) -> np.ndarray:
        u = np.ones(self.n_units)
        u[self.encoded] = np.asarray(u_enc, float)
        return u

    def gen_cost(self, pg_mw, u, dt_h: float = 0.25) -> float:
        U = self.units
        return float(np.sum(u * (U.a.to_numpy() * pg_mw ** 2 + U.b.to_numpy() * pg_mw + U.c.to_numpy())) * dt_h)

    def startup_cost(self, u, u_prev) -> float:
        if u_prev is None:
            return 0.0
        return float(np.sum(self.units.startup.to_numpy() * u * (1 - u_prev)))

    # -------------------------------------------------------- configured case
    def configured_case(self, pd_mw, qd_mvar, ppv_mw, u, pg_mw=None, vg_pu=None) -> dict:
        """MATPOWER case with realized loads, PV (as negative load), commitment and set-points.

        A decommitted unit at the reference bus is kept as a synchronous
        condenser (P limits 0) so that the reference bus retains a voltage source.
        """
        from pypower.idx_gen import GEN_STATUS, PG, VG

        ppc = copy.deepcopy(self.ppc)
        ppc["bus"][:, PD] = pd_mw - ppv_mw
        ppc["bus"][:, QD] = qd_mvar
        U = self.units
        for k in range(self.n_units):
            on = u[k] > 0.5
            at_ref = U.bus.iloc[k] == self.slack_bus
            if on:
                ppc["gen"][k, GEN_STATUS] = 1
                ppc["gen"][k, PMIN] = U.pmin.iloc[k]
                ppc["gen"][k, PMAX] = U.pmax.iloc[k]
            elif at_ref:
                ppc["gen"][k, GEN_STATUS] = 1
                ppc["gen"][k, PMIN] = 0.0
                ppc["gen"][k, PMAX] = 0.0
            else:
                ppc["gen"][k, GEN_STATUS] = 0
            if pg_mw is not None:
                ppc["gen"][k, PG] = pg_mw[k] if on else 0.0
            if vg_pu is not None and np.isfinite(vg_pu[k]):
                ppc["gen"][k, VG] = vg_pu[k]
        # a PV bus whose units are all off becomes a PQ bus
        from pypower.idx_bus import PQ, PV

        on_bus = set(U.bus[(u > 0.5) | (U.bus == self.slack_bus)].tolist())
        for b in range(self.n_bus):
            if ppc["bus"][b, BUS_TYPE] == PV and b not in on_bus:
                ppc["bus"][b, BUS_TYPE] = PQ
            if ppc["bus"][b, BUS_TYPE] == PQ and b in on_bus and b != self.slack_bus:
                ppc["bus"][b, BUS_TYPE] = PV
        return ppc

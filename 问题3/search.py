"""Reproducible candidate search for the two Q3 heater strategies.

The preheat search uses the exact seven-node no-load thermal network and
validates every candidate with the full finite-volume stack.  The cooperative
search uses bounded, recorded simulation sweeps followed by finer validation.
Neither stochastic nor grid search is presented as a global proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from aux_model import AuxStack, Q2_LINEAR_PARAMETERS
from stack_model import AREA_CM2, ENDPLATE_CAP_J_K, ENDPLATE_CONV_W_K, T_FREEZE

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def symmetric_power(x):
    return np.array([x[0], x[1], x[2], x[1], x[0]], float)


class NoLoadThermal:
    """Seven-node closed-form thermal model for zero-current preheat search."""

    def __init__(self):
        self.stack = AuxStack(symmetric=False, mesh_factor=1)
        c = self.stack.cell_capacity_j_k
        self.caps = np.array([ENDPLATE_CAP_J_K] + [c] * 5 + [ENDPLATE_CAP_J_K])
        self.k = np.zeros((7, 7))
        for edge in range(6):
            g = self.stack.plate_g_w_k if edge in (0, 5) else self.stack.link_g_w_k
            self.k[edge, edge] += g
            self.k[edge + 1, edge + 1] += g
            self.k[edge, edge + 1] -= g
            self.k[edge + 1, edge] -= g
        for edge in (0, 6):
            self.k[edge, edge] += ENDPLATE_CONV_W_K
        self.a = -self.k / self.caps[:, None]
        self.zeros = np.zeros(self.stack.cell.n)
        self.lam = self.stack.params.membrane_lambda

    def evaluate(self, x, loaded_current=Q2_LINEAR_PARAMETERS[0]):
        q = symmetric_power(x)
        heat_s = float(x[3])
        p = np.r_[0.0, AREA_CM2 * q, 0.0]
        steady = np.linalg.solve(self.k, p)
        temp_c = -30.0 + (np.eye(7) - expm(self.a * heat_s)) @ steady
        vv = []
        for k in range(5):
            state = np.full(self.stack.cell.n, temp_c[k + 1] + T_FREEZE)
            vv.append(self.stack._voltage(
                state, self.zeros, self.zeros, self.lam,
                1e4 * loaded_current, self.stack.factors[k])[0])
        return temp_c[1:6], np.array(vv)


def search_preheat():
    thermal = NoLoadThermal()
    starts = [[1, 1, 1, 50], [1, .8, .8, 65],
              [.8, .8, .8, 80], [1, .6, .4, 100]]
    evaluated = []
    best = None

    def cost(x):
        return float(AREA_CM2 * x[3] * (2*x[0] + 2*x[1] + x[2]))

    def constraints(x):
        t, v = thermal.evaluate(x)
        return np.r_[t - 0.01, v - 0.3005]

    for x0 in starts:
        opt = minimize(cost, x0, method="SLSQP", bounds=[(0, 1)] * 3 + [(20, 300)],
                       constraints=[{"type": "ineq", "fun": constraints}],
                       options={"maxiter": 180, "ftol": 1e-7})
        x = opt.x
        t, v = thermal.evaluate(x)
        item = {"start": x0, "parameters": [float(z) for z in x],
                "energy_j": cost(x), "min_temperature_c": float(min(t)),
                "postload_policy": "Q2 optimized linear 0.17 to 0.50 A/cm2 in 5 s",
                "loaded_current_at_switch_a_cm2": Q2_LINEAR_PARAMETERS[0],
                "min_loaded_voltage_v": float(min(v)),
                "feasible_surrogate": bool(np.min(constraints(x)) >= -1e-5),
                "solver_success": bool(opt.success), "message": str(opt.message),
                "evaluations": int(opt.nfev)}
        evaluated.append(item)
        if (item["feasible_surrogate"] and item["solver_success"]
                and (best is None or item["energy_j"] < best["energy_j"])):
            best = item
    OUT.mkdir(exist_ok=True)
    payload = {
        "interpretation": (
            "Pure preheating uses zero current until all five cells exceed 0 C, "
            "then starts Q2's optimized linear loading policy."
        ),
        "postload_policy": {
            "family": "linear",
            "parameters": list(Q2_LINEAR_PARAMETERS),
            "source": "Problem 2 fastest feasible candidate"
        },
        "runs": evaluated,
        "best": best,
    }
    (OUT / "preheat_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


CO_CANDIDATES = [
    ([0]*5, 0),
    ([.5, .32, .25, .32, .5], 3.5),
    ([.5, .32, .25, .32, .5], 4.0),
    ([.5, .32, .25, .32, .5], 4.05),
    ([.5, .32, .25, .32, .5], 4.10),
    ([.5, .32, .25, .32, .5], 4.20),
    ([.4]*5, 15), ([.5]*5, 20), ([.8]*5, 60), ([1]*5, 40),
    ([.5, .32, .25, .32, .5], 15),
    ([.45, .35, .3, .35, .45], 15),
    ([.55, .4, .3, .4, .55], 12),
    ([.55, .35, .25, .35, .55], 14),
    ([.45, .3, .25, .3, .45], 16),
    ([.6, .35, .25, .35, .6], 12),
    ([.5, .35, .25, .35, .5], 14),
    ([.7, .5, .35, .5, .7], 15),
]


def search_cooperative(dt=0.1, time_cap=310):
    stack = AuxStack(symmetric=True, mesh_factor=1)
    rows = []
    for q, heat_s in CO_CANDIDATES:
        r = stack.simulate("cooperative", q, heat_s, dt_max=dt,
                           time_cap=time_cap, capture_interval=1000)
        row = {"q": q, "heat_s": heat_s, "success": r.success,
               "reason": r.reason, "startup_s": r.startup_s,
               "aux_energy_j": r.heat_aux_j,
               "min_voltage_v": r.min_voltage_v,
               "max_ice_fraction": r.max_ice_fraction,
               "terminal_min_temp_c": float(min(r.final_temperature_c))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "cooperative_search.json").write_text(
        json.dumps({"dt_max_s": dt, "time_cap_s": time_cap, "rows": rows}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preheat", "cooperative"])
    args = parser.parse_args()
    if args.phase == "preheat":
        search_preheat()
    else:
        search_cooperative()

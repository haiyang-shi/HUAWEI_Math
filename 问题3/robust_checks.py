"""Independent Q2 regression and finer checks for Q3 comparison policies."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import aux_model
from aux_model import AuxStack
from stack_model import StackModel, calibrated_parameters, policy_current

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
Q = [.5, .32, .25, .32, .5]


def brief(r):
    return {"success": r.success, "reason": r.reason, "startup_s": r.startup_s,
            "energy_j": r.heat_aux_j, "min_voltage_v": r.min_voltage_v,
            "max_ice_fraction": r.max_ice_fraction,
            "max_pore_occupancy": r.max_pore_occupancy,
            "energy_balance_j": r.energy_balance_j}


def main():
    OUT.mkdir(exist_ok=True)
    cases = {}
    def add(name, model, mode, q, heat_s, dt, cap, **kwargs):
        r = model.simulate(mode, q, heat_s, dt_max=dt, time_cap=cap,
                           capture_interval=1000, **kwargs)
        cases[name] = brief(r)
        print(name, json.dumps(cases[name]), flush=True)
        (OUT / "robust_checks.json").write_text(
            json.dumps({"status": "running", "cases": cases}, indent=2),
            encoding="utf-8")

    add("conservative_84_dt0.05", AuxStack(symmetric=True, mesh_factor=3),
        "cooperative", Q, 15, .05, 310)
    add("conservative_freeze_x5_84_dt0.2",
        AuxStack(symmetric=True, mesh_factor=3,
                 params=replace(calibrated_parameters(), freeze_rate_s=1.0)),
        "cooperative", Q, 15, .2, 310)
    add("conservative_resistance_x0.5_84_dt0.2",
        AuxStack(symmetric=True, mesh_factor=3, link_resistance_scale=.5),
        "cooperative", Q, 15, .2, 310)
    add("preheat_84_dt0.05", AuxStack(symmetric=True, mesh_factor=3),
        "preheat", [1] * 5, 195, .05, 196, postload="step")

    # With the heater off, this must reduce to Q2's current and thermal model.
    original = aux_model.prescribed_current
    aux_model.prescribed_current = lambda t, mode, heat_s, postload="step": policy_current(
        "linear", [.17, .5, 5], t)
    try:
        q3_reduction = AuxStack(symmetric=True, mesh_factor=1).simulate(
            "cooperative", [0] * 5, 0, initial_c=-10, ambient_c=-10,
            dt_max=.05, time_cap=100, charge_limit=20, capture_interval=1000)
    finally:
        aux_model.prescribed_current = original
    q2_reference = StackModel(mesh_factor=1).simulate(
        "linear", [.17, .5, 5], -10, dt_max=.05, time_cap=100,
        capture_interval=1000)
    cases["q2_zero_heater_regression"] = {
        "q2_startup_s": q2_reference.startup_s,
        "q3_zero_heater_startup_s": q3_reduction.startup_s,
        "q2_min_voltage_v": q2_reference.min_voltage_v,
        "q3_zero_heater_min_voltage_v": q3_reduction.min_voltage_v,
        "startup_difference_s": (None if not q2_reference.success or not q3_reduction.success
                                 else q3_reduction.startup_s - q2_reference.startup_s)}
    print("q2_zero_heater_regression", json.dumps(cases["q2_zero_heater_regression"]), flush=True)
    payload = {"status": "completed", "cases": cases,
               "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "run_command": f"{sys.executable} 问题3/robust_checks.py"}
    (OUT / "robust_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

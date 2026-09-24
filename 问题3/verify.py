"""Check numerical resolution and uncertain Q1/Q2 closures for Q3 candidates."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from aux_model import AuxStack
from stack_model import calibrated_parameters

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
Q_LOW = [.5, .32, .25, .32, .5]
Q_ROBUST = [.5, .32, .25, .32, .5]


def record(name, r, mesh_factor, dt_max):
    return {"case": name, "mesh_cells_per_cell": 28 * mesh_factor,
            "dt_max_s": dt_max, "success": r.success, "reason": r.reason,
            "startup_s": r.startup_s, "heat_s": r.actual_heat_s,
            "energy_j": r.heat_aux_j, "min_voltage_v": r.min_voltage_v,
            "max_ice_fraction": r.max_ice_fraction,
            "max_pore_occupancy": r.max_pore_occupancy,
            "min_final_temp_c": float(min(r.final_temperature_c)),
            "water_balance_kg_m2": r.max_water_balance_kg_m2,
            "energy_balance_j": r.energy_balance_j}


def main():
    params = calibrated_parameters()
    cases = [
        ("low_energy_84_dt0.05", 3, .05, Q_LOW, 4, params, 1.0),
        ("low_energy_112_dt0.1", 4, .1, Q_LOW, 4, params, 1.0),
        ("low_energy_56_dt0.1", 2, .1, Q_LOW, 4, params, 1.0),
        ("conservative_28_dt0.1", 1, .1, Q_ROBUST, 15, params, 1.0),
        ("low_energy_freeze_x0.2", 3, .2, Q_LOW, 4,
         replace(params, freeze_rate_s=.04), 1.0),
        ("low_energy_freeze_x5", 3, .2, Q_LOW, 4,
         replace(params, freeze_rate_s=1.0), 1.0),
        ("low_energy_resistance_x0.5", 3, .2, Q_LOW, 4, params, .5),
        ("low_energy_resistance_x2", 3, .2, Q_LOW, 4, params, 2.0),
    ]
    rows = []
    OUT.mkdir(exist_ok=True)
    for name, factor, dt, q, heat_s, p, link in cases:
        r = AuxStack(symmetric=True, mesh_factor=factor, params=p,
                     link_resistance_scale=link).simulate(
                         "cooperative", q, heat_s, dt_max=dt,
                         time_cap=310, capture_interval=1000)
        item = record(name, r, factor, dt)
        rows.append(item)
        (OUT / "verification.json").write_text(
            json.dumps({"status": "running", "rows": rows}, indent=2),
            encoding="utf-8")
        print(json.dumps(item), flush=True)
    payload = {"status": "completed", "rows": rows,
               "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "run_command": f"{sys.executable} 问题3/verify.py"}
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

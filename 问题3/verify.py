"""Check the selected Q3 policies and the cooperative feasibility boundary."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

from aux_model import AuxStack
from stack_model import calibrated_parameters

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
Q_SELECTED = [.5, .32, .25, .32, .5]


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


def run_case(spec):
    name, factor, dt, q, heat_s, params, link = spec
    r = AuxStack(symmetric=True, mesh_factor=factor, params=params,
                 link_resistance_scale=link).simulate(
                     "cooperative", q, heat_s, dt_max=dt,
                     time_cap=310, capture_interval=1000)
    return record(name, r, factor, dt)


def main():
    params = calibrated_parameters()
    cases = [
        ("selected_84_dt0.025", 3, .025, Q_SELECTED, 4.10, params, 1.0),
        ("selected_112_dt0.05", 4, .05, Q_SELECTED, 4.10, params, 1.0),
        ("lower_boundary_84_dt0.025", 3, .025, Q_SELECTED, 4.05, params, 1.0),
    ]
    rows = []
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, spec): spec[0] for spec in cases}
        for future in as_completed(futures):
            item = future.result()
            rows.append(item)
            rows.sort(key=lambda row: [case[0] for case in cases].index(row["case"]))
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

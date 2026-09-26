"""Independent full-five-cell checks of the selected Problem 3 policies."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from aux_model import AuxStack

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
TEMP_MARGIN_C = 0.01
VOLTAGE_MARGIN_V = 0.3005
ICE_LIMIT = 0.99


def load_policies():
    pre_data = json.loads((OUT / "preheat_search.json").read_text(encoding="utf-8"))
    co_data = json.loads((OUT / "cooperative_search.json").read_text(encoding="utf-8"))
    pre_best = pre_data["best"]
    co_best = co_data["best"]
    lower = co_data.get("lower_boundary")
    if lower is None:
        raise RuntimeError("Cooperative search did not record a lower-energy infeasible neighbor")
    return (
        (pre_best["powers_w_cm2"], float(pre_best["heat_s"])),
        (co_best["powers_w_cm2"], float(co_best["heat_s"])),
        (lower["powers_w_cm2"], float(lower["heat_s"])),
    )


def record(name, result, mode, mesh_factor, dt_max):
    return {
        "case": name,
        "mode": mode,
        "powers_w_cm2": list(result.powers_w_cm2),
        "mesh_cells_per_cell": 28 * mesh_factor,
        "dt_max_s": dt_max,
        "success": result.success,
        "reason": result.reason,
        "startup_s": result.startup_s,
        "heat_s": result.actual_heat_s,
        "energy_j": result.heat_aux_j,
        "min_voltage_v": result.min_voltage_v,
        "min_voltage_by_cell_v": result.min_voltage_by_cell_v.tolist(),
        "max_ice_fraction": result.max_ice_fraction,
        "peak_ice_fraction_by_cell": result.peak_ice_fraction_by_cell.tolist(),
        "max_end_cell_ice_fraction": float(max(
            result.peak_ice_fraction_by_cell[0], result.peak_ice_fraction_by_cell[4])),
        "max_pore_occupancy": result.max_pore_occupancy,
        "min_final_temp_c": float(np.min(result.final_temperature_c)),
        "water_balance_kg_m2": result.max_water_balance_kg_m2,
        "energy_balance_j": result.energy_balance_j,
    }


def run_case(spec):
    name, mode, factor, dt, powers, heat_s, time_cap = spec
    result = AuxStack(symmetric=False, mesh_factor=factor).simulate(
        mode, powers, heat_s, dt_max=dt, time_cap=time_cap,
        capture_interval=1000, temp_margin=TEMP_MARGIN_C)
    return record(name, result, mode, factor, dt)


def satisfies_q2_with_margin(row):
    """Apply the same conservative numerical margins used by the search."""
    return (row["success"]
            and row["min_final_temp_c"] >= TEMP_MARGIN_C - 1e-8
            and row["max_ice_fraction"] < ICE_LIMIT
            and row["min_voltage_v"] >= VOLTAGE_MARGIN_V)


def main():
    pre, selected, lower = load_policies()
    cases = [
        ("preheat_selected_84_dt0.05", "preheat", 3, .05,
         pre[0], pre[1], pre[1] + .1),
        ("selected_84_dt0.025", "cooperative", 3, .025,
         selected[0], selected[1], 310.0),
        ("selected_112_dt0.05", "cooperative", 4, .05,
         selected[0], selected[1], 310.0),
        ("lower_boundary_84_dt0.025", "cooperative", 3, .025,
         lower[0], lower[1], 310.0),
    ]
    rows = []
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_case, spec): spec[0] for spec in cases}
        for future in as_completed(futures):
            item = future.result()
            rows.append(item)
            rows.sort(key=lambda row: [case[0] for case in cases].index(row["case"]))
            (OUT / "verification.json").write_text(
                json.dumps({"status": "running", "rows": rows},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(item, ensure_ascii=False), flush=True)
    payload = {
        "status": "completed",
        "rows": rows,
        "acceptance": {
            "preheat_selected_valid": satisfies_q2_with_margin(rows[0]),
            "cooperative_84_valid": satisfies_q2_with_margin(rows[1]),
            "cooperative_112_valid": satisfies_q2_with_margin(rows[2]),
            "lower_energy_boundary_invalid": not satisfies_q2_with_margin(rows[3]),
            "temperature_margin_c": TEMP_MARGIN_C,
            "voltage_margin_v": VOLTAGE_MARGIN_V,
            "ice_limit": ICE_LIMIT,
        },
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_command": f"{sys.executable} 问题3/verify.py",
    }
    (OUT / "verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(payload["acceptance"][key] for key in (
            "preheat_selected_valid", "cooperative_84_valid",
            "cooperative_112_valid", "lower_energy_boundary_invalid")):
        raise RuntimeError(
            "Problem 3 verification failed; inspect results/verification.json")


if __name__ == "__main__":
    main()

"""Recompute and save the Q3 model-prediction evidence and figures."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aux_model import AuxStack

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

TEMP_MARGIN_C = 0.01
VOLTAGE_MARGIN_V = 0.3005
ICE_LIMIT = 0.99


def load_selected_policies():
    pre_search = json.loads((OUT / "preheat_search.json").read_text(encoding="utf-8"))
    co_search = json.loads((OUT / "cooperative_search.json").read_text(encoding="utf-8"))
    pre = pre_search["best"]
    co = co_search["best"]
    return ((pre["powers_w_cm2"], float(pre["heat_s"])),
            (co["powers_w_cm2"], float(co["heat_s"])))


def as_record(r):
    final_t = [float(x) for x in r.final_temperature_c]
    return {
        "mode": r.mode, "powers_w_cm2": list(r.powers_w_cm2),
        "requested_heat_s": r.requested_heat_s,
        "actual_heat_s": r.actual_heat_s,
        "energy_each_j": [float(x) for x in r.energy_each_j],
        "energy_total_j": r.heat_aux_j,
        "success": r.success, "reason": r.reason, "startup_s": r.startup_s,
        "charge_c_cm2": float(r.charge_c_cm2[-1]),
        "minimum_voltage_v": r.min_voltage_v,
        "minimum_voltage_by_cell_v": [float(x) for x in r.min_voltage_by_cell_v],
        "maximum_local_ice_fraction": r.max_ice_fraction,
        "peak_local_ice_fraction_by_cell": [float(x) for x in r.peak_ice_fraction_by_cell],
        "maximum_end_cell_ice_fraction": float(max(
            r.peak_ice_fraction_by_cell[0], r.peak_ice_fraction_by_cell[4])),
        "maximum_pore_occupancy": r.max_pore_occupancy,
        "minimum_gas_mol_m3": r.min_gas_mol_m3,
        "coldest_cell": int(np.argmin(r.final_temperature_c) + 1),
        "terminal_temperature_c": final_t,
        "terminal_plate_temperature_c": [float(x) for x in r.plate_temperature_c[-1]],
        "terminal_voltage_v": [float(x) for x in r.final_voltage_v],
        "terminal_ice_fraction": [float(x) for x in r.final_ice_fraction],
        "maximum_water_balance_error_kg_m2": r.max_water_balance_kg_m2,
        "energy_balance_error_j": r.energy_balance_j,
        "heat_electrochemical_j": r.heat_electrochemical_j,
        "heat_latent_j": r.heat_latent_j,
        "heat_aux_j": r.heat_aux_j,
        "heat_lost_j": r.heat_lost_j,
        "heat_stored_j": r.heat_stored_j,
        "minimum_temperature_at_preheat_end_c": r.min_temperature_before_load_c,
    }


def satisfies_q2_with_margin(r):
    return (r.success
            and float(np.min(r.final_temperature_c)) >= TEMP_MARGIN_C - 1e-8
            and r.max_ice_fraction < ICE_LIMIT
            and r.min_voltage_v >= VOLTAGE_MARGIN_V)


def main():
    OUT.mkdir(exist_ok=True)
    preheat, cooperative = load_selected_policies()
    model = AuxStack(symmetric=False, mesh_factor=3)
    pre = model.simulate("preheat", *preheat, dt_max=.05,
                         time_cap=preheat[1] + .1,
                         postload="q3_ramp", capture_interval=.1,
                         temp_margin=TEMP_MARGIN_C)
    print("preheat", pre.success, pre.reason, pre.startup_s, pre.heat_aux_j,
          pre.min_voltage_v, flush=True)
    co = model.simulate("cooperative", *cooperative, dt_max=.05, time_cap=310,
                        capture_interval=.5, temp_margin=TEMP_MARGIN_C)
    print("cooperative", co.success, co.reason, co.startup_s, co.heat_aux_j,
          co.min_voltage_v, flush=True)
    if not satisfies_q2_with_margin(pre) or not satisfies_q2_with_margin(co):
        raise RuntimeError(
            "A selected primary strategy failed the final Q2 constraints with margins")
    table = {"model_status": "model_prediction",
             "mesh_cells_per_cell": model.cell.n,
             "dt_max_s": .05,
             "initial_and_ambient_c": -30,
             "current_law": (
                 "Both strategies use j(tau)=min(0.005*tau,0.3) A/cm2; "
                 "tau=t for cooperative startup and tau=t-t_h after pure preheating"
             ),
             "charge_limit_primary": None,
             "preheat": as_record(pre),
             "cooperative": as_record(co)}
    table["comparison"] = {
        "cooperative_aux_energy_saving_percent": float(
            100.0 * (pre.heat_aux_j - co.heat_aux_j) / pre.heat_aux_j),
        "startup_time_difference_s": float(co.startup_s - pre.startup_s),
        "pure_preheat_end_cell_peak_ice_fraction": float(max(
            pre.peak_ice_fraction_by_cell[0], pre.peak_ice_fraction_by_cell[4])),
        "cooperative_end_cell_peak_ice_fraction": float(max(
            co.peak_ice_fraction_by_cell[0], co.peak_ice_fraction_by_cell[4])),
        "end_cell_peak_ice_by_strategy": {
            "pure_preheat": [float(pre.peak_ice_fraction_by_cell[0]),
                             float(pre.peak_ice_fraction_by_cell[4])],
            "cooperative": [float(co.peak_ice_fraction_by_cell[0]),
                            float(co.peak_ice_fraction_by_cell[4])],
        },
    }
    (OUT / "table4.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "table4.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "q1", "q2", "q3", "q4", "q5",
                    "heater_duration_s", "E1_J", "E2_J", "E3_J", "E4_J", "E5_J",
                    "total_aux_J", "startup_s", "max_ice_fraction",
                    "max_end_cell_ice_fraction", "min_voltage_V", "success"])
        for name, r in (("pure_preheat", pre), ("cooperative", co)):
            w.writerow([name, *r.powers_w_cm2, r.actual_heat_s,
                        *r.energy_each_j, r.heat_aux_j, r.startup_s,
                        r.max_ice_fraction,
                        max(r.peak_ice_fraction_by_cell[0], r.peak_ice_fraction_by_cell[4]),
                        r.min_voltage_v, r.success])
if __name__ == "__main__":
    main()

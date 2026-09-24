"""Recompute and save the Q3 model-prediction evidence and figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aux_model import AuxStack

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

PREHEAT = ([1.0] * 5, 195.0)
COOPERATIVE = ([.5, .32, .25, .32, .5], 4.0)


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
        "maximum_local_ice_fraction": r.max_ice_fraction,
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


def save_trajectory(path, r):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "current_Acm2", "charge_Ccm2",
                    *[f"T{k}_C" for k in range(1, 6)],
                    "Tplate_left_C", "Tplate_right_C",
                    *[f"V{k}_V" for k in range(1, 6)],
                    *[f"ice{k}_fraction" for k in range(1, 6)]])
        for i in range(len(r.time_s)):
            w.writerow([r.time_s[i], r.current_a_cm2[i], r.charge_c_cm2[i],
                        *r.temperature_c[i], *r.plate_temperature_c[i],
                        *r.voltage_v[i], *r.ice_fraction[i]])


def plot_results(pre, co):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, r in enumerate((pre, co)):
        t = r.time_s
        for k in range(5):
            axes[row, 0].plot(t, r.temperature_c[:, k], lw=1.2, label=f"cell {k+1}")
            axes[row, 1].plot(t, r.voltage_v[:, k], lw=1.2)
            axes[row, 2].plot(t, r.ice_fraction[:, k], lw=1.2)
        axes[row, 0].axhline(0, color="black", ls="--", lw=.8)
        axes[row, 1].axhline(.30, color="black", ls="--", lw=.8)
        for ax in axes[row]:
            ax.axvline(r.actual_heat_s, color="#d95f02", ls=":", lw=1)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Preheat: temperature (C)")
    axes[1, 0].set_ylabel("Cooperative: temperature (C)")
    axes[0, 1].set_ylabel("Cell voltage (V)")
    axes[1, 1].set_ylabel("Cell voltage (V)")
    axes[0, 2].set_ylabel("Local ice fraction")
    axes[1, 2].set_ylabel("Local ice fraction")
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.savefig(OUT / "strategy_trajectories.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(1, 6)
    ax.bar(x - .18, pre.energy_each_j, width=.36, label="pure preheat")
    ax.bar(x + .18, co.energy_each_j, width=.36, label="cooperative")
    ax.set(xlabel="Cell index", ylabel="Auxiliary heater energy (J)", xticks=x)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "energy_distribution.png", dpi=170)
    plt.close(fig)


def file_info(path):
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=True)
    model = AuxStack(symmetric=False, mesh_factor=3)
    pre = model.simulate("preheat", *PREHEAT, dt_max=.1, time_cap=196,
                         postload="step", capture_interval=.5)
    print("preheat", pre.success, pre.reason, pre.startup_s, pre.heat_aux_j,
          pre.min_voltage_v, flush=True)
    save_trajectory(OUT / "preheat_trajectory.csv", pre)
    co = model.simulate("cooperative", *COOPERATIVE, dt_max=.1, time_cap=300,
                        capture_interval=.5)
    print("cooperative", co.success, co.reason, co.startup_s, co.heat_aux_j,
          co.min_voltage_v, flush=True)
    save_trajectory(OUT / "cooperative_trajectory.csv", co)
    if not pre.success or not co.success:
        raise RuntimeError("A selected primary strategy failed the final simulation")
    table = {"model_status": "model_prediction",
             "mesh_cells_per_cell": model.cell.n,
             "dt_max_s": .1,
             "initial_and_ambient_c": -30,
             "preheat_postload": "immediate 0.3 A/cm2 step, explicitly assumed",
             "charge_limit_primary": None,
             "preheat": as_record(pre),
             "cooperative": as_record(co)}
    (OUT / "table4.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "table4.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "q1", "q2", "q3", "q4", "q5",
                    "heater_duration_s", "E1_J", "E2_J", "E3_J", "E4_J", "E5_J",
                    "total_aux_J", "startup_s", "max_ice_fraction", "min_voltage_V", "success"])
        for name, r in (("pure_preheat", pre), ("cooperative", co)):
            w.writerow([name, *r.powers_w_cm2, r.actual_heat_s,
                        *r.energy_each_j, r.heat_aux_j, r.startup_s,
                        r.max_ice_fraction, r.min_voltage_v, r.success])
    plot_results(pre, co)

    # Interpretive checks from the written plan: the temperature-ready lower
    # bound, the first 60 seconds after a gentle preheat ramp, and the Q2 charge
    # budget as an explicitly additional constraint.
    ready = model.simulate("preheat", [1]*5, 46.5, postload="ramp",
                           dt_max=.2, time_cap=47, capture_interval=.5)
    ready_follow = model.simulate("preheat", [1]*5, 46.5, postload="ramp",
                                  dt_max=.2, time_cap=106.5,
                                  capture_interval=.5, stop_on_success=False)
    budget_co = model.simulate("cooperative", *COOPERATIVE, dt_max=.2,
                               time_cap=120, charge_limit=20, capture_interval=1)
    budget_feasible = model.simulate("cooperative", [1]*5, 60, dt_max=.2,
                                     time_cap=120, charge_limit=20, capture_interval=1)
    alternate = {
        "temperature_ready_preheat": as_record(ready),
        "temperature_ready_60s_followup": {
            "reason": ready_follow.reason,
            "minimum_voltage_v": ready_follow.min_voltage_v,
            "minimum_temperature_after_load_c": float(np.min(
                ready_follow.temperature_c[ready_follow.time_s >= 46.5])),
            "terminal_min_temperature_c": float(min(ready_follow.final_temperature_c)),
        },
        "additional_20_Ccm2_budget_on_selected_cooperative": as_record(budget_co),
        "additional_20_Ccm2_feasible_comparison": as_record(budget_feasible),
    }
    (OUT / "interpretation_checks.json").write_text(
        json.dumps(alternate, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = [HERE / "aux_model.py", HERE / "search.py", HERE / "finalize.py",
             ROOT / "问题1/model.py", ROOT / "问题2/stack_model.py",
             ROOT / "result/metrics.json", HERE / "问题3的解决方案.md",
             *[OUT / name for name in ("preheat_search.json", "table4.json", "table4.csv",
                "preheat_trajectory.csv", "cooperative_trajectory.csv",
                "strategy_trajectories.png", "energy_distribution.png",
                "interpretation_checks.json")]]
    manifest = {"status": "completed_model_prediction",
                "command": f"{sys.executable} 问题3/finalize.py",
                "files": [file_info(p) for p in paths if p.exists()]}
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

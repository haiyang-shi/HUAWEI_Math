"""Calibrate on -20 C, validate on -25 C, and audit cold-start uncertainty."""

import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from data import EXPERIMENT_FILE, ROOT, load_experiments
from model import ColdStartModel, Parameters


OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
CHECKPOINTS = np.array([0, 5, 10, 15, 20, 25, 30, 35])
PARAMETER_NAMES = ["log10_j0", "plate_capacity_scale", "interface_ohm_1e4", "interface_hydration_exponent"]
LOWER = np.array([-2.0, 0.7, 0.0, 0.5])
UPPER = np.array([0.0, 1.6, 4.0, 7.0])


def params_from_vector(x):
    return Parameters(j0_ref_am2=10 ** x[0], plate_capacity_scale=x[1],
                      interface_ohm_m2=x[2] * 1e-4, interface_hydration_exponent=x[3])


def vector_from_params(p):
    return np.array([np.log10(p.j0_ref_am2), p.plate_capacity_scale,
                     p.interface_ohm_m2 / 1e-4, p.interface_hydration_exponent])


def scaled_residual(sim, exp):
    return np.r_[(sim.voltage_v - exp.voltage_v) / 0.020,
                 (sim.temperature_c - exp.temperature_c) / 0.30]


def scores(exp, sim, model):
    rv = sim.voltage_v - exp.voltage_v
    rt = sim.temperature_c - exp.temperature_c
    return {
        "voltage_mae_v": float(np.mean(np.abs(rv))),
        "voltage_rmse_v": float(np.sqrt(np.mean(rv**2))),
        "voltage_mean_relative_error_percent": float(np.mean(np.abs(rv) / np.abs(exp.voltage_v)) * 100),
        "temperature_mae_c": float(np.mean(np.abs(rt))),
        "temperature_rmse_c": float(np.sqrt(np.mean(rt**2))),
        "temperature_mean_relative_error_percent_celsius": float(np.mean(np.abs(rt) / np.abs(exp.temperature_c)) * 100),
        "voltage_residual_lag1": float(np.corrcoef(rv[:-1], rv[1:])[0, 1]),
        "temperature_residual_lag1": float(np.corrcoef(rt[:-1], rt[1:])[0, 1]),
        "max_ice_fraction_end": float(sim.max_ice_fraction[-1]),
        "max_pore_ice_saturation_end": float(sim.max_pore_ice_saturation[-1]),
        "max_pore_occupancy_whole_run": float(np.max(sim.max_pore_occupancy)),
        "minimum_gas_concentration_mol_m3": float(np.min(sim.min_gas_concentration_molm3)),
        "minimum_voltage_v": float(np.min(sim.voltage_v)),
        "max_mass_balance_error_kg_m2": float(np.max(np.abs(sim.mass_balance_error_kgm2))),
        "max_energy_balance_error_j_m2": float(np.max(np.abs(sim.energy_balance_error_jm2))),
        "final_generated_water_kg_m2": float(sim.generated_water_kgm2[-1]),
        "final_retained_pore_water_kg_m2": float(sim.retained_pore_water_kgm2[-1]),
        "final_membrane_sorbed_increment_kg_m2": float(sim.membrane_sorbed_increment_kgm2[-1]),
        "final_escaped_water_kg_m2": float(sim.water_escaped_kgm2[-1]),
        "final_membrane_lambda": float(sim.membrane_lambda[-1]),
        "final_heat_generated_j_m2": float(sim.heat_generated_jm2[-1]),
        "final_latent_released_j_m2": float(sim.latent_released_jm2[-1]),
        "final_heat_lost_j_m2": float(sim.heat_lost_jm2[-1]),
        "final_heat_storage_j_m2": float(sim.heat_storage_jm2[-1]),
    }


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def checkpoint_rows(exp, sim):
    rows = []
    for target in CHECKPOINTS:
        i = int(np.argmin(np.abs(exp.time_s - target)))
        ve, vm = exp.voltage_v[i], sim.voltage_v[i]
        te, tm = exp.temperature_c[i], sim.temperature_c[i]
        rows.append([int(target), f"{ve:.3f}", f"{vm:.3f}", f"{100*abs(vm-ve)/abs(ve):.2f}",
                     f"{te:.2f}", f"{tm:.2f}", f"{100*abs(tm-te)/abs(te):.2f}",
                     f"{sim.max_ice_fraction[i]:.5f}"])
    return rows


def write_results(experiments, simulations, model):
    header = ["时间/s", "实验电压/V", "模型电压/V", "电压相对误差/%", "实验温度/℃",
              "模型温度/℃", "温度相对误差/%", "模型最大冰体积分数"]
    for label, sim in simulations.items():
        exp = experiments[label]
        stem = "minus20" if label == "-20℃" else "minus25"
        write_csv(OUT / f"table_{stem}.csv", header, checkpoint_rows(exp, sim))
        write_csv(OUT / f"trajectory_{stem}.csv",
                  ["time_s", "current_density_Acm2", "voltage_exp_V", "voltage_model_V",
                   "temperature_exp_C", "temperature_model_C", "max_ice_volume_fraction",
                   "max_pore_ice_saturation", "membrane_lambda_mean", "plate_anode_C", "plate_cathode_C",
                   "generated_water_kgm2", "retained_pore_water_kgm2", "membrane_sorbed_kgm2",
                   "escaped_water_kgm2", "mass_balance_error_kgm2", "heat_generated_Jm2",
                   "latent_released_Jm2", "heat_lost_Jm2", "heat_storage_Jm2", "energy_balance_error_Jm2"],
                  zip(exp.time_s, exp.current_density_acm2, exp.voltage_v, sim.voltage_v,
                      exp.temperature_c, sim.temperature_c, sim.max_ice_fraction,
                      sim.max_pore_ice_saturation, sim.membrane_lambda,
                      sim.plate_temperature_c[:, 0], sim.plate_temperature_c[:, 1],
                      sim.generated_water_kgm2, sim.retained_pore_water_kgm2,
                      sim.membrane_sorbed_increment_kgm2, sim.water_escaped_kgm2,
                      sim.mass_balance_error_kgm2, sim.heat_generated_jm2,
                      sim.latent_released_jm2, sim.heat_lost_jm2,
                      sim.heat_storage_jm2, sim.energy_balance_error_jm2))
        i = int(np.argmin(abs(exp.time_s - 35)))
        write_csv(OUT / f"field_{stem}_35s.csv",
                  ["layer", "x_center_um", "temperature_C", "total_pore_water_kgm3",
                   "ice_kgm3", "ice_volume_fraction", "membrane_lambda"],
                  zip(model.names, (np.cumsum(model.dx) - model.dx / 2) * 1e6,
                      sim.temperature_field_c[i], sim.water_field_kgm3[i], sim.ice_field_kgm3[i],
                      sim.ice_field_kgm3[i] / 920, sim.membrane_lambda_field[i]))


def block_bootstrap_parameter_draws(exp, sim, model, params, count=16, seed=20260923):
    """Local paired residual block bootstrap; retains serial and V/T correlation."""
    x = vector_from_params(params)
    n = len(exp.time_s)
    jac = np.empty((2 * n, len(x)))
    for k in range(len(x)):
        h = 1e-3 * max(abs(x[k]), 1)
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        jac[:, k] = (scaled_residual(model.simulate(exp, params_from_vector(xp)), exp)
                     - scaled_residual(model.simulate(exp, params_from_vector(xm)), exp)) / (2 * h)
    residual = scaled_residual(sim, exp)
    rng = np.random.default_rng(seed)
    draws = []
    block = 12  # 2.4 s per block; sensitivity to block length is a limitation
    for _ in range(count):
        indices = np.concatenate([np.arange(start, start + block) % n
                                  for start in rng.integers(0, n, size=int(np.ceil(n / block)))])[:n]
        target = np.r_[residual[:n][indices], residual[n:][indices]]
        delta = np.linalg.lstsq(jac, target, rcond=None)[0]
        draw = np.clip(x + delta, LOWER + 1e-6, UPPER - 1e-6)
        draws.append(draw)
    singular = np.linalg.svd(jac, compute_uv=False)
    return np.array(draws), {"method": "paired 2.4 s residual block bootstrap with local linear refit",
                              "seed": seed, "draw_count": count, "parameter_names": PARAMETER_NAMES,
                              "sensitivity_singular_values": singular.tolist(),
                              "sensitivity_condition_number": float(singular[0] / singular[-1])}


def uncertainty(experiments, model, params, calibration_sim):
    draws, audit = block_bootstrap_parameter_draws(experiments["-20℃"], calibration_sim, model, params)
    scenarios = [(0.2, 0.2), (1.0, 1.0), (5.0, 5.0)]
    envelopes = {}
    for label, exp in experiments.items():
        runs = []
        for x in draws:
            p = params_from_vector(x)
            for freeze, liquid in scenarios:
                p_nuisance = replace(p, freeze_rate_s=p.freeze_rate_s * freeze,
                                     liquid_diffusivity_scale=p.liquid_diffusivity_scale * liquid)
                sim = model.simulate(exp, p_nuisance)
                runs.append(np.column_stack([sim.voltage_v, sim.temperature_c, sim.max_ice_fraction]))
        array = np.stack(runs)
        envelopes[label] = {
            "time_s": exp.time_s.tolist(),
            "voltage_v_q025_q975": np.quantile(array[:, :, 0], [.025, .975], axis=0).tolist(),
            "temperature_c_q025_q975": np.quantile(array[:, :, 1], [.025, .975], axis=0).tolist(),
            "max_ice_fraction_q025_q975": np.quantile(array[:, :, 2], [.025, .975], axis=0).tolist(),
            "end_ice_scenario_min_max": [float(np.min(array[:, -1, 2])), float(np.max(array[:, -1, 2]))],
        }
    audit["nuisance_freeze_and_liquid_scales"] = scenarios
    audit["interpretation"] = "Exploratory scenario envelope, not a validated probability interval for unobserved ice."
    return audit, envelopes


def plot_results(experiments, simulations, envelopes, ablations):
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"-20℃": "tab:blue", "-25℃": "tab:orange"}
    for label, exp in experiments.items():
        sim = simulations[label]
        color = colors[label]
        axes[0, 0].plot(exp.time_s, exp.voltage_v, color=color, alpha=.45, label=f"{label} observed")
        axes[0, 0].plot(exp.time_s, sim.voltage_v, color=color, linestyle="--", label=f"{label} model")
        axes[0, 1].plot(exp.time_s, exp.temperature_c, color=color, alpha=.45)
        axes[0, 1].plot(exp.time_s, sim.temperature_c, color=color, linestyle="--")
        axes[1, 0].plot(exp.time_s, sim.max_ice_fraction, color=color, label=label)
        axes[1, 1].plot(exp.time_s, sim.voltage_v - exp.voltage_v, color=color, label=label)
        bounds = envelopes[label]["max_ice_fraction_q025_q975"]
        axes[1, 0].fill_between(exp.time_s, bounds[0], bounds[1], color=color, alpha=.13)
    axes[0, 0].set(ylabel="Voltage (V)", xlabel="Time (s)")
    axes[0, 1].set(ylabel="Mean temperature (°C)", xlabel="Time (s)")
    axes[1, 0].set(ylabel="Maximum ice volume fraction", xlabel="Time (s)")
    axes[1, 1].set(ylabel="Voltage residual (V)", xlabel="Time (s)")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 1].axhline(0, color="black", linewidth=.7)
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.savefig(OUT / "validation_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for label, exp in experiments.items():
        sim = simulations[label]
        axes[0].plot(exp.time_s, sim.membrane_lambda, label=label)
        axes[1].plot(exp.time_s, sim.max_pore_ice_saturation, label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Mean membrane water content λ")
    axes[1].set(xlabel="Time (s)", ylabel="Maximum pore ice saturation")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(OUT / "water_ice_dynamics.png")
    plt.close(fig)

    exp = experiments["-20℃"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for name, sim in ablations.items():
        axes[0].plot(exp.time_s, sim.voltage_v, label=name)
        axes[1].plot(exp.time_s, sim.max_ice_fraction, label=name)
    axes[0].set(xlabel="Time (s)", ylabel="Voltage (V)")
    axes[1].set(xlabel="Time (s)", ylabel="Maximum ice volume fraction")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(OUT / "ablation_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for label, sim in simulations.items():
        axes[0].plot(sim.time_s, sim.mass_balance_error_kgm2, label=label)
        axes[1].plot(sim.time_s, sim.energy_balance_error_jm2, label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Water balance residual (kg/m²)")
    axes[1].set(xlabel="Time (s)", ylabel="Energy balance residual (J/m²)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(OUT / "balance_checks.png")
    plt.close(fig)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    experiments = load_experiments()
    cal = experiments["-20℃"]
    coarse = ColdStartModel()
    x0 = vector_from_params(Parameters(j0_ref_am2=.34, plate_capacity_scale=1.06,
                                       interface_ohm_m2=1.4e-4, interface_hydration_exponent=4.2))

    def residual(x, model, max_step):
        sim = model.simulate(cal, params_from_vector(x), max_step=max_step)
        return scaled_residual(sim, cal)

    fit0 = least_squares(lambda x: residual(x, coarse, .1), x0, bounds=(LOWER, UPPER),
                         loss="soft_l1", max_nfev=80, xtol=2e-5, ftol=2e-5, gtol=2e-5)
    model = ColdStartModel(mesh_factor=3)
    fit = least_squares(lambda x: residual(x, model, .05), fit0.x, bounds=(LOWER, UPPER),
                        loss="soft_l1", max_nfev=50, xtol=2e-5, ftol=2e-5, gtol=2e-5)
    params = params_from_vector(fit.x)
    simulations = {label: model.simulate(exp, params) for label, exp in experiments.items()}
    results = {label: scores(experiments[label], sim, model) for label, sim in simulations.items()}
    write_results(experiments, simulations, model)

    fine = ColdStartModel(mesh_factor=4).simulate(cal, params)
    half_step = model.simulate(cal, params, max_step=.025)
    reference = simulations["-20℃"]
    numerics = {
        "grid_84_to_112_voltage_max_diff_v": float(np.max(abs(fine.voltage_v - reference.voltage_v))),
        "grid_84_to_112_temperature_max_diff_c": float(np.max(abs(fine.temperature_c - reference.temperature_c))),
        "grid_84_to_112_ice_max_diff": float(np.max(abs(fine.max_ice_fraction - reference.max_ice_fraction))),
        "time_step_0p05_to_0p025_voltage_max_diff_v": float(np.max(abs(half_step.voltage_v - reference.voltage_v))),
        "time_step_0p05_to_0p025_temperature_max_diff_c": float(np.max(abs(half_step.temperature_c - reference.temperature_c))),
        "time_step_0p05_to_0p025_ice_max_diff": float(np.max(abs(half_step.max_ice_fraction - reference.max_ice_fraction))),
    }
    ablations = {
        "no ice": model.simulate(cal, params, freezing=False),
        "ice, no electrochemical feedback": model.simulate(cal, params, ice_feedback=False),
        "full ice feedback": reference,
    }
    ablation_metrics = {name: {"voltage_rmse_v": scores(cal, sim, model)["voltage_rmse_v"],
                                "temperature_rmse_c": scores(cal, sim, model)["temperature_rmse_c"],
                                "max_ice_fraction_end": float(sim.max_ice_fraction[-1])}
                        for name, sim in ablations.items()}
    bootstrap, envelopes = uncertainty(experiments, model, params, reference)
    plot_results(experiments, simulations, envelopes, ablations)
    payload = {
        "status": "computed_q1_only", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_condition": "-20℃", "independent_validation_condition": "-25℃",
        "calibration_success": bool(fit0.success and fit.success), "calibration_message": fit.message,
        "calibration_mesh_cells": model.n, "report_time_step_s": .05,
        "parameters": asdict(params), "metrics": results, "numerical_checks": numerics,
        "ablation": ablation_metrics, "bootstrap": bootstrap,
        "uncertainty_envelopes_path": "问题1/results/uncertainty_envelopes.json",
        "limitations": [
            "Ice and thawing have no direct observations in the supplied experiments.",
            "Experimental voltage and temperature were observed under nearly the same loading path at two starting temperatures.",
            "Membrane sorption and interface resistance remain reduced constitutive laws; independent water measurements are needed.",
            "Bootstrap intervals are local approximations; ice bands are exploratory nuisance-scenario envelopes.",
            "The current and current-density columns imply an active area inconsistent with attachment 1's 25 cm2.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "uncertainty_envelopes.json").write_text(json.dumps(envelopes, ensure_ascii=False), encoding="utf-8")
    paths = [Path(__file__), Path(__file__).parent / "model.py", Path(__file__).parent / "data.py",
             EXPERIMENT_FILE, ROOT / "problem_files/附件1.xlsx",
             ROOT / "problem_files/氢燃料电池低温冷启动建模与控制策略研究.pdf"]
    paths += sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "status": "completed_q1_only", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": "D:/Anaconda/python.exe 问题1/run.py", "calibration_condition": "-20℃",
        "validation_condition": "-25℃",
        "files": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                   "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"fit_success": fit.success, "parameters": asdict(params),
                      "metrics": results, "numerical_checks": numerics,
                      "ablation": ablation_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

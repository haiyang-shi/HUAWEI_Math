"""Calibrate Q1 on -20 C and validate the frozen parameter set on -25 C."""

import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from data import EXPERIMENT_FILE, ROOT, load_experiments
from model import ColdStartModel, Parameters


OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    for attempt in range(10):
        try:
            with path.open('w', encoding='utf-8-sig', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(rows)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.25)


def params_from_vector(x):
    return Parameters(
        j0_ref_am2=10 ** x[0],
        plate_capacity_scale=x[1],
        uptake_fraction=x[2],
        hydration_activity_exponent=x[3],
        dry_interface_ohm_m2=x[4] * 1e-4,
    )


def metrics(exp, sim, model):
    dv = sim.voltage_v - exp.voltage_v
    dt = sim.temperature_c - exp.temperature_c
    return {
        'voltage_mae_v': float(np.mean(np.abs(dv))),
        'voltage_rmse_v': float(np.sqrt(np.mean(dv ** 2))),
        'voltage_mean_relative_error_percent': float(np.mean(np.abs(dv) / np.abs(exp.voltage_v)) * 100),
        'temperature_mae_c': float(np.mean(np.abs(dt))),
        'temperature_rmse_c': float(np.sqrt(np.mean(dt ** 2))),
        'temperature_mean_relative_error_percent_celsius': float(np.mean(np.abs(dt) / np.abs(exp.temperature_c)) * 100),
        'max_ice_fraction_end': float(sim.max_ice_fraction[-1]),
        'max_ice_fraction_whole_run': float(np.max(sim.max_ice_fraction)),
        'max_pore_ice_saturation_whole_run': float(np.max(sim.max_pore_ice_saturation)),
        'max_pore_occupancy_whole_run': float(np.max(sim.max_pore_occupancy)),
        'minimum_gas_concentration_mol_m3': float(np.min(sim.min_gas_concentration_molm3)),
        'minimum_voltage_v': float(np.min(sim.voltage_v)),
        'max_mass_balance_error_kg_m2': float(np.max(np.abs(sim.mass_balance_error_kgm2))),
        'max_energy_balance_error_j_m2': float(np.max(np.abs(sim.energy_balance_error_jm2))),
        'final_heat_generated_j_m2': float(sim.heat_generated_jm2[-1]),
        'final_latent_released_j_m2': float(sim.latent_released_jm2[-1]),
        'final_heat_lost_j_m2': float(sim.heat_lost_jm2[-1]),
        'final_heat_storage_j_m2': float(sim.heat_storage_jm2[-1]),
        'final_generated_water_kg_m2': float(sim.generated_water_kgm2[-1]),
        'final_retained_pore_water_kg_m2': float(sim.water_field_kgm3[-1] @ model.dx),
        'final_membrane_sorbed_increment_kg_m2': float(sim.membrane_sorbed_increment_kgm2[-1]),
        'final_escaped_water_kg_m2': float(sim.water_escaped_kgm2[-1]),
        'final_membrane_lambda': float(sim.membrane_lambda[-1]),
    }


def checkpoint_table(exp, sim):
    rows = []
    for target in (0, 5, 10, 15, 20, 25, 30, 35):
        i = int(np.argmin(np.abs(exp.time_s - target)))
        ve = exp.voltage_v[i]
        te = exp.temperature_c[i]
        rows.append([
            target, f'{ve:.3f}', f'{sim.voltage_v[i]:.3f}', f'{100*abs(sim.voltage_v[i]-ve)/abs(ve):.2f}',
            f'{te:.2f}', f'{sim.temperature_c[i]:.2f}', f'{100*abs(sim.temperature_c[i]-te)/abs(te):.2f}',
            f'{sim.max_ice_fraction[i]:.5f}',
        ])
    return rows


def plot_results(experiments, simulations, ice_sensitivity):
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 180, 'font.size': 10})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {'-20℃': 'tab:blue', '-25℃': 'tab:orange'}
    for label, exp in experiments.items():
        sim = simulations[label]
        c = colors[label]
        ax[0, 0].plot(exp.time_s, exp.voltage_v, color=c, alpha=.45, label=f'{label} measured')
        ax[0, 0].plot(exp.time_s, sim.voltage_v, color=c, linestyle='--', label=f'{label} model')
        ax[0, 1].plot(exp.time_s, exp.temperature_c, color=c, alpha=.45)
        ax[0, 1].plot(exp.time_s, sim.temperature_c, color=c, linestyle='--')
        ax[1, 0].plot(exp.time_s, sim.max_ice_fraction, color=c, label=label)
        ax[1, 1].plot(exp.time_s, sim.voltage_v-exp.voltage_v, color=c, label=label)
    ax[0, 0].set(ylabel='Voltage (V)', xlabel='Time (s)')
    ax[0, 0].legend(ncol=2, fontsize=8)
    ax[0, 1].set(ylabel='Mean temperature (°C)', xlabel='Time (s)')
    ax[1, 0].set(ylabel='Maximum ice volume fraction', xlabel='Time (s)')
    ax[1, 1].set(ylabel='Voltage residual (V)', xlabel='Time (s)')
    ax[1, 1].axhline(0, color='black', linewidth=.7)
    for a in ax.flat:
        a.grid(alpha=.25)
    ax[1, 0].legend()
    ax[1, 1].legend()
    fig.savefig(OUT / 'validation_curves.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for label, exp in experiments.items():
        sim = simulations[label]
        ax[0].plot(exp.time_s, sim.membrane_lambda, label=label)
        ax[1].plot(exp.time_s, sim.max_pore_ice_saturation, label=label)
    ax[0].set(xlabel='Time (s)', ylabel='Mean membrane water content λ')
    ax[1].set(xlabel='Time (s)', ylabel='Maximum pore ice saturation')
    for a in ax:
        a.grid(alpha=.25)
        a.legend()
    fig.savefig(OUT / 'water_ice_dynamics.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for label, sim in simulations.items():
        ax[0].plot(sim.time_s, sim.mass_balance_error_kgm2, label=label)
    for name, curve in ice_sensitivity.items():
        ax[1].plot(experiments['-20℃'].time_s, curve, label=name)
    ax[0].set(xlabel='Time (s)', ylabel='Water balance residual (kg/m²)')
    ax[1].set(xlabel='Time (s)', ylabel='Maximum ice volume fraction')
    for a in ax:
        a.grid(alpha=.25)
        a.legend(fontsize=8)
    fig.savefig(OUT / 'balance_and_ice_sensitivity.png')
    plt.close(fig)


def main():
    experiments = load_experiments()
    calibration = experiments['-20℃']
    coarse_model = ColdStartModel()
    evaluations = [0]

    def residual(x):
        evaluations[0] += 1
        sim = coarse_model.simulate(calibration, params_from_vector(x))
        # One temperature setting only; -25 C is not exposed to the optimizer.
        return np.r_[(sim.voltage_v-calibration.voltage_v)/0.020,
                     (sim.temperature_c-calibration.temperature_c)/0.30]

    initial = np.array([-1.0, 1.0, .75, 1.5, .6])
    bounds = ([-2.0, .5, .15, 0.0, 0.0], [0.0, 1.8, 1.0, 4.0, 2.0])
    coarse_fit = least_squares(residual, initial, bounds=bounds, loss='soft_l1',
                        f_scale=1.0, max_nfev=85, xtol=2e-5, ftol=2e-5, gtol=2e-5)
    model = ColdStartModel(mesh_factor=3)

    def refined_residual(x):
        evaluations[0] += 1
        sim = model.simulate(calibration, params_from_vector(x), max_step=.05)
        return np.r_[(sim.voltage_v-calibration.voltage_v)/0.020,
                     (sim.temperature_c-calibration.temperature_c)/0.30]

    fit = least_squares(refined_residual, coarse_fit.x, bounds=bounds, loss='soft_l1',
                        f_scale=1.0, max_nfev=35, xtol=2e-5, ftol=2e-5, gtol=2e-5)
    params = params_from_vector(fit.x)
    sims = {name: model.simulate(exp, params, max_step=.05) for name, exp in experiments.items()}
    result = {name: metrics(experiments[name], sim, model) for name, sim in sims.items()}

    headers = ['时间/s', '实验电压/V', '模型电压/V', '电压相对误差/%', '实验温度/℃',
               '模型温度/℃', '温度相对误差/%', '模型最大冰体积分数']
    for label, sim in sims.items():
        exp = experiments[label]
        stem = 'minus20' if label == '-20℃' else 'minus25'
        write_csv(OUT / f'table_{stem}.csv', headers, checkpoint_table(exp, sim))
        write_csv(OUT / f'trajectory_{stem}.csv',
                  ['time_s', 'current_density_Acm2', 'voltage_exp_V', 'voltage_model_V',
                   'temperature_exp_C', 'temperature_model_C', 'max_ice_volume_fraction',
                   'max_pore_ice_saturation', 'membrane_lambda', 'generated_water_kgm2',
                   'retained_pore_water_kgm2', 'membrane_sorbed_kgm2', 'escaped_water_kgm2',
                   'heat_generated_Jm2', 'latent_released_Jm2', 'heat_lost_Jm2',
                   'heat_storage_Jm2', 'energy_balance_error_Jm2'],
                  zip(exp.time_s, exp.current_density_acm2, exp.voltage_v, sim.voltage_v,
                      exp.temperature_c, sim.temperature_c, sim.max_ice_fraction,
                      sim.max_pore_ice_saturation, sim.membrane_lambda,
                      sim.generated_water_kgm2, sim.water_field_kgm3 @ model.dx,
                      sim.membrane_sorbed_increment_kgm2, sim.water_escaped_kgm2,
                      sim.heat_generated_jm2, sim.latent_released_jm2,
                      sim.heat_lost_jm2, sim.heat_storage_jm2,
                      sim.energy_balance_error_jm2))
        write_csv(OUT / f'field_{stem}_35s.csv',
                  ['layer', 'x_center_um', 'temperature_C', 'total_pore_water_kgm3',
                   'ice_kgm3', 'ice_volume_fraction'],
                  zip(model.names, (np.cumsum(model.dx)-model.dx/2)*1e6,
                      sim.temperature_field_c[175], sim.water_field_kgm3[175],
                      sim.ice_field_kgm3[175], sim.ice_field_kgm3[175]/920))

    # Numerical grid and step-size checks; diagnostics are model predictions,
    # not additional experimental validation.
    fine_mesh = ColdStartModel(mesh_factor=4)
    fine_sim = fine_mesh.simulate(calibration, params, max_step=.05)
    step_sim = model.simulate(calibration, params, max_step=.025)
    numerics = {
        'grid_84_to_112_voltage_max_diff_v': float(np.max(np.abs(fine_sim.voltage_v-sims['-20℃'].voltage_v))),
        'grid_84_to_112_temperature_max_diff_c': float(np.max(np.abs(fine_sim.temperature_c-sims['-20℃'].temperature_c))),
        'grid_84_to_112_ice_max_diff': float(np.max(np.abs(fine_sim.max_ice_fraction-sims['-20℃'].max_ice_fraction))),
        'time_step_0p05_to_0p025_voltage_max_diff_v': float(np.max(np.abs(step_sim.voltage_v-sims['-20℃'].voltage_v))),
        'time_step_0p05_to_0p025_temperature_max_diff_c': float(np.max(np.abs(step_sim.temperature_c-sims['-20℃'].temperature_c))),
        'time_step_0p05_to_0p025_ice_max_diff': float(np.max(np.abs(step_sim.max_ice_fraction-sims['-20℃'].max_ice_fraction))),
    }
    sensitivities = {}
    ice_curves = {'reference': sims['-20℃'].max_ice_fraction}
    for factor in (.2, 5.0):
        for key in ('freeze_rate_s', 'liquid_diffusivity_scale'):
            variant = replace(params, **{key: getattr(params, key)*factor})
            sim = model.simulate(calibration, variant, max_step=.05)
            sensitivities[f'{key}_x{factor}'] = {
                'max_ice_fraction': float(np.max(sim.max_ice_fraction)),
                'voltage_rmse_v': float(np.sqrt(np.mean((sim.voltage_v-calibration.voltage_v)**2))),
            }
            ice_curves[f'{key} x{factor}'] = sim.max_ice_fraction
    plot_results(experiments, sims, ice_curves)

    payload = {
        'status': 'computed_q1_only', 'generated_at': datetime.now().isoformat(timespec='seconds'),
        'source_statement': 'problem_files/氢燃料电池低温冷启动建模与控制策略研究.pdf',
        'calibration_condition': '-20℃', 'independent_validation_condition': '-25℃',
        'calibration_success': bool(coarse_fit.success and fit.success), 'calibration_message': fit.message,
        'calibration_mesh_cells': model.n, 'report_time_step_s': .05,
        'calibration_function_evaluations': evaluations[0],
        'parameters': asdict(params), 'metrics': result,
        'numerical_checks': numerics, 'ice_sensitivity': sensitivities,
        'limitations': [
            'No experimental ice fraction is provided; ice predictions are not directly validated.',
            'MEA one-dimensional field uses quasi-steady gas and mean membrane hydration.',
            'The supplied current and current-density columns imply a different active area than 25 cm2.',
            'Experiments end below 0 C; thawing and successful startup cannot be validated here.',
        ],
    }
    (OUT / 'metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = [Path(__file__), Path(__file__).parent/'model.py', Path(__file__).parent/'data.py',
             EXPERIMENT_FILE, ROOT/'problem_files/附件1.xlsx',
             ROOT/'problem_files/氢燃料电池低温冷启动建模与控制策略研究.pdf']
    paths += sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'run_manifest.json')
    manifest = {
        'status': 'completed_q1_only', 'generated_at': datetime.now().isoformat(timespec='seconds'),
        'command': 'D:/Anaconda/python.exe 问题1/run.py',
        'calibration_condition': '-20℃', 'validation_condition': '-25℃',
        'files': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in paths],
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'fit_success': fit.success, 'parameters': asdict(params), 'metrics': result,
                      'numerical_checks': numerics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

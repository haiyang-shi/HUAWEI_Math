"""Optimize Q2 current schedules and write reproducible stack evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

from stack_model import StackModel, QMAX_C_CM2


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def decode(family, x):
    x = np.asarray(x, float)
    if family == 'constant':
        return [float(x[0])]
    if family == 'linear':
        return [float(x[0]), float(x[0]+(0.5-x[0])*x[1]), float(x[2])]
    j1 = x[0]
    j2 = j1+(0.5-j1)*x[1]
    j3 = j2+(0.5-j2)*x[2]
    return [float(j1), float(j2), float(j3), float(x[3]), float(x[3]+x[4])]


BOUNDS = {
    'constant': [(.02, .21)],
    'linear': [(.03, .21), (0, 1), (5, 160)],
    'step': [(.03, .21), (0, 1), (0, 1), (3, 100), (3, 120)],
}


def failure_score(result):
    if result.success:
        return result.startup_s
    tmin = float(np.min(result.temperature_c[-1]))
    if result.reason == 'voltage_below_0.30V':
        return 500+100*max(0, .30-result.min_voltage_v)+20*max(0, -tmin)
    return 150+15*max(0, -tmin)+.02*float(result.time_s[-1])


def run_optimizer(family, initial_c, seed, *, maxiter=8, popsize=6,
                  dt_max=.2, time_cap=300, candidate_x=None, voltage_margin=.005):
    model = StackModel()
    cache = {}

    def objective(x):
        key = tuple(np.round(x, 8))
        if key not in cache:
            try:
                result = model.simulate(family, decode(family, x), initial_c,
                                        dt_max=dt_max, time_cap=time_cap,
                                        capture_interval=1000,
                                        voltage_margin=voltage_margin)
                cache[key] = failure_score(result)
            except (ValueError, RuntimeError, FloatingPointError):
                cache[key] = 1e4
        return cache[key]

    for x in candidate_x or []:
        objective(np.asarray(x, float))
    solution = differential_evolution(objective, BOUNDS[family], seed=seed,
                                      maxiter=maxiter, popsize=popsize,
                                      polish=False, tol=.002, updating='immediate')
    best = (solution.fun, solution.x)
    for x in candidate_x or []:
        score = objective(np.asarray(x, float))
        if score < best[0]:
            best = (score, np.asarray(x, float))
    if best[0] < 150 and family != 'constant':
        local = minimize(objective, best[1], method='Nelder-Mead',
                         options={'maxiter': 45, 'xatol': .005, 'fatol': .005})
        if local.fun < best[0]:
            best = (local.fun, local.x)
    # Exclude optimizer steps that crossed bounds during local refinement.
    x = np.clip(best[1], *np.array(BOUNDS[family]).T)
    result = model.simulate(family, decode(family, x), initial_c,
                            dt_max=dt_max, time_cap=time_cap,
                            capture_interval=1000,
                            voltage_margin=voltage_margin)
    return {'family': family, 'initial_c': initial_c, 'seed': seed,
            'x': [float(z) for z in x], 'parameters': decode(family, x),
            'score': failure_score(result), 'success': result.success,
            'reason': result.reason, 'startup_s': result.startup_s,
            'charge_c_cm2': result.used_charge,
            'terminal_min_temperature_c': float(np.min(result.temperature_c[-1])),
            'minimum_voltage_v': result.min_voltage_v,
            'evaluations': len(cache)}


def scan_constant(initial_c=-10.0, dt_max=.5):
    model = StackModel()
    rows = []
    for current in np.arange(.02, .22001, .005):
        result = model.simulate('constant', [current], initial_c,
                                dt_max=dt_max, time_cap=400,
                                capture_interval=1000)
        rows.append({'j': round(float(current), 5), 'success': result.success,
                     'reason': result.reason, 't': float(result.time_s[-1]),
                     'charge': result.used_charge,
                     'terminal_min_t': float(np.min(result.temperature_c[-1])),
                     'minimum_voltage': result.min_voltage_v})
    return rows


def write_trajectory(path, result):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s', 'current_density_Acm2', 'charge_Ccm2',
                         *[f'T_cell_{i}_C' for i in range(1, 6)],
                         'T_endplate_left_C', 'T_endplate_right_C',
                         *[f'V_cell_{i}_V' for i in range(1, 6)],
                         *[f'ice_cell_{i}_max_fraction' for i in range(1, 6)]])
        for z in range(len(result.time_s)):
            writer.writerow([result.time_s[z], result.current_acm2[z], result.charge_c_cm2[z],
                             *result.temperature_c[z], *result.endplate_temperature_c[z],
                             *result.voltage_v[z], *result.ice_fraction[z]])


def result_record(result):
    cold = int(np.argmin(result.temperature_c[-1]))+1
    lowest_v = int(np.argmin(np.min(result.voltage_v, axis=0)))+1
    return {'family': result.family, 'parameters': list(result.parameters),
            'initial_c': result.initial_c, 'success': result.success,
            'reason': result.reason, 'startup_s': result.startup_s,
            'charge_c_cm2': result.used_charge,
            'max_current_acm2': float(np.max(result.current_acm2)),
            'minimum_voltage_v': result.min_voltage_v,
            'maximum_local_ice_fraction': result.max_ice_fraction,
            'coldest_cell': cold, 'lowest_voltage_cell': lowest_v,
            'terminal_temperature_c': [float(x) for x in result.temperature_c[-1]],
            'terminal_endplate_temperature_c': [float(x) for x in result.endplate_temperature_c[-1]],
            'max_water_balance_kg_m2': result.max_water_balance_kg_m2,
            'energy_balance_j': result.energy_balance_j,
            'heat_generated_j': result.heat_generated_j,
            'latent_heat_j': result.latent_heat_j,
            'heat_lost_j': result.heat_lost_j,
            'heat_stored_j': result.heat_stored_j,
            'min_gas_molm3': result.min_gas_molm3,
            'max_pore_occupancy': result.max_pore_occupancy}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    scan = scan_constant(dt_max=1.0 if args.quick else .5)
    (OUT / 'constant_scan.json').write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding='utf-8')
    print('constant', max(scan, key=lambda row: row['terminal_min_t']), flush=True)
    searches = []
    for family in ('linear', 'step'):
        for seed in ((5,) if args.quick else (5, 17)):
            item = run_optimizer(family, -10.0, seed,
                                 maxiter=8 if args.quick else 13,
                                 popsize=6 if args.quick else 7,
                                 candidate_x=([[.1, 1, 60]] if family=='linear' else [[.1, .5, 1, 30, 30]]))
            searches.append(item)
            print('opt', item, flush=True)
    best = {}
    for family in ('linear', 'step'):
        item = min([x for x in searches if x['family']==family], key=lambda x:x['score'])
        best[family] = item
    (OUT / 'search.json').write_text(json.dumps(searches, ensure_ascii=False, indent=2), encoding='utf-8')
    # Search output is provisional.  Only finalize.py writes table 3 and
    # trajectories after resolving voltage transients on the 84-cell grid.
    print('best coarse candidates', best, flush=True)


if __name__ == '__main__':
    main()

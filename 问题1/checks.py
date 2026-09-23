"""Independent acceptance checks for the question-one result contract."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from data import load_experiments


OUT = Path(__file__).resolve().parent / 'results'
ROOT = OUT.parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def main():
    experiments = load_experiments()
    metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'run_manifest.json').read_text(encoding='utf-8'))
    envelope = json.loads((OUT / 'uncertainty_envelopes.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['bytes'], item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
    assert metrics['calibration_success']
    assert metrics['calibration_condition'] == '-20℃'
    assert metrics['independent_validation_condition'] == '-25℃'
    for label, stem in [('-20℃', 'minus20'), ('-25℃', 'minus25')]:
        exp = experiments[label]
        rows = read_csv(OUT / f'trajectory_{stem}.csv')
        table = read_csv(OUT / f'table_{stem}.csv')
        field = read_csv(OUT / f'field_{stem}_35s.csv')
        assert len(rows) == len(exp.time_s) == 184
        assert len(table) == 8
        pem_lambda = np.array([float(row['membrane_lambda']) for row in field if row['layer'] == 'PEM'])
        assert len(pem_lambda) == 12 and np.ptp(pem_lambda) > 0.1
        assert abs(float(rows[-1]['plate_anode_C']) - float(rows[-1]['temperature_model_C'])) > 0.01
        assert len(envelope[label]['time_s']) == 184
        for key in ('voltage_v_q025_q975', 'temperature_c_q025_q975', 'max_ice_fraction_q025_q975'):
            bounds = np.asarray(envelope[label][key], dtype=float)
            assert bounds.shape == (2, 184)
            assert np.all(np.isfinite(bounds)) and np.all(bounds[0] <= bounds[1])
        for index, row in enumerate(rows):
            assert abs(float(row['time_s']) - exp.time_s[index]) < 1e-10
            assert abs(float(row['current_density_Acm2']) - exp.current_density_acm2[index]) < 1e-10
            assert abs(float(row['voltage_exp_V']) - exp.voltage_v[index]) < 1e-10
            assert abs(float(row['temperature_exp_C']) - exp.temperature_c[index]) < 1e-10
            assert float(row['max_ice_volume_fraction']) >= 0
            assert float(row['max_pore_ice_saturation']) < 1
            assert float(row['voltage_model_V']) > 0
            assert abs(float(row['mass_balance_error_kgm2'])) < 1e-10
            assert abs(float(row['energy_balance_error_Jm2'])) < 1e-3
            assert 0.3 <= float(row['membrane_lambda_mean']) <= 22
        for row in table:
            target = int(row['时间/s'])
            i = int(np.argmin(abs(exp.time_s - target)))
            assert abs(exp.time_s[i] - target) < 1e-10
            ve = float(row['实验电压/V'])
            vm = float(row['模型电压/V'])
            te = float(row['实验温度/℃'])
            tm = float(row['模型温度/℃'])
            assert abs(ve - exp.voltage_v[i]) < 0.0005
            assert abs(te - exp.temperature_c[i]) < 0.005
            assert abs(float(row['电压相对误差/%']) - 100*abs(vm-ve)/abs(ve)) < 0.15
            assert abs(float(row['温度相对误差/%']) - 100*abs(tm-te)/abs(te)) < 0.15
        item = metrics['metrics'][label]
        assert item['max_mass_balance_error_kg_m2'] < 1e-10
        assert item['max_energy_balance_error_j_m2'] < 1e-3
        assert item['max_pore_occupancy_whole_run'] < 1
        assert item['minimum_gas_concentration_mol_m3'] > 0
    numerical = metrics['numerical_checks']
    assert numerical['grid_84_to_112_voltage_max_diff_v'] < 1e-3
    assert numerical['time_step_0p05_to_0p025_temperature_max_diff_c'] < 1e-2
    assert len(metrics['bootstrap']['parameter_names']) == 4
    assert metrics['bootstrap']['sensitivity_condition_number'] < 1e5
    assert set(metrics['ablation']) == {'no ice', 'ice, no electrochemical feedback', 'full ice feedback'}
    print('Q1 acceptance checks: PASS (data, tables, mass/energy conservation, bounds, convergence, uncertainty, ablation)')


if __name__ == '__main__':
    main()

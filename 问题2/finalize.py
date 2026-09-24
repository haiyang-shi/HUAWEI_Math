"""Finalize audited Q2 numerical evidence from selected search candidates."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run import OUT, HERE, result_record, write_trajectory
from stack_model import StackModel, calibrated_parameters, QMAX_C_CM2


# Selected from coarse global search plus a focused grid around the voltage
# constraint.  Their numerical feasibility is checked again below at 84 cells.
CANDIDATES = {
    'constant': [0.2],
    'linear': [0.17, 0.5, 5.0],
    'step': [0.2, 0.38, 0.5, 4.0, 6.0],
}
DT = .05
BOUNDARY_DT = .0125
MESH_FACTOR = 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_row(record):
    return [record['family'], json.dumps(record['parameters']), record['success'],
            record['startup_s'], record['charge_c_cm2'], record['max_current_acm2'],
            record['minimum_voltage_v'], record['maximum_local_ice_fraction'],
            record['reason']]


def main():
    OUT.mkdir(exist_ok=True)
    model = StackModel(mesh_factor=MESH_FACTOR)
    results = {}
    records = {}
    for family, parameters in CANDIDATES.items():
        r = model.simulate(family, parameters, -10, dt_max=DT, time_cap=300,
                           capture_interval=.1)
        results[family] = r
        records[family] = result_record(r)
        write_trajectory(OUT / f'{family}_trajectory.csv', r)
        if abs(r.energy_balance_j) > 1e-3 or r.max_water_balance_kg_m2 > 1e-9:
            raise AssertionError(f'{family}: conservation failed')
        if r.success and (r.used_charge > 20+1e-8 or r.min_voltage_v < .30-1e-8
                          or r.max_ice_fraction >= .99):
            raise AssertionError(f'{family}: safety constraint failed')
        print(family, records[family], flush=True)
    (OUT / 'table3.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'table3.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['strategy', 'parameters', 'success', 'startup_s',
                         'charge_Ccm2', 'max_current_Acm2', 'min_cell_voltage_V',
                         'max_local_ice_fraction', 'outcome'])
        for r in records.values():
            writer.writerow(table_row(r))

    # Refined feasibility bracket for the linear schedule.  The control is
    # fixed here; hence this is a found feasible boundary, not a global proof.
    boundary = []
    for t0 in (-11.20, -11.24, -11.26, -11.27, -11.28, -11.29):
        r = model.simulate('linear', CANDIDATES['linear'], t0,
                           dt_max=BOUNDARY_DT, time_cap=100, capture_interval=.1)
        boundary.append(result_record(r))
        write_trajectory(OUT / f'boundary_{abs(t0):.2f}.csv', r)
        print('boundary', t0, r.success, r.reason,
              r.startup_s, r.used_charge, float(np.min(r.temperature_c[-1])), flush=True)
    (OUT / 'temperature_boundary.json').write_text(json.dumps(boundary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'temperature_boundary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['initial_temperature_C', 'success', 'startup_s', 'charge_Ccm2',
                         'terminal_min_temperature_C', 'minimum_voltage_V',
                         'coldest_cell', 'reason'])
        for r in boundary:
            writer.writerow([r['initial_c'], r['success'], r['startup_s'], r['charge_c_cm2'],
                             min(r['terminal_temperature_c']), r['minimum_voltage_v'],
                             r['coldest_cell'], r['reason']])

    params = calibrated_parameters()
    sensitivity = []
    for name, p, factor, link in [
        ('base', params, 10, 1),
        ('freeze_x0.2', replace(params, freeze_rate_s=.04), 10, 1),
        ('freeze_x5', replace(params, freeze_rate_s=1.0), 10, 1),
        ('liquid_transport_x0.2', replace(params, liquid_diffusivity_scale=.2), 10, 1),
        ('liquid_transport_x5', replace(params, liquid_diffusivity_scale=5), 10, 1),
        ('terminal_concentration_x1', params, 1, 1),
        ('thermal_resistance_x0.5', params, 10, .5),
        ('thermal_resistance_x2', params, 10, 2),
    ]:
        mod = StackModel(mesh_factor=1, params=p,
                         end_concentration_factor=factor, link_resistance_scale=link)
        r = mod.simulate('linear', CANDIDATES['linear'], -10,
                         dt_max=.2, time_cap=100, capture_interval=1000)
        item = result_record(r)
        item['case'] = name
        sensitivity.append(item)
    (OUT / 'sensitivity.json').write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 180})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {'constant': '#888888', 'linear': '#1874ba', 'step': '#ef8b2c'}
    for family, r in results.items():
        t = r.time_s
        color = colors[family]
        axes[0, 0].plot(t, np.min(r.temperature_c, axis=1), color=color, label=family)
        axes[0, 1].plot(t, np.min(r.voltage_v, axis=1), color=color, label=family)
        axes[1, 0].plot(t, r.charge_c_cm2, color=color, label=family)
        axes[1, 1].plot(t, np.max(r.ice_fraction, axis=1), color=color, label=family)
    axes[0, 0].axhline(0, color='black', ls='--', lw=.8)
    axes[0, 1].axhline(.30, color='black', ls='--', lw=.8)
    axes[1, 0].axhline(20, color='black', ls='--', lw=.8)
    labels = ['Coldest cell temperature (C)', 'Minimum cell voltage (V)',
              'Charge per area (C/cm2)', 'Maximum local ice volume fraction']
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel='Time (s)', ylabel=label)
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(OUT / 'strategy_comparison.png')
    plt.close(fig)

    bad = boundary[-2] if not boundary[-2]['success'] else boundary[-1]
    good = next((x for x in reversed(boundary) if x['success']), None)
    close_cold = min(bad['terminal_temperature_c'])
    text = f'''# 问题2求解结果与分析

本报告执行[问题2的解决方案](问题2的解决方案.md)：在问题1校准模型上建立五片串联、双端板热容和片间导热的电堆，按题面 20 C·cm⁻² 电荷量、0.5 A·cm⁻² 电流密度及单片全过程不低于 0.30 V 的要求计算。数值来自[可复算程序](stack_model.py)、[搜索程序](run.py)、[最终复核程序](finalize.py)和[完整结果](results/table3.json)。它们是**模型预测**；问题1实验仅覆盖 −20/−25 ℃ 的 36.6 s 窗口，没有直接验证本问的融冰和成功时刻。

## 1. 三类策略的表3结果

| 加载策略 | 搜索所得加载参数 | 启动时间/s | 累计电荷量/C·cm⁻² | 最大电流密度/A·cm⁻² | 最低单片电压/V | 最大局部冰体积分数 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 恒流 | $j={CANDIDATES['constant'][0]:.3f}$ | — | {records['constant']['charge_c_cm2']:.3f} | {records['constant']['max_current_acm2']:.3f} | {records['constant']['minimum_voltage_v']:.4f} | {records['constant']['maximum_local_ice_fraction']:.5f} | 未启动；电荷达到上限时最冷片 {min(records['constant']['terminal_temperature_c']):.3f} ℃ |
| 线性升载 | $j_0={CANDIDATES['linear'][0]:.3f}$，$j_1={CANDIDATES['linear'][1]:.3f}$，$t_r={CANDIDATES['linear'][2]:.2f}$ s | {records['linear']['startup_s']:.2f} | {records['linear']['charge_c_cm2']:.3f} | {records['linear']['max_current_acm2']:.3f} | {records['linear']['minimum_voltage_v']:.4f} | {records['linear']['maximum_local_ice_fraction']:.5f} | 成功 |
| 三段阶梯 | $j_1={CANDIDATES['step'][0]:.4f}$，$j_2={CANDIDATES['step'][1]:.4f}$，$j_3={CANDIDATES['step'][2]:.4f}$；$\\tau_1={CANDIDATES['step'][3]:.2f}$ s，$\\tau_2={CANDIDATES['step'][4]:.2f}$ s | {records['step']['startup_s']:.2f} | {records['step']['charge_c_cm2']:.3f} | {records['step']['max_current_acm2']:.3f} | {records['step']['minimum_voltage_v']:.4f} | {records['step']['maximum_local_ice_fraction']:.5f} | 成功 |

表中恒流行的电荷量是**未启动时的上限消耗**，并非启动消耗。恒流对 $0.020$–$0.220$ A·cm⁻² 以 0.005 间隔扫描；扫描中末时最接近 0 ℃ 的恒流为 0.200 A·cm⁻²，仍未启动。线性和阶梯参数由有界随机搜索、局部细化及临界区网格搜索取得；因此表中“最优”是**已搜索范围内的最佳可行候选**，尚无全局最优性证明。阶梯策略的 4 s、6 s 两次切换均发生在启动前，三段都实际执行。其最低电压仅比安全线高 {records['step']['minimum_voltage_v']-.30:.4f} V，对参数扰动敏感。

![五片电堆三种加载策略的温度、电压、电荷与冰量曲线](results/strategy_comparison.png)

## 2. 最低初始温度与失败单片

以 $T_{{\\rm amb}}=T_0$、各部件同温起始、线性策略参数固定为表3的最优候选，在 84 网格/片、0.0125 s 最大步长下，**{good['initial_c']:.2f} ℃ 可启动**，而 **{bad['initial_c']:.2f} ℃ 未启动**；后者在电荷耗尽时最冷片仍为 **{close_cold:.4f} ℃**。因此，对该固定策略，最低可启动温度被数值夹在 {bad['initial_c']:.2f} ℃ 与 {good['initial_c']:.2f} ℃ 之间，搜索间距 {abs(good['initial_c']-bad['initial_c']):.2f} ℃。临界温度的判定对时间步敏感：0.05 s 步长下 −11.28 ℃ 未启动，减至 0.0125 s 后变为可启动。因此更稳妥地将结果写作**约 −11.3 ℃（模型预测）**，不要把 0.01 ℃ 当成物理精度。这不是所有可能电流曲线的数学最优下界；当前没有足够的实验验证可将其解释为真实电堆的最低温度。[逐温度结果](results/temperature_boundary.csv)给出完整检查点。

低于边界时，第 **{bad['coldest_cell']}、5** 片是最后升至 0 ℃ 的两片；五片镜像对称，故两端同时构成热学瓶颈。失效记录中第1/5片约 {bad['terminal_temperature_c'][0]:.3f} ℃，第3片约 {bad['terminal_temperature_c'][2]:.3f} ℃，两块端板约 {bad['terminal_endplate_temperature_c'][0]:.3f} ℃。端板吸收反应热并向外散热，首尾电池的浓差极化系数又降低它们的电压，形成端部热量与电压的双重劣势。该失效案例首先触发的是**电荷耗尽而端片未达温度目标**；最低电压 {bad['minimum_voltage_v']:.4f} V，尚未跌破 0.30 V。

## 3. 守恒、阈值与可信度

三个表3工况的最大水量闭合误差为 {max(x['max_water_balance_kg_m2'] for x in records.values()):.2e} kg·m⁻²、最大全堆能量闭合误差为 {max(abs(x['energy_balance_j']) for x in records.values()):.2e} J。线性和阶梯方案均满足 $q\\le20$、$j\\le0.5$、全过程 $V_k\\ge0.30$ V。题目指定的冰阈值 0.99 是**控制体总体积中的冰体积分数**；本模型各多孔层初始孔隙率至多 0.8，且冰受孔容限制，所以这一阈值在本模型中不可能先于孔容约束触发，不能用“冰未达 0.99”证明冰过程已获验证。更有判别力的是电压、温度与孔隙冰饱和度。

作为数值与物理不确定性，需留意问题1的冻结动力学/液水迁移未被局部冰量实验标定，−10 ℃ 初温不在问题1的两组试验工况内，最低温度搜索还涉及更长时段。[敏感性结果](results/sensitivity.json)分别改变了冻结速率、液水迁移、端部浓差系数和热阻。冻结速率改变五倍时最大冰体积分数明显变化；片间及端板等效热阻减半时，固定线性策略在 −10 ℃ 反而未能启动，热阻加倍时约 26 s 启动。这说明端板连接的未知热阻对结果影响远大于表中启动时间的百分位差。若采用另一种端板热容或端部极化闭合，需重算表3与温度边界。

## 4. 复算口径

输入采用[问题1当前参数](../result/metrics.json)，面积 25 cm²，片间导热按五层 MEA 加 4 mm 双极板的串联热阻，端板独立热容均为 98.75 J·K⁻¹。利用首尾镜像对称性，以三组完整的一维单片状态代表五片，并用七节点网络计算两端板和五片的热交换。五片串联的电荷预算只积分一次电流密度；电压阈值在每个内部积分步及切换点检查。运行 `D:/Anaconda/python.exe 问题2/finalize.py` 可重新生成本报告引用的表、轨迹和图片；`D:/Anaconda/python.exe 问题2/run.py` 可重新执行候选参数搜索。表3使用每片 84 格和 0.05 s 最大步长，最低温度边界使用 0.0125 s 最大步长。
'''
    report = HERE / '问题2求解结果与分析.md'
    report.write_text(text, encoding='utf-8')

    files = [HERE / 'stack_model.py', HERE / 'run.py', HERE / 'finalize.py',
             HERE / '问题2的解决方案.md', ROOT / 'result/metrics.json']
    files += [p for p in OUT.iterdir() if p.is_file() and p.name != 'run_manifest.json']
    files += [report]
    manifest = {'status': 'completed_q2_model_prediction', 'command': 'D:/Anaconda/python.exe 问题2/finalize.py',
                'mesh_factor': MESH_FACTOR, 'table3_max_step_s': DT,
                'boundary_max_step_s': BOUNDARY_DT,
                'strategy_parameters': CANDIDATES,
                'files': [{'path': str(p.relative_to(HERE.parent)).replace('\\', '/'),
                           'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]}
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


ROOT = HERE.parent

if __name__ == '__main__':
    main()

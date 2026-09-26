"""Build the Chinese Problem 4 report directly from verified result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
REPORT = HERE / "问题4的解决方案与结果分析.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, digits=4):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main():
    table = load_json(OUT / "table_q4.json")
    control = load_json(OUT / "control_parameters.json")
    verification = load_json(OUT / "verification.json")
    calibration = load_json(ROOT / "result" / "metrics.json")
    scan = load_csv(OUT / "constant_strategy_cooling_scan.csv")
    rows = table["rows"]
    selected = control["selected_by_case"]
    baseline = {row["case"]: row for row in rows if row["strategy"] == "constant"}
    dynamic = {row["case"]: row for row in rows if row["strategy"] == "dynamic"}

    initial_lines = []
    for case in (1, 2, 3):
        item = selected[str(case)]
        cells = ", ".join(number(x, 4) for x in item["initial_cell_temperature_c"])
        plates = ", ".join(number(x, 4) for x in item["initial_plate_temperature_c"])
        initial_lines.append(
            f"| {case} | {item['cooling_minutes'] if item['cooling_minutes'] is not None else '完全冷却'} "
            f"| {cells} | {plates} |")

    result_lines = []
    for row in rows:
        label = "问题3恒功率" if row["strategy"] == "constant" else "问题4动态功率"
        result_lines.append(
            f"| {row['case']} | {label} | {row['power_strategy']} | "
            f"{number(row['startup_s'], 3)} | {number(row['auxiliary_energy_j'], 3)} | "
            f"{number(row['charge_c_cm2'], 3)} | "
            f"{number(row['max_temperature_spread_c'], 4)} | "
            f"{number(row['min_voltage_v'], 6)} | "
            f"{number(row['max_ice_fraction'], 6)} | "
            f"{'成功' if row['success'] else '失败'} |")

    comparison_lines = []
    for case in (1, 2, 3):
        b, d = baseline[case], dynamic[case]
        energy_change = 100.0 * (d["auxiliary_energy_j"] - b["auxiliary_energy_j"]) / b["auxiliary_energy_j"]
        comparison_lines.append(
            f"- 工况{case}：动态策略相对恒功率，启动时间变化 "
            f"{d['startup_s'] - b['startup_s']:+.3f} s，辅助能耗变化 "
            f"{d['auxiliary_energy_j'] - b['auxiliary_energy_j']:+.3f} J "
            f"（{energy_change:+.2f}%）；最低电压 {d['min_voltage_v']:.6f} V，"
            f"最大冰体积分数 {d['max_ice_fraction']:.6f}。")

    scan_lines = []
    for row in scan:
        if abs(float(row["cooling_minutes"]) % 10.0) < 1e-9:
            scan_lines.append(
                f"| {number(row['cooling_minutes'], 0)} | "
                f"{number(row['initial_min_cell_temperature_c'], 4)} | "
                f"{number(row['initial_temperature_spread_c'], 5)} | "
                f"{number(row['startup_s'], 3)} | "
                f"{number(row['auxiliary_energy_j'], 3)} | "
                f"{number(row['min_voltage_v'], 6)} | "
                f"{number(row['max_ice_fraction'], 6)} | "
                f"{'成功' if row['success'].lower() == 'true' else '失败'} |")

    parameter_rows = []
    for key in (
        "j0_ref_am2", "plate_capacity_scale", "freeze_rate_s", "melt_rate_s",
        "liquid_diffusivity_scale", "ice_area_exponent", "membrane_lambda",
        "uptake_fraction", "hydration_activity_exponent", "contact_ohm_m2",
        "dry_interface_ohm_m2",
    ):
        parameter_rows.append(f"| `{key}` | {calibration['parameters'][key]:.10g} |")

    sensitivity_lines = []
    for case in (1, 2, 3):
        values = selected[str(case)]["weight_sensitivity"]
        compact = ", ".join(
            f"$w_E={item['energy_weight']:.2f}\u2192a={item['selected_aggressiveness']:g}$"
            for item in values)
        sensitivity_lines.append(f"- 工况{case}：{compact}。")

    content = rf"""# 问题4：动态辅助加热控制策略与预冷温度场

## 1. 结论先行

本次重算没有沿用旧的“只看辅助能耗”结果。动态方案先满足功率、电压、冰量和启动成功硬约束，再要求启动时间不劣于同工况的问题3恒功率基线；在此基础上，对辅助能耗、启动时间、最大温差、最低电压和最大冰体积分数进行五目标 Pareto 筛选。最终自动验收状态为 **{verification['status']}**。

这种处理回答了此前“为什么问题4反而比问题2或问题3更慢”的质疑：若只最小化辅助能耗，模型会偏向少加热甚至不加热，依靠后续电化学产热慢慢升温，数学上节能但工程上启动慢。现在把“不比问题3恒功率更慢”写成明确的性能准入条件，慢而省的退化解不能进入最终方案。

## 2. 题目要求与判据

五片单电池各自具有动态功率 $q_k(t)$，并满足

$$0\le q_k(t)\le 1\ \mathrm{{W\,cm^{{-2}}}},\qquad k=1,\ldots,5.$$

恒功率与动态协同启动均从加载题定电流时开始加热。启动成功时刻 $t_s$ 定义为五片平均温度首次全部越过 $0\ ^\circ\mathrm C$，同时全过程满足

$$\min_{{k,t\le t_s}}V_k(t)\ge0.30\ \mathrm V,\qquad
\max_{{k,x,t\le t_s}}\varepsilon_{{\mathrm{{ice}},k}}(x,t)<0.99.$$

一旦五片均达到启动温度，程序在该时刻把五路电热丝全部置零。数值验收使用 0.3005 V 搜索裕量，最终仍按题目的 0.30 V 判据报告。

辅助能耗和启动电量分别为

$$E_{{\mathrm{{aux}}}}=A\sum_{{k=1}}^5\int_0^{{t_s}}q_k(t)\,\mathrm dt,$$

$$Q_s=\int_0^{{t_s}}j(t)\,\mathrm dt.$$

这里额外报告 $Q_s$，是为了区分“真正在更短时间内完成启动”和“仅仅少用电热丝、靠更长时间反应产热”两类方案。

## 3. 参数来源与可复算链路

`result/metrics.json` 已由附件1、附件2和问题1程序重新生成，而不是手填恢复。问题4显式读取其中的校准参数；主要值如下。

| 参数 | 重生成值 |
|---|---:|
{chr(10).join(parameter_rows)}

模型继承关系为：问题1单片水–热–电–冰模型 → 问题2五片/双端板热网络 → 问题3电流曲线与恒功率基线 → 问题4预冷初值、五路动态电热丝和多目标筛选。附件没有给出动态加热实验，因此问题4数值是“经问题1数据校准后的模型预测”，不能表述成新实验验证值。

## 4. 预冷温度场

沿电堆方向设置左端板、单片1–5、右端板共七个节点：

$$\boldsymbol C\frac{{\mathrm d\boldsymbol T}}{{\mathrm dt}}
=-\boldsymbol K\boldsymbol T+\boldsymbol bT_{{\mathrm{{amb}}}},
\qquad T_{{\mathrm{{amb}}}}=-30\ ^\circ\mathrm C.$$

令 $\boldsymbol\theta=\boldsymbol T-T_{{\mathrm{{amb}}}}\boldsymbol 1$，预冷阶段无电流、无电热丝，故

$$\boldsymbol\theta(t)=
\exp\!\left(-\boldsymbol C^{{-1}}\boldsymbol Kt\right)\boldsymbol\theta(0).$$

三个主工况的初值为：

| 工况 | 冷却时间/min | 单片1–5温度/℃ | 左、右端板温度/℃ |
|---|---:|---|---|
{chr(10).join(initial_lines)}

`precooling_temperature_field.csv` 给出 0–100 min、1 min 间隔的七节点宽表；`precooling_temperature_field_long.csv` 同时给出物理坐标 $x$、节点名称和温度，因而这里的 $T(x,t)$ 不再只是序号热图。

## 5. 动态功率控制律

控制器每 0.10 s（最终验算）读取 $T_k,V_k$，并以后向差分获得 $\dot T_k,\dot V_k$。以问题3恒功率向量 $\boldsymbol q^{{(3)}}$ 和持续时间 $t_h^{{(3)}}$ 为参考，离线候选参数 $a$ 生成

$$q_{{k,\mathrm{{ff}}}}=\min\!\left(1,a q_k^{{(3)}}\right),\qquad
t_{{h,\mathrm{{ff}}}}=a t_h^{{(3)}}.$$

在线控制由每片独立的四种状态组成：

1. **增强**：温度落后、升温不足，或预测电压接近安全边界时提高该片功率；
2. **维持**：前馈脉冲有效且无须增强时保持基准功率，低温电压风险出现时至少维持安全补热；
3. **降低**：脉冲结束后以 0.10 W/cm² 做一次平滑过渡；
4. **关闭**：该片无补热需求时关闭；五片均成功后强制全部关闭。

冰体积分数只用于模型安全验收，不作为现实中难以直接测量的反馈量。冰堵风险由低温、低电压、$\dot V_k$ 下降及短时预测电压联合识别。

## 6. 五目标选优方法

对每个工况枚举

$$a\in\{{0,0.5,0.75,1,1.25,1.5,2,3,4,6,8,12\}}.$$

候选先满足：启动成功、$V_{{\min}}\ge0.3005$ V、冰量小于 0.99，以及 $t_s^{{\rm dyn}}\le t_s^{{\rm const}}$。再在下列五维目标中删除被支配解：

$$\boldsymbol f=[E_{{\rm aux}},\ t_s,\ \Delta T_{{\max}},\ -V_{{\min}},\ \varepsilon_{{\rm ice,max}}].$$

在 Pareto 集内，各目标按本工况候选范围归一化，以权重

$$\boldsymbol w=[0.45,0.25,0.10,0.10,0.10]$$

计算到理想点的加权距离。能耗权重最高，体现“辅助加热总能耗最小为主要目标”；其余四项真实参与筛选，而不是事后只做展示。能耗权重敏感性结果为：

{chr(10).join(sensitivity_lines)}

## 7. 表4重算结果

| 工况 | 控制策略 | 功率控制策略 | 启动时间/s | 辅助能耗/J | 启动电量/(C·cm⁻²) | 最大温差/℃ | 最低单片电压/V | 最大冰体积分数 | 结果 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(result_lines)}

同工况比较如下：

{chr(10).join(comparison_lines)}

动态结果不允许用“零加热但更慢”的方案替代表中结果；实际采用的 $a$、候选全集、Pareto 标记、权重敏感性和高精度复核尝试均保存在 `results/multiobjective_candidates*.json` 与 `results/control_parameters.json` 中。

## 8. 为什么启动时间仍可能是百秒量级

即使动态方案不再比恒功率基线慢，完全冷却或深度预冷工况仍可能需要较长时间。这不是优化器遗漏启动时间，而是以下条件共同决定的物理结果：初温最低可达 $-30\ ^\circ\mathrm C$；启动要求是五片全部越过 $0\ ^\circ\mathrm C$；端板和外界持续吸热；电流在 60 s 内才由 0 升到 0.3 A/cm²；同时功率、电压和冰量均受硬约束。若进一步大幅压缩时间，只能把“时间优先级高于能耗”写进新目标，或提高允许的加热硬件功率，这都会改变题目当前的主目标或边界。

因此，应比较同一工况、同一电流曲线和同一成功判据下的恒功率/动态功率结果，而不能仅凭“问题编号”与问题2的某个时间直接横比。本文已经加入恒功率同基线的启动时间不退化约束，保证问题4的动态结果在自身比较口径内合理。

## 9. 10–100 min 预冷与恒功率冷启动

下面列出每 10 min 的摘要；程序实际按 5 min 间隔计算 19 个工况。

| 预冷时间/min | 初始最低单片温度/℃ | 初始温差/℃ | 启动时间/s | 辅助能耗/J | 最低电压/V | 最大冰量 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(scan_lines)}

温度场、启动时间、最低电压、冰量、能耗和启动电量的连续变化分别见 `precooling_temperature_field.png` 与 `constant_strategy_cooling_scan.png`。这些数据回答了题目第二小问，而不只给 20 min、40 min 两个离散点。

## 10. 数值验证与边界

最终主表采用每片 84 个有限体积网格（3×基础网格）、最大时间步 0.025 s。候选筛选采用同样空间网格和 0.05 s 时间步，并对入选方案回到 0.025 s 逐条复核。低分辨率模型曾把可行的问题3基线误判为欠压，因此没有用于最终选优。

自动检查覆盖：表4六行及全部字段、六个方案成功、全过程功率范围、电压和冰量约束、终点五片温度全部超过 0 ℃、终点五路加热全部关闭、动态加热非零、四种控制状态、候选网格、预冷温度场的尺寸/对称性、10–100 min 扫描范围和温度单调性。检查结果写入 `results/verification.json`。

适用范围必须明确：参数虽由附件重生成，但动态加热与冰量没有独立实验数据；Pareto 结果只对当前电流曲线、候选控制族、模型参数和约束成立。工程部署前仍需加入传感器噪声、执行器时滞、接触热阻不确定性并开展实堆验证。

## 11. 复现

在项目根目录运行：

```text
python 问题4/run.py
```

该入口在 `result/metrics.json` 缺失时先由附件重建问题1参数，然后依次生成问题4全部数值结果、自动验收、本文档和 SHA-256 运行清单。核心输出为：

- `results/table_q4.csv` / `table_q4.json`：表4；
- `results/case*_constant_trajectory.csv`、`case*_dynamic_trajectory.csv`：六条完整轨迹；
- `results/multiobjective_candidates*.json`：五目标候选、Pareto 与敏感性；
- `results/precooling_temperature_field*.csv`：$T(x,t)$；
- `results/constant_strategy_cooling_scan.csv`：10–100 min 恒功率结果；
- `results/verification.json`：逐项验收；
- `results/run_manifest.json`：输入、代码、报告和输出哈希。
"""
    REPORT.write_text(content, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    raise SystemExit(main())

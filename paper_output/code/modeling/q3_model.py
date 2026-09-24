"""Publish the verified Problem 3 cold-start results into paper contracts."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from result_contract_io import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    TABLES_DIR,
    rel,
    table_entry,
    upsert_question_contracts,
)


QUESTION = {
    "question_id": "Q3",
    "title": "问题三：电堆辅助冷启动策略建模与优化",
    "task_type": "机理仿真与约束优化",
    "baseline_model": "纯预加热与恒定功率协同启动对照",
    "main_model": "五片一维多相冷启动模型、七节点热网络与恒功率边界搜索",
}


def finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite result: {value}")
    return number


def main() -> int:
    source_dir = PROJECT_ROOT / "问题3" / "results"
    table_json_path = source_dir / "table4.json"
    verification_path = source_dir / "verification.json"
    table_csv_path = source_dir / "table4.csv"
    if not all(path.exists() for path in (table_json_path, verification_path, table_csv_path)):
        raise FileNotFoundError("Problem 3 result evidence is incomplete")

    table_data = json.loads(table_json_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    pre = table_data["preheat"]
    co = table_data["cooperative"]
    rows = verification.get("rows", [])
    selected_checks = [row for row in rows if str(row.get("case", "")).startswith("selected_")]
    lower_checks = [row for row in rows if str(row.get("case", "")).startswith("lower_boundary_")]
    if not pre.get("success") or not co.get("success"):
        raise RuntimeError("A primary Problem 3 strategy is not feasible")
    if len(selected_checks) < 2 or not all(row.get("success") for row in selected_checks):
        raise RuntimeError("The selected cooperative policy lacks two successful resolution checks")
    if not lower_checks or any(row.get("success") for row in lower_checks):
        raise RuntimeError("The lower cooperative boundary is not recorded as infeasible")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_out = TABLES_DIR / "table_q3_auxiliary_cold_start.csv"
    shutil.copyfile(table_csv_path, table_out)

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    energy_figure = figures_dir / "fig_q3_1.png"
    trajectory_figure = figures_dir / "fig_q3_2.png"
    boundary_figure = figures_dir / "fig_q3_3.png"
    shutil.copyfile(source_dir / "energy_distribution.png", energy_figure)
    shutil.copyfile(source_dir / "strategy_trajectories.png", trajectory_figure)

    boundary_rows = sorted(
        [lower_checks[0], selected_checks[0]], key=lambda item: float(item["heat_s"])
    )
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    xs = [float(item["heat_s"]) for item in boundary_rows]
    ys = [float(item["min_voltage_v"]) for item in boundary_rows]
    colors = ["#d62728" if not item["success"] else "#2ca02c" for item in boundary_rows]
    ax.plot(xs, ys, color="#4c78a8", lw=1.5, zorder=1)
    ax.scatter(xs, ys, c=colors, s=70, zorder=2)
    ax.axhline(0.30, color="black", ls="--", lw=1, label="0.30 V limit")
    for item, x, y in zip(boundary_rows, xs, ys):
        label = f"{float(item['energy_j']):.3f} J, {'pass' if item['success'] else 'fail'}"
        ax.annotate(
            label, (x, y),
            xytext=(-6, -16) if item["success"] else (6, 10),
            textcoords="offset points",
            ha="right" if item["success"] else "left",
            va="top" if item["success"] else "bottom",
            fontsize=9,
        )
    ax.set(xlabel="Auxiliary-heating duration (s)", ylabel="Minimum cell voltage (V)",
           title="Cooperative-start feasibility boundary")
    ax.grid(alpha=.25)
    ax.margins(x=.06, y=.12)
    ax.legend(loc="best")
    fig.savefig(boundary_figure, dpi=180)
    plt.close(fig)
    shutil.copyfile(boundary_figure, source_dir / "feasibility_boundary.png")

    pre_energy = finite(pre["energy_total_j"])
    co_energy = finite(co["energy_total_j"])
    saving = 100.0 * (pre_energy - co_energy) / pre_energy
    lower_energy = finite(lower_checks[0]["energy_j"])

    outputs = [
        {"name": "q3_table4", "path": rel(table_out)},
        {"name": "q3_energy_figure", "path": rel(energy_figure)},
        {"name": "q3_trajectory_figure", "path": rel(trajectory_figure)},
        {"name": "q3_boundary_figure", "path": rel(boundary_figure)},
        {"name": "q3_source_result", "path": rel(table_json_path)},
        {"name": "q3_resolution_verification", "path": rel(verification_path)},
    ]
    metrics = [
        {"metric_name": "pure_preheat_aux_energy", "metric_role": "纯预热辅助能耗", "value": pre_energy, "unit": "J"},
        {"metric_name": "cooperative_aux_energy", "metric_role": "协同启动辅助能耗", "value": co_energy, "unit": "J"},
        {"metric_name": "cooperative_energy_saving", "metric_role": "协同相对纯预热节能率", "value": saving, "unit": "%"},
        {"metric_name": "pure_preheat_startup_time", "metric_role": "纯预热总启动时间", "value": finite(pre["startup_s"]), "unit": "s"},
        {"metric_name": "cooperative_startup_time", "metric_role": "协同总启动时间", "value": finite(co["startup_s"]), "unit": "s"},
        {"metric_name": "cooperative_min_voltage", "metric_role": "协同全过程最低电压", "value": finite(co["minimum_voltage_v"]), "unit": "V"},
        {"metric_name": "cooperative_max_ice_fraction", "metric_role": "协同最大局部冰体积分数", "value": finite(co["maximum_local_ice_fraction"]), "unit": ""},
        {"metric_name": "cooperative_infeasible_lower_energy", "metric_role": "细时间步下不可行能耗下界", "value": lower_energy, "unit": "J"},
    ]
    tables = [
        table_entry(
            "Q3", "table_q3_auxiliary_cold_start", "不同辅助冷启动策略优化结果及性能对比",
            "给出两类策略的功率、时间、分片能耗、总能耗、启动时间、冰量、电压和成败。",
            table_out, "computed"
        )
    ]
    conclusions = [
        {
            "question_id": "Q3",
            "conclusion_text": (
                "纯预热应在五片越过0摄氏度后接入问题二的最优线性加载，"
                "而不是瞬时施加0.3 A/cm2；离散复算值为46.5 s和5812.5 J。"
            ),
            "evidence_status": "computed",
        },
        {
            "question_id": "Q3",
            "conclusion_text": (
                "恒功率协同候选采用(0.50,0.32,0.25,0.32,0.50) W/cm2加热4.10 s，"
                "辅助能耗193.725 J，总启动约272.8 s；4.05 s细步长复核因电压低于0.30 V失败。"
            ),
            "evidence_status": "computed",
        },
        {
            "question_id": "Q3",
            "conclusion_text": (
                "协同方案辅助电热丝能耗降低约96.67%，但启动慢约226.3 s，"
                "且首尾片最大局部冰体积分数约0.417，端部仍是主要瓶颈。"
            ),
            "evidence_status": "computed",
        },
    ]
    parameters = [
        {"name": "preheat_power_w_cm2", "value": pre["powers_w_cm2"]},
        {"name": "preheat_duration_s", "value": pre["actual_heat_s"]},
        {"name": "preheat_postload", "value": table_data["preheat_postload"]},
        {"name": "cooperative_power_w_cm2", "value": co["powers_w_cm2"]},
        {"name": "cooperative_duration_s", "value": co["actual_heat_s"]},
    ]

    upsert_question_contracts(
        question=QUESTION,
        result_summary=(
            "问题三已完成两类恒功率辅助冷启动优化。纯预热为五片各1 W/cm2、46.5 s、"
            "5812.5 J，并在越零后接入问题二最优线性加载；协同候选为"
            "(0.50,0.32,0.25,0.32,0.50) W/cm2、4.10 s、193.725 J，"
            "总启动约272.8 s且通过两组离散复核。"
        ),
        metrics=metrics,
        tables=tables,
        conclusions=conclusions,
        outputs=outputs,
        parameters=parameters,
        status="computed",
    )
    print("Q3 verified contracts published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

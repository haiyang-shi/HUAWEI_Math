"""Reproducible multi-criterion search for the Problem 4 controller.

The statement names auxiliary energy as the primary objective but also asks
that startup time, stack temperature spread, minimum cell voltage and maximum
ice fraction be considered.  This implementation evaluates a fixed,
deterministic aggressiveness grid, removes dominated candidates in the five
metrics, and selects an energy-oriented compromise from the Pareto set.  No
random optimizer or hidden penalty value is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamic_model import (
    AREA_CM2,
    AMBIENT_C,
    ConstantPolicy,
    DynamicAuxStack,
    FeedbackPolicy,
    ICE_LIMIT,
    PrecoolingModel,
    VOLTAGE_SEARCH_MARGIN_V,
    initial_state,
    result_record,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
Q3_RESULT = ROOT / "问题3" / "results" / "cooperative_search.json"
AGGRESSIVENESS_GRID = np.array(
    [0.0, 0.50, 0.75, 1.00, 1.25, 1.50,
     2.00, 3.00, 4.00, 6.00, 8.00, 12.00],
    dtype=float,
)
OBJECTIVE_NAMES = [
    "auxiliary_energy_j",
    "startup_s",
    "max_temperature_spread_c",
    "negative_min_voltage_v",
    "max_ice_fraction",
]
OBJECTIVE_WEIGHTS = np.array([0.45, 0.25, 0.10, 0.10, 0.10])


def json_dump(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_q3_policy():
    source = json.loads(Q3_RESULT.read_text(encoding="utf-8"))
    best = source["best"]
    return np.asarray(best["powers_w_cm2"], float), float(best["heat_s"])


def feasible(result, voltage_margin=VOLTAGE_SEARCH_MARGIN_V):
    return (result.success
            and result.min_voltage_v >= voltage_margin - 1e-9
            and result.max_ice_fraction < ICE_LIMIT
            and np.all(np.isfinite(result.energy_each_j)))


def simulate_feedback(stack, cells, plates, reference_powers,
                      reference_duration, aggressiveness, *, dt_max,
                      control_interval=0.25, capture_interval=1.0):
    policy = FeedbackPolicy(
        reference_powers, reference_duration, aggressiveness)
    return stack.simulate_policy(
        policy, strategy="dynamic_feedback",
        initial_cell_c=cells, initial_plate_c=plates,
        dt_max=dt_max, time_cap=360.0,
        control_interval=control_interval,
        capture_interval=capture_interval,
    )


def candidate_record(aggressiveness, result, reference_powers,
                     reference_duration):
    return {
        "aggressiveness": float(aggressiveness),
        "feedforward_powers_w_cm2": np.clip(
            np.asarray(reference_powers) * aggressiveness, 0.0, 1.0).tolist(),
        "feedforward_duration_s": float(reference_duration * aggressiveness),
        "success": bool(result.success),
        "reason": result.reason,
        "auxiliary_energy_j": float(result.auxiliary_energy_j),
        "startup_s": result.startup_s,
        "charge_c_cm2": float(result.charge_c_cm2[-1]),
        "max_temperature_spread_c": float(result.max_temperature_spread_c),
        "min_voltage_v": float(result.min_voltage_v),
        "max_ice_fraction": float(result.max_ice_fraction),
        "is_pareto": False,
        "selected": False,
    }


def objective_vector(row):
    return np.array([
        row["auxiliary_energy_j"], row["startup_s"],
        row["max_temperature_spread_c"], -row["min_voltage_v"],
        row["max_ice_fraction"],
    ], dtype=float)


def pareto_indices(rows):
    values = np.vstack([objective_vector(row) for row in rows])
    keep = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        dominated_by_another = np.all(
            values <= values[i] + 1e-12, axis=1) & np.any(
                values < values[i] - 1e-12, axis=1)
        if np.any(dominated_by_another):
            keep[i] = False
    return np.flatnonzero(keep).tolist()


def compromise_scores(rows, weights=OBJECTIVE_WEIGHTS):
    values = np.vstack([objective_vector(row) for row in rows])
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    span = np.where(high - low > 1e-12, high - low, 1.0)
    normalized = (values - low) / span
    return np.sqrt(np.sum(np.asarray(weights) * normalized ** 2, axis=1))


def select_compromise(records, startup_ceiling_s):
    feasible_records = [
        row for row in records
        if (row["success"] and row["startup_s"] is not None
            and row["min_voltage_v"] >= VOLTAGE_SEARCH_MARGIN_V
            and row["max_ice_fraction"] < ICE_LIMIT
            and row["startup_s"] <= startup_ceiling_s + 1e-9)
    ]
    if not feasible_records:
        raise RuntimeError(
            "No candidate satisfies safety and the no-startup-regression rule")
    front = [feasible_records[i] for i in pareto_indices(feasible_records)]
    scores = compromise_scores(front)
    for row, score in zip(front, scores):
        row["compromise_score"] = float(score)
        row["is_pareto"] = True
    selected = min(
        front,
        key=lambda row: (row["compromise_score"],
                         row["auxiliary_energy_j"], row["startup_s"]),
    )

    sensitivity = []
    for energy_weight in (0.35, 0.45, 0.55):
        rest = (1.0 - energy_weight) * np.array([5, 2, 2, 2]) / 11.0
        weights = np.r_[energy_weight, rest]
        local_scores = compromise_scores(front, weights)
        index = int(np.argmin(local_scores))
        sensitivity.append({
            "energy_weight": energy_weight,
            "weights": weights.tolist(),
            "selected_aggressiveness": front[index]["aggressiveness"],
            "score": float(local_scores[index]),
        })
    return selected, front, sensitivity


def _candidate_worker(payload):
    (cells, plates, reference_powers, reference_duration, aggressiveness,
     mesh_factor, dt_max) = payload
    stack = DynamicAuxStack(symmetric=False, mesh_factor=mesh_factor)
    result = simulate_feedback(
        stack, cells, plates, reference_powers, reference_duration,
        aggressiveness, dt_max=dt_max,
        control_interval=0.25, capture_interval=5.0)
    return candidate_record(
        aggressiveness, result, reference_powers, reference_duration)


def search_candidates(cells, plates, reference_powers, reference_duration,
                      startup_ceiling_s, *, mesh_factor=3, dt_max=0.05,
                      workers=3):
    payloads = [
        (cells, plates, reference_powers, reference_duration,
         float(aggressiveness), mesh_factor, dt_max)
        for aggressiveness in AGGRESSIVENESS_GRID
    ]
    if workers == 1:
        records = [_candidate_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_candidate_worker, payloads))
    for row in records:
        row["meets_startup_no_regression"] = bool(
            row["success"] and row["startup_s"] is not None
            and row["startup_s"] <= startup_ceiling_s + 1e-9)
        print(
            f"candidate a={row['aggressiveness']:.2f}: {row['reason']}, "
            f"E={row['auxiliary_energy_j']:.3f}, t={row['startup_s']}, "
            f"Vmin={row['min_voltage_v']:.6f}, "
            f"no_regression={row['meets_startup_no_regression']}", flush=True)
    selected, front, sensitivity = select_compromise(
        records, startup_ceiling_s)
    return selected, front, sensitivity, records


def validate_feedback(cells, plates, reference_powers, reference_duration,
                      aggressiveness, *, mesh_factor, dt_max):
    stack = DynamicAuxStack(symmetric=False, mesh_factor=mesh_factor)
    result = simulate_feedback(
        stack, cells, plates, reference_powers, reference_duration,
        aggressiveness, dt_max=dt_max,
        control_interval=0.10, capture_interval=0.10)
    if not feasible(result):
        raise RuntimeError(
            f"Selected feedback a={aggressiveness} failed fine validation: "
            f"{result.reason}, Vmin={result.min_voltage_v:.6f}")
    return result


def baseline_result(cells, plates, powers, duration, *, mesh_factor, dt_max,
                    capture_interval=0.25):
    stack = DynamicAuxStack(symmetric=False, mesh_factor=mesh_factor)
    policy = ConstantPolicy(powers, duration)
    return stack.simulate_policy(
        policy, strategy="q3_constant_power",
        initial_cell_c=cells, initial_plate_c=plates,
        dt_max=dt_max, time_cap=360.0,
        control_interval=0.25, capture_interval=capture_interval,
    )


def write_trajectory(path: Path, result):
    columns = ["time_s", "current_A_cm2", "charge_C_cm2"]
    columns += [f"T{k}_C" for k in range(1, 6)]
    columns += ["left_plate_C", "right_plate_C"]
    columns += [f"V{k}_V" for k in range(1, 6)]
    columns += [f"ice{k}" for k in range(1, 6)]
    columns += [f"q{k}_W_cm2" for k in range(1, 6)]
    columns += [f"mode{k}" for k in range(1, 6)]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i, t in enumerate(result.time_s):
            writer.writerow(
                [t, result.current_a_cm2[i], result.charge_c_cm2[i]]
                + result.temperature_c[i].tolist()
                + result.plate_temperature_c[i].tolist()
                + result.voltage_v[i].tolist()
                + result.ice_fraction[i].tolist()
                + result.heater_power_w_cm2[i].tolist()
                + result.controller_mode[i].tolist()
            )


def write_table(records):
    columns = [
        "case", "cooling_minutes", "strategy", "power_strategy",
        "aggressiveness", "startup_s", "auxiliary_energy_j", "charge_c_cm2",
        "max_temperature_spread_c", "min_voltage_v", "max_ice_fraction",
        "success", "reason",
    ]
    with (OUT / "table_q4.csv").open("w", encoding="utf-8-sig",
                                             newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in columns})
    json_dump(OUT / "table_q4.json", {
        "status": "completed_model_prediction",
        "objective_names": OBJECTIVE_NAMES,
        "selection_rule": (
            "hard constraints first; nondominated set in five requested "
            "metrics; normalized distance-to-ideal with declared weights"),
        "objective_weights": OBJECTIVE_WEIGHTS.tolist(),
        "rows": records,
    })


def cooling_field_output(model: PrecoolingModel):
    minutes, fields = model.history(100.0, 1.0)
    path = OUT / "precooling_temperature_field.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cooling_minutes"] + list(model.node_names))
        writer.writerows(np.c_[minutes, fields])
    with (OUT / "precooling_temperature_field_long.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "cooling_minutes", "node_index", "position_m", "node",
            "temperature_c"])
        for i, minute in enumerate(minutes):
            for k, name in enumerate(model.node_names):
                writer.writerow([
                    minute, k, model.positions_m[k], name, fields[i, k]])
    return minutes, fields


def cooling_scan(model, powers, duration, scan_step,
                 *, mesh_factor=3, dt_max=0.1):
    rows = []
    for minutes in np.arange(10.0, 100.0 + 0.5 * scan_step, scan_step):
        field = model.field(float(minutes))
        result = baseline_result(
            field[1:6], field[[0, 6]], powers, duration,
            mesh_factor=mesh_factor, dt_max=dt_max, capture_interval=5.0)
        rows.append({
            "cooling_minutes": float(minutes),
            "initial_min_cell_temperature_c": float(np.min(field[1:6])),
            "initial_max_cell_temperature_c": float(np.max(field[1:6])),
            "initial_temperature_spread_c": float(np.ptp(field[1:6])),
            "success": bool(result.success),
            "reason": result.reason,
            "startup_s": result.startup_s,
            "charge_c_cm2": float(result.charge_c_cm2[-1]),
            "auxiliary_energy_j": result.auxiliary_energy_j,
            "max_temperature_spread_c": result.max_temperature_spread_c,
            "min_voltage_v": result.min_voltage_v,
            "max_ice_fraction": result.max_ice_fraction,
        })
        print(f"cooling scan {minutes:.0f} min: {result.reason}, "
              f"startup={result.startup_s}, Vmin={result.min_voltage_v:.6f}",
              flush=True)
    with (OUT / "constant_strategy_cooling_scan.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_dump(OUT / "constant_strategy_cooling_scan.json", rows)
    return rows


def plots(records, results, minutes, fields, scan_rows, candidate_rows,
          positions_m):
    labels = [f"C{r['case']}-{('const' if r['strategy']=='constant' else 'dyn')}"
              for r in records]
    colors = ["#8B9DAB" if r["strategy"] == "constant" else "#E86A33"
              for r in records]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [
        ("auxiliary_energy_j", "Auxiliary energy (J)"),
        ("startup_s", "Startup time (s)"),
        ("max_temperature_spread_c", "Maximum cell spread (degC)"),
        ("min_voltage_v", "Minimum single-cell voltage (V)"),
        ("max_ice_fraction", "Maximum local ice fraction"),
        ("charge_c_cm2", "Charge to startup (C/cm2)"),
    ]
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        values = [np.nan if r[key] is None else r[key] for r in records]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Problem 4: constant and dynamic strategies")
    fig.savefig(OUT / "strategy_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row, case in enumerate((1, 2, 3)):
        result = results[(case, "dynamic")]
        for k in range(5):
            axes[row, 0].step(result.time_s, result.heater_power_w_cm2[:, k],
                              where="post", label=f"cell {k+1}")
            axes[row, 1].plot(result.time_s, result.temperature_c[:, k])
            axes[row, 2].plot(result.time_s, result.voltage_v[:, k])
        axes[row, 0].set_ylabel(f"Case {case}\nq (W/cm2)")
        axes[row, 1].set_ylabel("T (degC)")
        axes[row, 2].set_ylabel("V (V)")
        axes[row, 2].axhline(0.30, color="black", ls="--", lw=0.8)
        for ax in axes[row]:
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("Dynamic feedback trajectories")
    fig.savefig(OUT / "dynamic_control_trajectories.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    image = ax.imshow(
        fields.T, aspect="auto", origin="lower",
        extent=[minutes[0], minutes[-1], positions_m[0] * 1e3,
                positions_m[-1] * 1e3],
        cmap="coolwarm", vmin=-30, vmax=25)
    ax.set_yticks(positions_m * 1e3, PrecoolingModel.node_names)
    ax.set_xlabel("Cooling time (min)")
    ax.set_ylabel("Stack position (mm)")
    ax.set_title("Pre-cooling temperature field from 25 degC in -30 degC ambient")
    fig.colorbar(image, ax=ax, label="Temperature (degC)")
    fig.savefig(OUT / "precooling_temperature_field.png", dpi=220)
    plt.close(fig)

    scan_t = np.array([r["cooling_minutes"] for r in scan_rows])
    initial_t = np.array([r["initial_min_cell_temperature_c"] for r in scan_rows])
    startup = np.array([np.nan if r["startup_s"] is None else r["startup_s"]
                        for r in scan_rows])
    voltage = np.array([r["min_voltage_v"] for r in scan_rows])
    ice = np.array([r["max_ice_fraction"] for r in scan_rows])
    energy = np.array([r["auxiliary_energy_j"] for r in scan_rows])
    charge = np.array([r["charge_c_cm2"] for r in scan_rows])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(scan_t, initial_t, marker="o", ms=3)
    axes[0, 0].set_ylabel("Initial minimum cell T (degC)")
    axes[0, 1].plot(scan_t, startup, marker="o", ms=3)
    axes[0, 1].set_ylabel("Startup time (s)")
    axes[0, 2].plot(scan_t, energy, marker="o", ms=3)
    axes[0, 2].set_ylabel("Auxiliary energy (J)")
    axes[1, 0].plot(scan_t, voltage, marker="o", ms=3)
    axes[1, 0].axhline(0.30, color="black", ls="--", lw=0.8)
    axes[1, 0].set_ylabel("Minimum voltage (V)")
    axes[1, 1].plot(scan_t, ice, marker="o", ms=3)
    axes[1, 1].axhline(0.99, color="black", ls="--", lw=0.8)
    axes[1, 1].set_ylabel("Maximum ice fraction")
    axes[1, 2].plot(scan_t, charge, marker="o", ms=3)
    axes[1, 2].set_ylabel("Charge to startup (C/cm2)")
    for ax in axes.flat:
        ax.set_xlabel("Cooling time (min)")
        ax.grid(alpha=0.25)
    fig.suptitle("Q3 constant-power strategy versus pre-cooling duration")
    fig.savefig(OUT / "constant_strategy_cooling_scan.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for case, ax in zip((1, 2, 3), axes):
        rows = [row for row in candidate_rows if row["case"] == case
                and row["success"]]
        energy = np.array([row["auxiliary_energy_j"] for row in rows])
        startup = np.array([row["startup_s"] for row in rows])
        aggressiveness = np.array([row["aggressiveness"] for row in rows])
        scatter = ax.scatter(energy, startup, c=aggressiveness,
                             cmap="viridis", s=45)
        front = [row for row in rows if row["is_pareto"]]
        if front:
            front = sorted(front, key=lambda row: row["auxiliary_energy_j"])
            ax.plot([row["auxiliary_energy_j"] for row in front],
                    [row["startup_s"] for row in front], color="black",
                    lw=1.0, label="Pareto set")
        chosen = [row for row in rows if row["selected"]]
        if chosen:
            ax.scatter(chosen[0]["auxiliary_energy_j"], chosen[0]["startup_s"],
                       marker="*", s=220, color="#E63946", edgecolor="black",
                       label="selected", zorder=5)
        ax.set_title(f"Case {case}")
        ax.set_xlabel("Auxiliary energy (J)")
        ax.set_ylabel("Startup time (s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=axes, label="Aggressiveness a", shrink=0.85)
    fig.suptitle("Five-objective candidate screening (energy-time projection)")
    fig.savefig(OUT / "multiobjective_tradeoff.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-mesh", type=int, default=3)
    parser.add_argument("--validation-dt", type=float, default=0.025)
    parser.add_argument("--search-mesh", type=int, default=3)
    parser.add_argument("--search-dt", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--scan-step", type=float, default=5.0)
    parser.add_argument("--scan-mesh", type=int, default=1)
    parser.add_argument("--scan-dt", type=float, default=0.1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    precooling = PrecoolingModel()
    powers, duration = load_q3_policy()
    records, selected, results = [], {}, {}
    search_audit, all_candidate_rows = [], []
    for case_id in (1, 2, 3):
        cells, plates, cooling_minutes = initial_state(case_id, precooling)
        baseline = baseline_result(
            cells, plates, powers, duration,
            mesh_factor=args.validation_mesh, dt_max=args.validation_dt)
        if not feasible(baseline):
            raise RuntimeError(
                f"Case {case_id}: Q3 baseline failed fine reproduction: "
                f"{baseline.reason}, Vmin={baseline.min_voltage_v:.6f}")
        baseline_row = result_record(
            baseline, case_id=case_id, cooling_minutes=cooling_minutes,
            strategy_label="constant")
        baseline_row["power_strategy"] = (
            f"q={np.round(powers, 5).tolist()} W/cm2 for {duration:.5f} s")
        baseline_row["aggressiveness"] = None
        records.append(baseline_row)
        results[(case_id, "constant")] = baseline
        write_trajectory(OUT / f"case{case_id}_constant_trajectory.csv", baseline)

        chosen, front, sensitivity, candidates = search_candidates(
            cells, plates, powers, duration, baseline.startup_s,
            mesh_factor=args.search_mesh, dt_max=args.search_dt,
            workers=args.workers)
        selected_a = float(chosen["aggressiveness"])
        # A coarse step can reject a marginally safe policy.  Validate the
        # selected point and, if needed, move to the next higher-a Pareto/grid
        # point until the declared fine model confirms all hard constraints.
        validation_attempts = []
        feasible_as = sorted({
            float(row["aggressiveness"]) for row in candidates
            if row["success"] and row["meets_startup_no_regression"]
            and row["aggressiveness"] >= selected_a})
        dynamic = None
        for candidate_a in feasible_as:
            try:
                dynamic = validate_feedback(
                    cells, plates, powers, duration, candidate_a,
                    mesh_factor=args.validation_mesh, dt_max=args.validation_dt)
                if dynamic.startup_s > baseline.startup_s + 1e-9:
                    raise RuntimeError(
                        f"fine startup {dynamic.startup_s:.6f} s exceeds "
                        f"Q3 baseline {baseline.startup_s:.6f} s")
                selected_a = candidate_a
                validation_attempts.append({
                    "aggressiveness": candidate_a, "success": True,
                    "reason": dynamic.reason})
                break
            except RuntimeError as exc:
                dynamic = None
                validation_attempts.append({
                    "aggressiveness": candidate_a, "success": False,
                    "reason": str(exc)})
        if dynamic is None:
            raise RuntimeError(
                f"Case {case_id}: no coarse-feasible candidate passed fine validation")

        for row in candidates:
            row["case"] = case_id
            row["is_pareto"] = any(
                abs(row["aggressiveness"] - item["aggressiveness"]) < 1e-12
                for item in front)
            row["selected"] = abs(row["aggressiveness"] - selected_a) < 1e-12
        all_candidate_rows.extend(candidates)
        json_dump(OUT / f"multiobjective_candidates_case{case_id}.json", {
            "case": case_id,
            "objective_names": OBJECTIVE_NAMES,
            "weights": OBJECTIVE_WEIGHTS.tolist(),
            "candidates": candidates,
            "weight_sensitivity": sensitivity,
            "validation_attempts": validation_attempts,
        })
        dynamic_row = result_record(
            dynamic, case_id=case_id, cooling_minutes=cooling_minutes,
            strategy_label="dynamic")
        dynamic_row["aggressiveness"] = selected_a
        feedforward = np.clip(powers * selected_a, 0.0, 1.0)
        dynamic_row["power_strategy"] = (
            "independent boost/hold/reduce/off feedback; "
            f"a={selected_a:g}, feedforward q="
            f"{np.round(feedforward, 5).tolist()} W/cm2, "
            f"duration={duration * selected_a:.5f} s")
        records.append(dynamic_row)
        results[(case_id, "dynamic")] = dynamic
        selected[str(case_id)] = {
            "cooling_minutes": cooling_minutes,
            "initial_cell_temperature_c": cells.tolist(),
            "initial_plate_temperature_c": plates.tolist(),
            "coarse_selected_aggressiveness": chosen["aggressiveness"],
            "validated_aggressiveness": selected_a,
            "fine_metrics": {
                "auxiliary_energy_j": dynamic.auxiliary_energy_j,
                "startup_s": dynamic.startup_s,
                "max_temperature_spread_c": dynamic.max_temperature_spread_c,
                "min_voltage_v": dynamic.min_voltage_v,
                "max_ice_fraction": dynamic.max_ice_fraction,
            },
            "weight_sensitivity": sensitivity,
        }
        search_audit.append({
            "case": case_id,
            "coarse_evaluations": len(candidates),
            "coarse_feasible_count": sum(bool(row["success"]) for row in candidates),
            "coarse_pareto_count": len(front),
            "coarse_selected_aggressiveness": chosen["aggressiveness"],
            "validated_aggressiveness": selected_a,
            "validated_success": dynamic.success,
            "validated_energy_j": dynamic.auxiliary_energy_j,
            "validation_attempts": validation_attempts,
        })
        write_trajectory(OUT / f"case{case_id}_dynamic_trajectory.csv", dynamic)

    write_table(records)
    json_dump(OUT / "multiobjective_candidates.json", all_candidate_rows)
    with (OUT / "multiobjective_candidates.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_candidate_rows[0]))
        writer.writeheader()
        writer.writerows(all_candidate_rows)
    json_dump(OUT / "control_parameters.json", {
        "status": "completed",
        "controller": "Q3-referenced independent four-mode feedback",
        "feedback_signals": ["T_k", "V_k", "dT_k/dt", "dV_k/dt"],
        "modes": ["boost", "hold", "reduce", "off"],
        "offline_selection": {
            "candidate_aggressiveness_grid": AGGRESSIVENESS_GRID.tolist(),
            "objective_names": OBJECTIVE_NAMES,
            "objective_weights": OBJECTIVE_WEIGHTS.tolist(),
            "selection_rule": (
                "hard constraints, Pareto nondominance, then normalized "
                "weighted distance to the ideal point"),
            "startup_requirement": (
                "dynamic startup time must not exceed the Q3 constant-policy "
                "startup time in the same case"),
        },
        "fixed_settings": {
            "validation_control_interval_s": 0.10,
            "voltage_prediction_threshold_v": 0.305,
            "voltage_hold_threshold_v": 0.320,
            "prediction_horizon_s": 1.5,
            "voltage_drop_risk_v_s": -0.006,
            "temperature_rate_hold_c_s": 0.015,
            "temperature_lag_c": 0.15,
            "boost_increment_w_cm2": 0.20,
            "hold_power_w_cm2": 0.20,
            "reduce_power_w_cm2": 0.10,
        },
        "q3_constant_baseline": {
            "powers_w_cm2": powers.tolist(),
            "duration_s": duration,
            "source": str(Q3_RESULT.relative_to(ROOT)).replace("\\", "/"),
        },
        "selected_by_case": selected,
            "search_audit": search_audit,
        "validation": {
            "mesh_factor": args.validation_mesh,
            "dt_max_s": args.validation_dt,
            "search_mesh_factor": args.search_mesh,
            "search_dt_max_s": args.search_dt,
            "voltage_margin_v": VOLTAGE_SEARCH_MARGIN_V,
            "ice_limit": ICE_LIMIT,
        },
    })

    minutes, fields = cooling_field_output(precooling)
    scan_rows = cooling_scan(
        precooling, powers, duration, args.scan_step,
        mesh_factor=args.scan_mesh, dt_max=args.scan_dt)
    plots(records, results, minutes, fields, scan_rows, all_candidate_rows,
          precooling.positions_m)
    print(json.dumps({
        "table": str(OUT / "table_q4.json"),
        "rows": [{k: r[k] for k in (
            "case", "strategy", "success", "startup_s",
            "auxiliary_energy_j", "min_voltage_v", "max_ice_fraction")}
                 for r in records],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

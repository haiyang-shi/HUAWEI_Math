"""Independent consistency and requirement checks for Problem 4 outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    summary = json.loads((OUT / "table_q4.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    required = {
        "case", "strategy", "power_strategy", "startup_s",
        "auxiliary_energy_j", "charge_c_cm2", "max_temperature_spread_c",
        "min_voltage_v", "max_ice_fraction", "success", "reason",
    }
    check("six_main_rows", len(rows) == 6, f"rows={len(rows)}")
    check("table4_fields", all(required <= set(row) for row in rows),
          sorted(required))
    check("all_main_cases_success", all(row["success"] for row in rows),
          [row["reason"] for row in rows])
    check("voltage_constraint", all(row["min_voltage_v"] >= 0.30 for row in rows),
          min(row["min_voltage_v"] for row in rows))
    check("ice_constraint", all(row["max_ice_fraction"] < 0.99 for row in rows),
          max(row["max_ice_fraction"] for row in rows))
    check("finite_metrics", all(np.isfinite([
        row["startup_s"], row["auxiliary_energy_j"], row["charge_c_cm2"],
        row["max_temperature_spread_c"], row["min_voltage_v"],
        row["max_ice_fraction"],
    ]).all() for row in rows), "all requested metrics finite")
    dynamic_rows = [row for row in rows if row["strategy"] == "dynamic"]
    check("dynamic_heating_nonzero",
          all(row["auxiliary_energy_j"] > 1e-6 for row in dynamic_rows),
          [row["auxiliary_energy_j"] for row in dynamic_rows])

    trajectory_checks = []
    dynamic_mode_union = set()
    allowed_modes = {"boost", "hold", "reduce", "off"}
    for case in (1, 2, 3):
        for strategy in ("constant", "dynamic"):
            path = OUT / f"case{case}_{strategy}_trajectory.csv"
            data = load_csv(path)
            q = np.array([[float(row[f"q{k}_W_cm2"]) for k in range(1, 6)]
                          for row in data])
            v = np.array([[float(row[f"V{k}_V"]) for k in range(1, 6)]
                          for row in data])
            ice = np.array([[float(row[f"ice{k}"]) for k in range(1, 6)]
                            for row in data])
            temp = np.array([[float(row[f"T{k}_C"]) for k in range(1, 6)]
                             for row in data])
            modes = {row[f"mode{k}"] for row in data for k in range(1, 6)}
            if strategy == "dynamic":
                dynamic_mode_union |= modes
            terminal_off = bool(np.max(np.abs(q[-1])) < 1e-12)
            terminal_ready = bool(np.min(temp[-1]) > 0.0)
            item = {
                "case": case,
                "strategy": strategy,
                "rows": len(data),
                "power_min": float(np.min(q)),
                "power_max": float(np.max(q)),
                "sampled_voltage_min": float(np.min(v)),
                "sampled_ice_max": float(np.max(ice)),
                "terminal_min_temperature_c": float(np.min(temp[-1])),
                "terminal_heaters_off": terminal_off,
                "modes": sorted(modes),
                "passed": bool(np.min(q) >= -1e-12 and np.max(q) <= 1 + 1e-12
                               and modes <= allowed_modes
                               and np.min(v) >= 0.30 - 1e-8
                               and np.max(ice) < 0.99
                               and terminal_ready and terminal_off),
            }
            trajectory_checks.append(item)
    check("trajectory_bounds_and_shutdown",
          all(x["passed"] for x in trajectory_checks), trajectory_checks)
    check("four_controller_modes_exercised",
          allowed_modes <= dynamic_mode_union, sorted(dynamic_mode_union))

    candidates = json.loads(
        (OUT / "multiobjective_candidates.json").read_text(encoding="utf-8"))
    selected = [row for row in candidates if row["selected"]]
    check("candidate_grid_complete", len(candidates) == 36,
          f"candidates={len(candidates)}")
    check("one_selected_per_case", len(selected) == 3 and
          {row["case"] for row in selected} == {1, 2, 3}, selected)
    check("selected_candidates_feasible",
          all(row["success"] and row["min_voltage_v"] >= 0.3005
              and row["max_ice_fraction"] < 0.99
              and row["meets_startup_no_regression"] for row in selected), selected)

    field = load_csv(OUT / "precooling_temperature_field.csv")
    field_long = load_csv(OUT / "precooling_temperature_field_long.csv")
    first = field[0]
    last = field[-1]
    node_names = ["left_plate", "cell_1", "cell_2", "cell_3",
                  "cell_4", "cell_5", "right_plate"]
    symmetry_error = max(
        abs(float(row[a]) - float(row[b]))
        for row in field
        for a, b in (("left_plate", "right_plate"),
                     ("cell_1", "cell_5"), ("cell_2", "cell_4"))
    )
    check("precooling_field_complete",
          len(field) == 101 and len(field_long) == 707,
          {"wide_rows": len(field), "long_rows": len(field_long)})
    check("precooling_initial_25C",
          max(abs(float(first[name]) - 25.0) for name in node_names) < 1e-10,
          {name: first[name] for name in node_names})
    check("precooling_symmetry", symmetry_error < 1e-8,
          f"max_abs_error={symmetry_error}")
    check("precooling_approaches_ambient",
          all(-30.0 < float(last[name]) < -28.0 for name in node_names),
          {name: last[name] for name in node_names})

    scan = load_csv(OUT / "constant_strategy_cooling_scan.csv")
    times = [float(row["cooling_minutes"]) for row in scan]
    min_t = [float(row["initial_min_cell_temperature_c"]) for row in scan]
    check("cooling_scan_range", times[0] == 10.0 and times[-1] == 100.0,
          [times[0], times[-1], len(times)])
    check("cooling_scan_uniform_grid",
          len(times) >= 10 and np.ptp(np.diff(times)) < 1e-10, times)
    check("cooling_scan_all_success",
          all(row["success"].lower() == "true" for row in scan),
          [row["reason"] for row in scan])
    check("cooling_scan_voltage_constraint",
          min(float(row["min_voltage_v"]) for row in scan) >= 0.30,
          min(float(row["min_voltage_v"]) for row in scan))
    check("cooling_temperature_monotone",
          all(b < a for a, b in zip(min_t, min_t[1:])),
          [min_t[0], min_t[-1]])

    report = {
        "status": "PASS" if all(x["passed"] for x in checks) else "FAIL",
        "checks": checks,
        "trajectory_checks": trajectory_checks,
    }
    (OUT / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic constrained searches for the two Problem 3 strategies.

Pure preheating is a seven-node linear thermal problem. For every candidate
duration, all five heater powers are optimized independently by linear
programming; a scalar search then minimizes total auxiliary energy.

Cooperative startup uses the full five-cell multiphase simulator and searches
q1, ..., q5 and the heater duration as six independent variables.  A symmetric
vector may be used as an initial point, but no equality is imposed between
different heaters.  The only objective is auxiliary-heater energy; startup
time, voltage and ice are reported or constrained, never added as objectives.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog, minimize_scalar

from aux_model import AuxStack
from stack_model import AREA_CM2, ENDPLATE_CAP_J_K, ENDPLATE_CONV_W_K, T_FREEZE

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
TEMP_MARGIN_C = 0.01
VOLTAGE_MARGIN_V = 0.3005
ICE_LIMIT = 0.99

class NoLoadThermal:
    """Exact seven-node mean-temperature model for zero-current preheating."""

    def __init__(self):
        self.stack = AuxStack(symmetric=False, mesh_factor=1)
        c = self.stack.cell_capacity_j_k
        self.caps = np.array([ENDPLATE_CAP_J_K] + [c] * 5 + [ENDPLATE_CAP_J_K])
        self.k = np.zeros((7, 7))
        for edge in range(6):
            g = self.stack.plate_g_w_k if edge in (0, 5) else self.stack.link_g_w_k
            self.k[edge, edge] += g
            self.k[edge + 1, edge + 1] += g
            self.k[edge, edge + 1] -= g
            self.k[edge + 1, edge] -= g
        for edge in (0, 6):
            self.k[edge, edge] += ENDPLATE_CONV_W_K
        self.a = -self.k / self.caps[:, None]
        self.power_map = np.zeros((7, 5))
        self.power_map[1:6, :] = AREA_CM2 * np.eye(5)
        self.zeros = np.zeros(self.stack.cell.n)
        self.lam = self.stack.params.membrane_lambda

    def response_matrix(self, heat_s):
        steady_per_power = np.linalg.solve(self.k, self.power_map)
        transition = (np.eye(7) - expm(self.a * heat_s)) @ steady_per_power
        return transition[1:6, :]

    def evaluate(self, powers, heat_s):
        powers = np.asarray(powers, float)
        temp_c = -30.0 + self.response_matrix(heat_s) @ powers
        voltage = []
        for k in range(5):
            state = np.full(self.stack.cell.n, temp_c[k] + T_FREEZE)
            voltage.append(self.stack._voltage(
                state, self.zeros, self.zeros, self.lam, 0.0,
                self.stack.factors[k])[0])
        return temp_c, np.asarray(voltage)

    def optimize_powers(self, heat_s, target_c):
        response = self.response_matrix(heat_s)
        required_rise = 30.0 + target_c
        result = linprog(
            np.ones(5), A_ub=-response, b_ub=-np.full(5, required_rise),
            bounds=[(0.0, 1.0)] * 5, method="highs"
        )
        if not result.success:
            return None
        # Keep the five LP decisions exactly as optimized.  Mirror symmetry
        # may emerge from the model, but it is never imposed or postprocessed.
        powers = result.x.copy()
        temp_c, voltage = self.evaluate(powers, heat_s)
        if np.min(temp_c) < target_c - 1e-7:
            return None
        energy = AREA_CM2 * heat_s * float(np.sum(powers))
        return {
            "powers_w_cm2": powers.tolist(),
            "heat_s": float(heat_s),
            "energy_each_j": (AREA_CM2 * heat_s * powers).tolist(),
            "energy_j": float(energy),
            "temperature_c": temp_c.tolist(),
            "voltage_at_loading_start_v": voltage.tolist(),
            "min_temperature_c": float(np.min(temp_c)),
            "min_voltage_v": float(np.min(voltage)),
        }


def search_preheat(args):
    thermal = NoLoadThermal()
    durations = np.arange(args.min_heat_s, args.max_heat_s + 0.5 * args.grid_step_s,
                          args.grid_step_s)
    grid = []
    for heat_s in durations:
        item = thermal.optimize_powers(float(heat_s), args.preheat_target_c)
        if item is not None:
            grid.append(item)
    if not grid:
        raise RuntimeError("No feasible pure-preheating policy in the duration bounds")
    coarse = min(grid, key=lambda item: item["energy_j"])

    def objective(heat_s):
        item = thermal.optimize_powers(float(heat_s), args.preheat_target_c)
        return 1e12 if item is None else item["energy_j"]

    lo = max(args.min_heat_s, coarse["heat_s"] - args.grid_step_s)
    hi = min(args.max_heat_s, coarse["heat_s"] + args.grid_step_s)
    scalar = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"xatol": args.duration_tol_s})
    best = thermal.optimize_powers(float(scalar.x), args.preheat_target_c)
    if best is None or coarse["energy_j"] < best["energy_j"]:
        best = coarse
    validation_model = AuxStack(symmetric=False, mesh_factor=args.validation_mesh)
    validation = validation_model.simulate(
        "preheat", best["powers_w_cm2"], best["heat_s"],
        dt_max=args.validation_dt, time_cap=best["heat_s"] + args.validation_dt,
        capture_interval=1000, temp_margin=TEMP_MARGIN_C)
    best["full_model_validation"] = {
        "success": validation.success,
        "reason": validation.reason,
        "startup_s": validation.startup_s,
        "energy_j": validation.heat_aux_j,
        "min_voltage_v": validation.min_voltage_v,
        "max_ice_fraction": validation.max_ice_fraction,
        "min_final_temperature_c": float(np.min(validation.final_temperature_c)),
        "mesh_cells_per_cell": int(validation_model.cell.n),
        "dt_max_s": args.validation_dt,
    }
    if not validation.success:
        failure = {
            "status": "failed_validation",
            "reason": validation.reason,
            "preheat_design_target_c": args.preheat_target_c,
            "success_temperature_margin_c": TEMP_MARGIN_C,
            "candidate": best,
            "validation_terminal_temperature_c": validation.final_temperature_c.tolist(),
            "validation_min_temperature_before_load_c": validation.min_temperature_before_load_c,
        }
        OUT.mkdir(exist_ok=True)
        (OUT / "preheat_search_failed.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "The LP optimum failed independent full-model validation; "
            f"reason={validation.reason}, "
            f"min_final_T={float(np.min(validation.final_temperature_c)):.8f} C. "
            "Details were written to results/preheat_search_failed.json"
        )
    payload = {
        "status": "completed",
        "decision_variables": ["q1", "q2", "q3", "q4", "q5", "heat_s"],
        "objective": "minimize 25*heat_s*sum(qk)",
        "constraints": {
            "power_bounds_w_cm2": [0.0, 1.0],
            "temperature_c": (
                f"LP design target >= {args.preheat_target_c} C; "
                f"independent success check > {TEMP_MARGIN_C} C"
            ),
            "ice_fraction": f"< {ICE_LIMIT}",
            "voltage_v": f">= {VOLTAGE_MARGIN_V} V numerical margin",
        },
        "current_law": "j=0 during preheat; then j(tau)=min(0.005*tau,0.3) A/cm2",
        "method": "duration grid + bounded scalar refinement; five-power LP at each duration",
        "grid_step_s": args.grid_step_s,
        "duration_tolerance_s": args.duration_tol_s,
        "preheat_design_target_c": args.preheat_target_c,
        "best": best,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "preheat_search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def co_key(x):
    return ",".join(f"{float(value):.9f}" for value in x)


def run_co_candidate(spec):
    x, mesh_factor, dt_max, time_cap = spec
    x = np.asarray(x, float)
    if x.shape != (6,):
        raise ValueError("Cooperative search requires q1,...,q5 and heat_s")
    powers = x[:5].copy()
    heat_s = float(x[5])
    nominal_energy = AREA_CM2 * heat_s * float(np.sum(powers))
    result = AuxStack(symmetric=False, mesh_factor=mesh_factor).simulate(
        "cooperative", powers, heat_s, dt_max=dt_max, time_cap=time_cap,
        capture_interval=1000, temp_margin=TEMP_MARGIN_C)
    return {
        "parameters": [float(value) for value in x],
        "powers_w_cm2": powers.tolist(),
        "heat_s": heat_s,
        "energy_each_j": (AREA_CM2 * heat_s * powers).tolist(),
        "energy_j": float(nominal_energy),
        "actual_energy_until_stop_j": float(result.heat_aux_j),
        "success": bool(result.success),
        "reason": result.reason,
        "startup_s": result.startup_s,
        "min_voltage_v": float(result.min_voltage_v),
        "min_voltage_by_cell_v": result.min_voltage_by_cell_v.tolist(),
        "max_ice_fraction": float(result.max_ice_fraction),
        "peak_ice_fraction_by_cell": result.peak_ice_fraction_by_cell.tolist(),
        "max_pore_occupancy": float(result.max_pore_occupancy),
        "min_final_temperature_c": float(np.min(result.final_temperature_c)),
    }


def co_feasible(row):
    return (row["success"] and row["min_voltage_v"] >= VOLTAGE_MARGIN_V
            and row["max_ice_fraction"] < ICE_LIMIT
            and row["min_final_temperature_c"] >= TEMP_MARGIN_C - 1e-8)


def decreasing_neighbors(x, power_step, time_step):
    """One-step lower neighbors for all six independent variables.

    These neighbors provide the final numerical optimality certificate: at
    the selected point, lowering any one q_k or the heating duration by the
    declared resolution must violate at least one Q2 startup constraint.
    """
    x = np.asarray(x, float)
    if x.shape != (6,):
        raise ValueError("Expected five powers plus one heating duration")
    steps = np.array([power_step] * 5 + [time_step], float)
    candidates = []
    for axis in range(6):
        trial = x.copy()
        trial[axis] -= steps[axis]
        if np.all((trial[:5] >= 0) & (trial[:5] <= 1)) and trial[5] > 0:
            candidates.append(trial)
    return candidates


def evaluate_batch(candidates, cache, args):
    pending = [x for x in candidates if co_key(x) not in cache]
    if pending:
        specs = [(x.tolist(), args.mesh_factor, args.dt, args.time_cap)
                 for x in pending]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_co_candidate, spec): np.asarray(spec[0])
                       for spec in specs}
            for future in as_completed(futures):
                row = future.result()
                cache[co_key(row["parameters"])] = row
                print(json.dumps(row, ensure_ascii=False), flush=True)
    return [cache[co_key(x)] for x in candidates]


def search_cooperative(args):
    OUT.mkdir(exist_ok=True)
    work_path = OUT / "cooperative_search_work.json"
    settings = {
        "mesh_factor": args.mesh_factor, "dt_max_s": args.dt,
        "time_cap_s": args.time_cap, "temperature_margin_c": TEMP_MARGIN_C,
        "voltage_margin_v": VOLTAGE_MARGIN_V,
        "search_dimension": 6, "independent_heater_powers": True,
        "neighborhood": "independent one-step downward neighbors of q1,...,q5,heat_s",
    }
    cache = {}
    previous_best = None
    if args.resume and work_path.exists():
        previous = json.loads(work_path.read_text(encoding="utf-8"))
        if previous.get("settings") == settings:
            cache = {co_key(row["parameters"]): row
                     for row in previous.get("evaluations", [])}
            previous_best = previous.get("best")

    best_x = np.array(
        previous_best["parameters"] if previous_best else args.seed, float)
    seed_row = evaluate_batch([best_x], cache, args)[0]
    if not co_feasible(seed_row):
        raise RuntimeError("The cooperative seed is not feasible at the requested fidelity")
    best = seed_row
    work_path.write_text(json.dumps({
        "status": "running", "settings": settings,
        "best": best, "evaluations": list(cache.values())
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    stages = list(zip(args.power_steps, args.time_steps))
    for power_step, time_step in stages:
        for _ in range(args.max_rounds):
            candidates = decreasing_neighbors(best_x, power_step, time_step)
            rows = evaluate_batch(candidates, cache, args)
            work_path.write_text(json.dumps({
                "status": "running", "settings": settings,
                "best": best, "evaluations": list(cache.values())
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            feasible = [row for row in rows if co_feasible(row)]
            if not feasible:
                break
            candidate = min(feasible, key=lambda row: row["energy_j"])
            if candidate["energy_j"] >= best["energy_j"] - 1e-9:
                break
            best = candidate
            best_x = np.array(best["parameters"], float)
            work_path.write_text(json.dumps({
                "status": "running", "settings": settings,
                "best": best, "evaluations": list(cache.values())
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    lower = [row for row in cache.values()
             if not co_feasible(row) and row["energy_j"] < best["energy_j"]]
    lower_boundary = max(lower, key=lambda row: row["energy_j"]) if lower else None
    payload = {
        "status": "completed",
        "decision_variables": ["q1", "q2", "q3", "q4", "q5", "heat_s"],
        "independence": "q1,...,q5 are searched independently; symmetry is used only in the initial seed",
        "objective": "minimize 25*heat_s*sum(qk)",
        "constraints": {
            "power_bounds_w_cm2": [0.0, 1.0],
            "temperature_c": f"all five cells >= {TEMP_MARGIN_C} C numerical margin",
            "ice_fraction": f"maximum local value over all cells, mesh points, and 0<=t<=ts < {ICE_LIMIT}",
            "voltage_v": f"all cells and all times >= {VOLTAGE_MARGIN_V} V numerical margin",
        },
        "current_law": "j(t)=min(0.005*t,0.3) A/cm2",
        "method": (
            "single-objective six-variable energy boundary refinement on the "
            "full five-cell multiphase simulation"
        ),
        "settings": settings,
        "power_steps": args.power_steps,
        "time_steps": args.time_steps,
        "best": best,
        "lower_boundary": lower_boundary,
        "numerical_optimality": {
            "meaning": (
                "minimum auxiliary energy at the declared search resolution; "
                "each one-step lower decision neighbor violates a Q2 constraint"
            ),
            "evaluation_count": len(cache),
        },
    }
    (OUT / "cooperative_search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    work_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "lower_boundary": lower_boundary},
                     ensure_ascii=False, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    pre = sub.add_parser("preheat")
    pre.add_argument("--min-heat-s", type=float, default=20.0)
    pre.add_argument("--max-heat-s", type=float, default=300.0)
    pre.add_argument("--grid-step-s", type=float, default=0.25)
    pre.add_argument("--duration-tol-s", type=float, default=1e-3)
    pre.add_argument("--preheat-target-c", type=float, default=0.05)
    pre.add_argument("--validation-mesh", type=int, default=3)
    pre.add_argument("--validation-dt", type=float, default=0.05)

    co = sub.add_parser("cooperative")
    co.add_argument("--seed", nargs=6, type=float,
                    default=[0.50, 0.32, 0.25, 0.32, 0.50, 4.10],
                    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "HEAT_S"))
    co.add_argument("--mesh-factor", type=int, default=3)
    co.add_argument("--dt", type=float, default=0.025)
    co.add_argument("--time-cap", type=float, default=310.0)
    co.add_argument("--workers", type=int, default=4)
    co.add_argument("--power-steps", nargs="+", type=float,
                    default=[0.005, 0.001, 0.0001, 0.00001])
    co.add_argument("--time-steps", nargs="+", type=float,
                    default=[0.025, 0.005, 0.0001, 0.00001])
    co.add_argument("--max-rounds", type=int, default=20)
    co.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.phase == "preheat":
        search_preheat(arguments)
    else:
        if len(arguments.power_steps) != len(arguments.time_steps):
            raise SystemExit("--power-steps and --time-steps must have equal lengths")
        search_cooperative(arguments)

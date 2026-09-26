"""Dynamic auxiliary-heating model for Problem 4.

The plant reuses the calibrated one-dimensional cell model and five-cell
thermal network from Problems 1--3.  The additions in this module are:

* exact seven-node pre-cooling from 25 degC in a -30 degC environment;
* arbitrary, non-uniform initial cell/end-plate temperatures;
* independently commanded, time-varying heaters; and
* an event-triggered feedback controller using only T, V, dT/dt and dV/dt.

The feedback policy has four explicit modes: boost, hold, reduce and off.
Its feed-forward scale is screened by ``optimize.py`` on the five metrics
named in the statement; voltage and temperature-rate feedback may override
the scheduled pulse when safety is threatened.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "问题3"))
sys.path.insert(0, str(ROOT / "问题2"))
from aux_model import AuxStack, prescribed_current  # noqa: E402
from stack_model import (  # noqa: E402
    AREA_CM2,
    AREA_M2,
    ENDPLATE_CAP_J_K,
    ENDPLATE_CONV_W_K,
    ETH,
    F,
    LATENT_FREEZE,
    MW,
    Parameters,
    T_FREEZE,
    VSAFE_V,
)


AMBIENT_C = -30.0
CONTROL_INTERVAL_S = 0.25
TEMP_MARGIN_C = 0.01
VOLTAGE_SEARCH_MARGIN_V = 0.3005
ICE_LIMIT = 0.99
Q1_METRICS = ROOT / "result" / "metrics.json"


def q4_calibrated_parameters() -> Parameters:
    """Load the reproducible Q1 calibration used by the Q4 plant.

    ``result/metrics.json`` is regenerated directly from Attachments 1 and 2
    by ``问题1/run.py`` before the Problem 4 calculation.  Naming it here
    keeps the calibrated input explicit and lets the run manifest hash it.
    """
    payload = json.loads(Q1_METRICS.read_text(encoding="utf-8"))
    values = payload["parameters"]
    allowed = Parameters.__dataclass_fields__
    return Parameters(**{key: value for key, value in values.items()
                         if key in allowed})


class PrecoolingModel:
    """Exact mean-temperature solution for the seven-node stack network."""

    node_names = ("left_plate", "cell_1", "cell_2", "cell_3",
                  "cell_4", "cell_5", "right_plate")

    def __init__(self):
        stack = AuxStack(symmetric=False, mesh_factor=1,
                         params=q4_calibrated_parameters())
        cell_cap = stack.cell_capacity_j_k
        self.caps = np.array([ENDPLATE_CAP_J_K] + [cell_cap] * 5
                             + [ENDPLATE_CAP_J_K])
        conductance = np.zeros((7, 7))
        for edge in range(6):
            g = stack.plate_g_w_k if edge in (0, 5) else stack.link_g_w_k
            conductance[edge, edge] += g
            conductance[edge + 1, edge + 1] += g
            conductance[edge, edge + 1] -= g
            conductance[edge + 1, edge] -= g
        for edge in (0, 6):
            conductance[edge, edge] += ENDPLATE_CONV_W_K
        self.system = -conductance / self.caps[:, None]
        cell_pitch_m = stack.cell.length + 0.004
        end_to_cell_m = 0.005 + 0.5 * cell_pitch_m
        self.positions_m = np.array(
            [0.0]
            + [end_to_cell_m + k * cell_pitch_m for k in range(5)]
            + [2.0 * end_to_cell_m + 4.0 * cell_pitch_m]
        )

    def field(self, cooling_minutes: float, initial_c: float = 25.0,
              ambient_c: float = AMBIENT_C) -> np.ndarray:
        seconds = 60.0 * float(cooling_minutes)
        relative = np.full(7, float(initial_c) - float(ambient_c))
        return float(ambient_c) + expm(self.system * seconds) @ relative

    def history(self, end_minutes: float = 100.0, step_minutes: float = 1.0,
                initial_c: float = 25.0,
                ambient_c: float = AMBIENT_C) -> tuple[np.ndarray, np.ndarray]:
        minutes = np.arange(0.0, end_minutes + 0.5 * step_minutes,
                            step_minutes)
        fields = np.vstack([self.field(x, initial_c, ambient_c)
                            for x in minutes])
        return minutes, fields


def initial_state(case_id: int, precooling: PrecoolingModel | None = None):
    """Return (cell temperatures, plate temperatures, cooling minutes)."""
    if case_id == 1:
        return np.full(5, -30.0), np.full(2, -30.0), None
    duration = {2: 20.0, 3: 40.0}.get(int(case_id))
    if duration is None:
        raise ValueError(f"Unknown case: {case_id}")
    model = precooling or PrecoolingModel()
    field = model.field(duration)
    return field[1:6].copy(), field[[0, 6]].copy(), duration


class ConstantPolicy:
    """Problem 3 constant-power cooperative strategy."""

    def __init__(self, powers_w_cm2, duration_s):
        self.powers = np.asarray(powers_w_cm2, float)
        self.duration_s = float(duration_s)

    def reset(self):
        pass

    def event_times(self):
        return (self.duration_s,)

    def command(self, t, temperature_c, voltage_v, dtemp_dt, dvolt_dt,
                current_a_cm2, delivered_energy_j):
        del temperature_c, voltage_v, dtemp_dt, dvolt_dt
        del current_a_cm2, delivered_energy_j
        if t < self.duration_s - 1e-12:
            return self.powers.copy(), np.array(["hold"] * 5, object)
        return np.zeros(5), np.array(["off"] * 5, object)


class FeedbackPolicy:
    """Independent four-mode feedback built around the Q3 constant policy.

    ``aggressiveness=1`` reproduces the Q3 power vector and heating duration
    before feedback corrections.  Increasing it raises both the feed-forward
    powers (capped at 1 W/cm2) and their duration.  The offline search therefore
    spans the energy--speed trade-off while the online law still uses only the
    measured T, V and their rates; simulated ice is never a controller input.
    """

    def __init__(self, reference_powers_w_cm2, reference_duration_s,
                 aggressiveness, *, v_predict_v=0.305,
                 v_hold_v=0.320, prediction_horizon_s=1.5,
                 dvdt_risk_v_s=-0.006, dtemp_hold_c_s=0.015,
                 temperature_lag_c=0.15, boost_increment_w_cm2=0.20,
                 hold_power_w_cm2=0.20, reduce_power_w_cm2=0.10):
        reference = np.asarray(reference_powers_w_cm2, float)
        if reference.shape != (5,) or np.any(reference < 0):
            raise ValueError("reference_powers_w_cm2 must contain five nonnegative values")
        if not np.isfinite(aggressiveness) or aggressiveness < 0:
            raise ValueError("aggressiveness must be nonnegative")
        self.aggressiveness = float(aggressiveness)
        self.feedforward_power = np.clip(
            reference * self.aggressiveness, 0.0, 1.0)
        self.feedforward_duration_s = (
            float(reference_duration_s) * self.aggressiveness)
        self.v_predict_v = float(v_predict_v)
        self.v_hold_v = float(v_hold_v)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.dvdt_risk_v_s = float(dvdt_risk_v_s)
        self.dtemp_hold_c_s = float(dtemp_hold_c_s)
        self.temperature_lag_c = float(temperature_lag_c)
        self.boost_increment_w_cm2 = float(boost_increment_w_cm2)
        self.hold_power_w_cm2 = float(hold_power_w_cm2)
        self.reduce_power_w_cm2 = float(reduce_power_w_cm2)
        self.previous_power = np.zeros(5)

    def reset(self):
        self.previous_power[:] = 0.0

    def command(self, t, temperature_c, voltage_v, dtemp_dt, dvolt_dt,
                current_a_cm2, delivered_energy_j):
        del delivered_energy_j
        temperature_c = np.asarray(temperature_c, float)
        voltage_v = np.asarray(voltage_v, float)
        dtemp_dt = np.asarray(dtemp_dt, float)
        dvolt_dt = np.asarray(dvolt_dt, float)
        power = np.zeros(5)
        modes = np.empty(5, object)

        all_ready = bool(np.min(temperature_c) > TEMP_MARGIN_C)
        stack_mean = float(np.mean(temperature_c))
        pulse_open = t < self.feedforward_duration_s - 1e-12
        for k in range(5):
            v_pred = voltage_v[k] + self.prediction_horizon_s * dvolt_dt[k]
            base = self.feedforward_power[k] if pulse_open else 0.0
            near_success = (temperature_c[k] > -0.20 and dtemp_dt[k] > 0
                            and voltage_v[k] > self.v_hold_v)
            temperature_lag = temperature_c[k] < stack_mean - self.temperature_lag_c
            warming_insufficient = (
                temperature_c[k] < 0.0
                and dtemp_dt[k] < self.dtemp_hold_c_s
                and current_a_cm2 > 0.02
            )
            ice_risk = (temperature_c[k] < 0.0
                        and voltage_v[k] < 0.36
                        and (dvolt_dt[k] < self.dvdt_risk_v_s
                             or v_pred < self.v_hold_v))

            if all_ready:
                power[k], modes[k] = 0.0, "off"
            elif v_pred < self.v_predict_v and voltage_v[k] < 0.36:
                power[k], modes[k] = 1.0, "boost"
            elif ice_risk and current_a_cm2 > 0.02:
                power[k] = max(base, self.hold_power_w_cm2)
                modes[k] = "boost" if power[k] > base + 1e-12 else "hold"
            elif near_success:
                power[k], modes[k] = 0.0, "off"
            elif pulse_open and (temperature_lag or warming_insufficient):
                power[k] = min(1.0, base + self.boost_increment_w_cm2)
                modes[k] = "boost"
            elif pulse_open:
                power[k], modes[k] = base, "hold"
            elif self.previous_power[k] > self.reduce_power_w_cm2 + 1e-12:
                power[k], modes[k] = self.reduce_power_w_cm2, "reduce"
            else:
                power[k], modes[k] = 0.0, "off"

        self.previous_power = np.clip(power, 0.0, 1.0)
        return self.previous_power.copy(), modes


@dataclass
class DynamicResult:
    strategy: str
    success: bool
    reason: str
    startup_s: float | None
    initial_cell_temperature_c: np.ndarray
    initial_plate_temperature_c: np.ndarray
    time_s: np.ndarray
    current_a_cm2: np.ndarray
    charge_c_cm2: np.ndarray
    temperature_c: np.ndarray
    plate_temperature_c: np.ndarray
    voltage_v: np.ndarray
    ice_fraction: np.ndarray
    heater_power_w_cm2: np.ndarray
    controller_mode: np.ndarray
    energy_each_j: np.ndarray
    auxiliary_energy_j: float
    min_voltage_v: float
    min_voltage_by_cell_v: np.ndarray
    max_ice_fraction: float
    peak_ice_fraction_by_cell: np.ndarray
    max_temperature_spread_c: float
    max_pore_occupancy: float
    min_gas_mol_m3: float
    max_water_balance_kg_m2: float
    energy_balance_j: float
    heat_electrochemical_j: float
    heat_latent_j: float
    heat_lost_j: float
    heat_stored_j: float


class DynamicAuxStack(AuxStack):
    """Q3 plant with non-uniform initialization and dynamic heaters."""

    def __init__(self, *args, params=None, **kwargs):
        if params is None:
            params = q4_calibrated_parameters()
        super().__init__(*args, params=params, **kwargs)

    def simulate_policy(self, policy, *, strategy="dynamic",
                        initial_cell_c=None, initial_plate_c=None,
                        ambient_c=AMBIENT_C, dt_max=0.1, time_cap=360.0,
                        control_interval=CONTROL_INTERVAL_S,
                        capture_interval=0.5, temp_margin=TEMP_MARGIN_C):
        if self.symmetric:
            raise ValueError("Problem 4 uses five independent cell states")
        if dt_max <= 0 or control_interval <= 0 or time_cap <= 0:
            raise ValueError("Invalid time controls")
        initial_cell = np.asarray(initial_cell_c if initial_cell_c is not None
                                  else np.full(5, -30.0), float)
        initial_plate = np.asarray(initial_plate_c if initial_plate_c is not None
                                   else np.full(2, -30.0), float)
        if initial_cell.shape != (5,) or initial_plate.shape != (2,):
            raise ValueError("Initial state must contain five cells and two plates")

        m, params = self.cell, self.params
        temp = np.repeat((initial_cell + T_FREEZE)[:, None], m.n, axis=1)
        temp_initial = temp.copy()
        plate = initial_plate + T_FREEZE
        plate_initial = plate.copy()
        water = np.zeros((5, m.n))
        ice = np.zeros((5, m.n))
        lam = np.full(5, params.membrane_lambda)
        escaped = np.zeros(5)
        produced = np.zeros(5)
        sorbed = np.zeros(5)
        water_per_lambda = 2150.0 * MW * 12e-6

        t = charge = q_elec = q_latent = q_lost = 0.0
        energy_each = np.zeros(5)
        min_v = float("inf")
        min_v_by_cell = np.full(5, np.inf)
        peak_ice_by_cell = np.zeros(5)
        max_ice = max_occ = max_water_error = max_spread = 0.0
        min_gas = float("inf")
        success, reason, startup = False, "time_cap", None
        next_control = 0.0
        last_control_t = None
        last_control_temp = None
        last_control_voltage = None
        current_power = np.zeros(5)
        current_modes = np.array(["off"] * 5, object)
        policy.reset()

        times, currents, charges = [], [], []
        temperatures, plates, voltages, ice_series = [], [], [], []
        powers, modes = [], []
        last_capture = -float("inf")

        def state(current):
            stats = [self._voltage(temp[z], water[z], ice[z], lam[z],
                                   current * 1e4, self.factors[z])
                     for z in range(5)]
            tt = temp @ m.dx / m.length - T_FREEZE
            vv = np.array([x[0] for x in stats])
            ii = np.array([x[2] for x in stats])
            gg = np.array([x[1] for x in stats])
            oo = np.array([x[3] for x in stats])
            return tt, vv, ii, gg, oo

        def track(current, force=False):
            nonlocal min_v, min_gas, max_ice, max_occ, max_spread
            tt, vv, ii, gg, oo = state(current)
            min_v = min(min_v, float(np.min(vv)))
            np.minimum(min_v_by_cell, vv, out=min_v_by_cell)
            max_ice = max(max_ice, float(np.max(ii)))
            np.maximum(peak_ice_by_cell, ii, out=peak_ice_by_cell)
            min_gas = min(min_gas, float(np.min(gg)))
            max_occ = max(max_occ, float(np.max(oo)))
            max_spread = max(max_spread, float(np.max(tt) - np.min(tt)))
            if force or t - last_capture >= capture_interval - 1e-9:
                times.append(t)
                currents.append(current)
                charges.append(charge)
                temperatures.append(tt.copy())
                plates.append((plate - T_FREEZE).copy())
                voltages.append(vv.copy())
                ice_series.append(ii.copy())
                powers.append(current_power.copy())
                modes.append(current_modes.copy())
                return tt, vv, ii, True
            return tt, vv, ii, False

        current = prescribed_current(0.0, "cooperative", 0.0)
        means, vv, ii, captured = track(current, True)
        if captured:
            last_capture = t
        if np.min(means) > temp_margin and np.max(ii) < ICE_LIMIT:
            success, reason, startup = True, "startup", 0.0

        while not success and t < time_cap - 1e-11:
            current = prescribed_current(t, "cooperative", 0.0)
            means, vv, ii, _ = track(current, False)
            if min_v < VSAFE_V - 1e-9:
                reason = "voltage_below_0.30V"
                break
            if max_ice >= ICE_LIMIT:
                reason = "ice_limit"
                break

            if t >= next_control - 1e-10:
                if last_control_t is None or t <= last_control_t + 1e-12:
                    dtemp = np.zeros(5)
                    dvolt = np.zeros(5)
                else:
                    elapsed = t - last_control_t
                    dtemp = (means - last_control_temp) / elapsed
                    dvolt = (vv - last_control_voltage) / elapsed
                current_power, current_modes = policy.command(
                    t, means, vv, dtemp, dvolt, current,
                    energy_each.copy())
                current_power = np.clip(np.asarray(current_power, float), 0.0, 1.0)
                current_modes = np.asarray(current_modes, object)
                last_control_t = t
                last_control_temp = means.copy()
                last_control_voltage = vv.copy()
                next_control = t + control_interval
                if hasattr(policy, "event_times"):
                    future = [point for point in policy.event_times()
                              if point > t + 1e-10]
                    if future:
                        next_control = min(next_control, min(future))

            step = min(dt_max, time_cap - t, next_control - t)
            # Snap to the current-ramp breakpoint within floating tolerance.
            # Without this guard, a residual of order 1e-12 s at t=60 can
            # repeatedly generate a zero-length step on 0.05 s grids.
            if abs(t - 60.0) <= 1e-9:
                t = 60.0
                step = min(dt_max, time_cap - t, next_control - t)
            elif t < 60.0 < t + step:
                step = 60.0 - t
            if step < 1e-11:
                if 0.0 < 60.0 - t <= 1e-9:
                    t = 60.0
                next_control = t
                continue
            jmid = prescribed_current(t + 0.5 * step, "cooperative", 0.0)
            current_am2 = 1e4 * jmid
            intrinsic = np.empty_like(temp)
            for z in range(5):
                production = current_am2 * MW / (2 * F)
                available = max(14.0 - lam[z], 0.0)
                uptake = min(
                    production * params.uptake_fraction * available
                    / max(14.0 - params.membrane_lambda, 1e-6),
                    available * water_per_lambda / step,
                )
                new_water, new_ice, outflow, ice_rate = m._water_step(
                    temp[z], water[z], ice[z], step, current_am2, uptake, params)
                new_lam = lam[z] + uptake * step / water_per_lambda
                v_heat, _, _, _ = self._voltage(
                    temp[z], new_water, new_ice, new_lam,
                    current_am2, self.factors[z])
                intrinsic[z] = self._intrinsic_heat_step(
                    temp[z], new_water, new_ice, ice_rate, step,
                    current_am2, v_heat)
                intrinsic[z] += (AREA_CM2 * current_power[z] * step
                                 / self.cell_capacity_j_k)
                q_elec += AREA_M2 * step * current_am2 * (ETH - v_heat)
                q_latent += (AREA_M2 * step * LATENT_FREEZE
                             * float(np.dot(ice_rate, m.dx)))
                escaped[z] += outflow * step
                produced[z] += production * step
                sorbed[z] += uptake * step
                water[z], ice[z], lam[z] = new_water, new_ice, new_lam

            energy_each += AREA_CM2 * current_power * step
            temp, plate, lost = self._heat_network(
                intrinsic, plate, ambient_c + T_FREEZE, step)
            q_lost += lost
            t += step
            charge += jmid * step
            errors = (produced
                      - np.array([float(np.dot(w, m.dx)) for w in water])
                      - escaped - sorbed)
            max_water_error = max(max_water_error,
                                  float(np.max(np.abs(errors))))

            current = prescribed_current(t, "cooperative", 0.0)
            means, vv, ii, captured = track(current, False)
            if captured:
                last_capture = t
            if min_v < VSAFE_V - 1e-9:
                reason = "voltage_below_0.30V"
                break
            if max_ice >= ICE_LIMIT:
                reason = "ice_limit"
                break
            if (np.min(means) > temp_margin and np.max(ii) < ICE_LIMIT
                    and min_v >= VSAFE_V - 1e-9):
                success, reason, startup = True, "startup", t

        if success:
            current_power = np.zeros(5)
            current_modes = np.array(["off"] * 5, object)
            if times and abs(times[-1] - t) <= 1e-8:
                powers[-1] = current_power.copy()
                modes[-1] = current_modes.copy()

        current = prescribed_current(t, "cooperative", 0.0)
        if not times or abs(times[-1] - t) > 1e-8:
            _, _, _, captured = track(current, True)
            if captured:
                last_capture = t

        storage = (
            AREA_M2 * float(np.sum(
                (temp - temp_initial) * self.capacity_areal * m.dx))
            + ENDPLATE_CAP_J_K * float(np.sum(plate - plate_initial))
        )
        aux_energy = float(np.sum(energy_each))
        balance = q_elec + q_latent + aux_energy - q_lost - storage
        return DynamicResult(
            strategy=strategy,
            success=success,
            reason=reason,
            startup_s=startup,
            initial_cell_temperature_c=initial_cell.copy(),
            initial_plate_temperature_c=initial_plate.copy(),
            time_s=np.asarray(times),
            current_a_cm2=np.asarray(currents),
            charge_c_cm2=np.asarray(charges),
            temperature_c=np.asarray(temperatures),
            plate_temperature_c=np.asarray(plates),
            voltage_v=np.asarray(voltages),
            ice_fraction=np.asarray(ice_series),
            heater_power_w_cm2=np.asarray(powers),
            controller_mode=np.asarray(modes, object),
            energy_each_j=energy_each.copy(),
            auxiliary_energy_j=aux_energy,
            min_voltage_v=min_v,
            min_voltage_by_cell_v=min_v_by_cell.copy(),
            max_ice_fraction=max_ice,
            peak_ice_fraction_by_cell=peak_ice_by_cell.copy(),
            max_temperature_spread_c=max_spread,
            max_pore_occupancy=max_occ,
            min_gas_mol_m3=min_gas,
            max_water_balance_kg_m2=max_water_error,
            energy_balance_j=balance,
            heat_electrochemical_j=q_elec,
            heat_latent_j=q_latent,
            heat_lost_j=q_lost,
            heat_stored_j=storage,
        )


def symmetric_targets(values3) -> np.ndarray:
    end, inner, middle = np.asarray(values3, float)
    return np.array([end, inner, middle, inner, end], float)


def result_record(result: DynamicResult, *, case_id: int,
                  cooling_minutes, strategy_label: str,
                  energy_targets_j=None) -> dict:
    return {
        "case": int(case_id),
        "cooling_minutes": cooling_minutes,
        "strategy": strategy_label,
        "success": bool(result.success),
        "reason": result.reason,
        "startup_s": result.startup_s,
        "charge_c_cm2": float(result.charge_c_cm2[-1]),
        "auxiliary_energy_j": result.auxiliary_energy_j,
        "energy_each_j": result.energy_each_j.tolist(),
        "energy_targets_j": (None if energy_targets_j is None
                             else np.asarray(energy_targets_j, float).tolist()),
        "max_temperature_spread_c": result.max_temperature_spread_c,
        "min_voltage_v": result.min_voltage_v,
        "min_voltage_by_cell_v": result.min_voltage_by_cell_v.tolist(),
        "max_ice_fraction": result.max_ice_fraction,
        "peak_ice_fraction_by_cell": result.peak_ice_fraction_by_cell.tolist(),
        "max_pore_occupancy": result.max_pore_occupancy,
        "initial_cell_temperature_c": result.initial_cell_temperature_c.tolist(),
        "initial_plate_temperature_c": result.initial_plate_temperature_c.tolist(),
        "max_water_balance_kg_m2": result.max_water_balance_kg_m2,
        "energy_balance_j": result.energy_balance_j,
        "heat_electrochemical_j": result.heat_electrochemical_j,
        "heat_latent_j": result.heat_latent_j,
        "heat_lost_j": result.heat_lost_j,
        "heat_stored_j": result.heat_stored_j,
    }

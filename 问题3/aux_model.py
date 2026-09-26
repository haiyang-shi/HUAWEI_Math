"""Five-cell auxiliary-heating extension of the calibrated Q1/Q2 stack.

The heater is represented by a mean-temperature increment in each cell.  This
is consistent with Q2's lumped bipolar-plate heat capacity and adds exactly
25*q_k W to the stack energy balance.  The optimization and final validation
use five independent cell states.  The optional symmetric mode is retained
only for diagnostics and is not used to constrain Problem 3's decisions.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "问题2"))
from stack_model import (AREA_M2, AREA_CM2, ENDPLATE_CAP_J_K,
                         ENDPLATE_CONV_W_K, ETH, LATENT_FREEZE,
                         MW, F, T_FREEZE, VSAFE_V, StackModel)


Q3_RAMP_RATE_A_CM2_S = 0.005
Q3_RAMP_DURATION_S = 60.0
Q3_HOLD_CURRENT_A_CM2 = 0.30


@dataclass
class AuxResult:
    mode: str
    powers_w_cm2: tuple
    requested_heat_s: float
    actual_heat_s: float
    success: bool
    reason: str
    startup_s: float | None
    time_s: np.ndarray
    current_a_cm2: np.ndarray
    charge_c_cm2: np.ndarray
    temperature_c: np.ndarray
    plate_temperature_c: np.ndarray
    voltage_v: np.ndarray
    ice_fraction: np.ndarray
    min_voltage_v: float
    min_voltage_by_cell_v: np.ndarray
    max_ice_fraction: float
    peak_ice_fraction_by_cell: np.ndarray
    max_pore_occupancy: float
    min_gas_mol_m3: float
    max_water_balance_kg_m2: float
    energy_balance_j: float
    heat_electrochemical_j: float
    heat_latent_j: float
    heat_aux_j: float
    heat_lost_j: float
    heat_stored_j: float
    final_temperature_c: np.ndarray
    final_voltage_v: np.ndarray
    final_ice_fraction: np.ndarray
    min_temperature_before_load_c: float | None

    @property
    def energy_each_j(self):
        return np.array(self.powers_w_cm2) * AREA_CM2 * self.actual_heat_s


def prescribed_current(t, mode, heat_s, postload="q3_ramp"):
    """Return the Q3 current density in A/cm2.

    Both strategies use the same prescribed Q3 loading law.  Cooperative
    startup applies it from t=0.  Pure preheating keeps j=0 during [0, heat_s)
    and applies the same law with local loading time tau=t-heat_s afterwards.
    """
    if postload != "q3_ramp":
        raise ValueError("Problem 3 uses only the prescribed 0-to-0.3 A/cm2 ramp")
    if mode == "cooperative":
        tau = max(t, 0.0)
    else:
        if t < heat_s - 1e-10:
            return 0.0
        tau = max(t - heat_s, 0.0)
    return min(Q3_RAMP_RATE_A_CM2_S * tau, Q3_HOLD_CURRENT_A_CM2)


def current_breakpoints(mode, heat_s, postload):
    if postload != "q3_ramp":
        raise ValueError("Problem 3 uses only the prescribed current ramp")
    if mode == "cooperative":
        return tuple(sorted(set((heat_s, Q3_RAMP_DURATION_S))))
    return (heat_s, heat_s + Q3_RAMP_DURATION_S)


class AuxStack(StackModel):
    def __init__(self, *, symmetric=False, **kwargs):
        super().__init__(**kwargs)
        self.symmetric = symmetric
        self.positions = (0, 1, 2) if symmetric else (0, 1, 2, 3, 4)
        self.weights = np.array((2.0, 2.0, 1.0) if symmetric else (1.0,) * 5)
        self.factors = np.array((self.end_concentration_factor, 1.0, 1.0)
                                if symmetric else
                                (self.end_concentration_factor, 1.0, 1.0,
                                 1.0, self.end_concentration_factor))
        self._heat_solver_cache = {}

    def _heat_network(self, profiles, old_plate, ambient_k, dt):
        m = self.cell
        means = np.array([float(np.dot(row, m.dx) / m.length) for row in profiles])
        cell_means = means[[0, 1, 2, 1, 0]] if self.symmetric else means
        before = np.r_[old_plate[0], cell_means, old_plate[1]]
        caps = np.array([ENDPLATE_CAP_J_K] + [self.cell_capacity_j_k] * 5
                        + [ENDPLATE_CAP_J_K])
        rhs = caps * before
        key = round(float(dt), 12)
        solver = self._heat_solver_cache.get(key)
        if solver is None:
            mat = np.diag(caps.copy())
            for edge in range(6):
                g = self.plate_g_w_k if edge in (0, 5) else self.link_g_w_k
                mat[edge, edge] += dt * g
                mat[edge + 1, edge + 1] += dt * g
                mat[edge, edge + 1] -= dt * g
                mat[edge + 1, edge] -= dt * g
            for edge in (0, 6):
                mat[edge, edge] += dt * ENDPLATE_CONV_W_K
            solver = np.linalg.inv(mat)
            self._heat_solver_cache[key] = solver
        for edge in (0, 6):
            rhs[edge] += dt * ENDPLATE_CONV_W_K * ambient_k
        after = solver @ rhs
        for z, pos in enumerate(self.positions):
            profiles[z] += after[pos + 1] - means[z]
        lost = dt * ENDPLATE_CONV_W_K * (after[0] + after[6] - 2 * ambient_k)
        return profiles, after[[0, 6]], float(lost)

    def simulate(self, mode, powers, heat_s, *, initial_c=-30.0,
                 ambient_c=-30.0, dt_max=0.5, time_cap=240.0,
                 capture_interval=1.0, postload="q3_ramp", charge_limit=None,
                 temp_margin=0.0, stop_on_success=True):
        if mode not in ("preheat", "cooperative"):
            raise ValueError(mode)
        powers = np.asarray(powers, float)
        if (powers.shape != (5,) or not np.all(np.isfinite(powers))
                or np.any(powers < 0) or np.any(powers > 1)
                or not np.isfinite(heat_s) or heat_s < 0):
            raise ValueError("Invalid heater policy")
        if dt_max <= 0 or time_cap <= 0:
            raise ValueError("Invalid time controls")
        if self.symmetric and not np.allclose(powers, powers[::-1], atol=1e-12):
            raise ValueError("Symmetric model requires symmetric heaters")
        local_power = powers[:3] if self.symmetric else powers
        m, params = self.cell, self.params
        nstate = len(self.positions)
        temp = np.full((nstate, m.n), initial_c + T_FREEZE)
        water = np.zeros((nstate, m.n))
        ice = np.zeros((nstate, m.n))
        lam = np.full(nstate, params.membrane_lambda)
        plate = np.full(2, initial_c + T_FREEZE)
        escaped = np.zeros(nstate)
        produced = np.zeros(nstate)
        sorbed = np.zeros(nstate)
        water_per_lambda = 2150.0 * MW * 12e-6
        t = charge = aux_energy = q_elec = q_latent = q_lost = 0.0
        min_v, min_gas = float("inf"), float("inf")
        min_v_by_cell = np.full(5, np.inf)
        peak_ice_by_cell = np.zeros(5)
        max_ice = max_occ = max_water_error = 0.0
        times, currents, charges, temperatures, plates, voltages, ice_series = ([] for _ in range(7))
        last_capture = -float("inf")
        success, reason, startup = False, "time_cap", None
        min_before_load = None

        def observe(current, force=False, track_voltage=True):
            nonlocal min_v, min_gas, max_ice, max_occ, last_capture
            stats = [self._voltage(temp[z], water[z], ice[z], lam[z],
                                   current * 1e4, self.factors[z])
                     for z in range(nstate)]
            vv = np.array([x[0] for x in stats])
            ii = np.array([x[2] for x in stats])
            cell_t = temp @ m.dx / m.length - T_FREEZE
            full_t = cell_t[[0, 1, 2, 1, 0]] if self.symmetric else cell_t
            full_v = vv[[0, 1, 2, 1, 0]] if self.symmetric else vv
            full_i = ii[[0, 1, 2, 1, 0]] if self.symmetric else ii
            if track_voltage:
                min_v = min(min_v, float(np.min(vv)))
                np.minimum(min_v_by_cell, full_v, out=min_v_by_cell)
            min_gas = min(min_gas, min(x[1] for x in stats))
            max_ice = max(max_ice, float(np.max(ii)))
            np.maximum(peak_ice_by_cell, full_i, out=peak_ice_by_cell)
            max_occ = max(max_occ, max(x[3] for x in stats))
            if force or t - last_capture >= capture_interval - 1e-9:
                times.append(t)
                currents.append(current)
                charges.append(charge)
                temperatures.append(full_t.copy())
                plates.append((plate - T_FREEZE).copy())
                voltages.append(full_v.copy())
                ice_series.append(full_i.copy())
                last_capture = t
            return full_t, full_v, full_i

        current = prescribed_current(0.0, mode, heat_s, postload)
        means, vv, ii = observe(current, True, True)
        if mode == "cooperative" and np.min(means) > temp_margin and np.max(ii) < 0.99 and min_v >= VSAFE_V:
            success, reason, startup = True, "startup", 0.0
        if mode == "preheat" and heat_s == 0 and np.min(means) > temp_margin and min_v >= VSAFE_V:
            success, reason, startup = True, "startup", 0.0

        while t < time_cap - 1e-11 and (not success or not stop_on_success):
            step = min(dt_max, time_cap - t)
            for point in current_breakpoints(mode, heat_s, postload):
                if t + 1e-10 < point < t + step - 1e-10:
                    step = point - t
            jstart = prescribed_current(t, mode, heat_s, postload)
            jmid = prescribed_current(t + step / 2, mode, heat_s, postload)
            if charge_limit is not None and jmid > 0:
                step = min(step, max((charge_limit - charge) / jmid, 0.0))
            if step < 1e-10:
                reason = "charge_limit"
                break
            # At a preheat switch, check the right-side loaded voltage before
            # letting a subsequent temperature rise conceal a violation.
            if mode == "preheat" and t >= heat_s - 1e-10 and min_before_load is None:
                min_before_load = float(np.min(means))
                if min_before_load <= temp_margin:
                    reason = "not_preheated"
                    break
            if jstart > 0 or mode == "cooperative":
                means, vv, ii = observe(jstart, False, True)
                if min_v < VSAFE_V - 1e-9:
                    reason = "voltage_below_0.30V"
                    break
            if mode == "preheat" and not success and t >= heat_s - 1e-10 and min_before_load is not None:
                if float(np.min(means)) > temp_margin and np.max(ii) < 0.99 and min_v >= VSAFE_V - 1e-9:
                    success, reason, startup = True, "startup", t
                    if stop_on_success:
                        observe(jstart, True, True)
                        break
            current_am2 = 1e4 * jmid
            intrinsic = np.empty_like(temp)
            for z in range(nstate):
                production = current_am2 * MW / (2 * F)
                available = max(14.0 - lam[z], 0.0)
                uptake = min(production * params.uptake_fraction * available
                             / max(14.0 - params.membrane_lambda, 1e-6),
                             available * water_per_lambda / step)
                new_water, new_ice, outflow, ice_rate = m._water_step(
                    temp[z], water[z], ice[z], step, current_am2, uptake, params)
                new_lam = lam[z] + uptake * step / water_per_lambda
                v_heat, _, _, _ = self._voltage(temp[z], new_water, new_ice,
                                                new_lam, current_am2, self.factors[z])
                intrinsic[z] = self._intrinsic_heat_step(
                    temp[z], new_water, new_ice, ice_rate, step,
                    current_am2, v_heat)
                if t < heat_s - 1e-10:
                    # Uniform mean shift adds exactly 25*q_k*dt J.
                    intrinsic[z] += AREA_CM2 * local_power[z] * step / self.cell_capacity_j_k
                q_elec += self.weights[z] * AREA_M2 * step * current_am2 * (ETH - v_heat)
                q_latent += self.weights[z] * AREA_M2 * step * LATENT_FREEZE * float(np.dot(ice_rate, m.dx))
                escaped[z] += outflow * step
                produced[z] += production * step
                sorbed[z] += uptake * step
                water[z], ice[z], lam[z] = new_water, new_ice, new_lam
            if t < heat_s - 1e-10:
                aux_energy += AREA_CM2 * float(np.sum(powers)) * step
            temp, plate, lost = self._heat_network(intrinsic, plate, ambient_c + T_FREEZE, step)
            q_lost += lost
            t += step
            charge += jmid * step
            errors = produced - np.array([float(np.dot(w, m.dx)) for w in water]) - escaped - sorbed
            max_water_error = max(max_water_error, float(np.max(np.abs(errors))))
            current = prescribed_current(t, mode, heat_s, postload)
            track = True
            means, vv, ii = observe(current, t >= heat_s - 1e-10 and abs(t - heat_s) < 1e-8, track)
            if track and min_v < VSAFE_V - 1e-9:
                reason = "voltage_below_0.30V"
                break
            if mode == "preheat" and t >= heat_s - 1e-10 and min_before_load is None:
                min_before_load = float(np.min(means))
                if min_before_load <= temp_margin:
                    reason = "not_preheated"
                    break
            eligible = mode == "cooperative" or t >= heat_s - 1e-10
            if (eligible and not success and np.min(means) > temp_margin
                    and np.max(ii) < 0.99 and min_v >= VSAFE_V - 1e-9):
                success, reason, startup = True, "startup", t
            if charge_limit is not None and charge >= charge_limit - 1e-9 and not success:
                reason = "charge_limit"
                break

        if not times or abs(times[-1] - t) > 1e-8:
            means, vv, ii = observe(prescribed_current(t, mode, heat_s, postload), True,
                                    True)
        actual_heat_s = min(t, heat_s)
        storage = (AREA_M2 * float(np.sum(self.weights * np.sum(
            (temp - (initial_c + T_FREEZE)) * self.capacity_areal * m.dx, axis=1)))
            + ENDPLATE_CAP_J_K * float(np.sum(plate - (initial_c + T_FREEZE))))
        balance = q_elec + q_latent + aux_energy - q_lost - storage
        return AuxResult(mode, tuple(powers.tolist()), heat_s, actual_heat_s,
                         success, reason, startup, np.array(times),
                         np.array(currents), np.array(charges),
                         np.array(temperatures), np.array(plates),
                         np.array(voltages), np.array(ice_series),
                         min_v, min_v_by_cell.copy(),
                         max_ice, peak_ice_by_cell.copy(),
                         max_occ, min_gas, max_water_error,
                         balance, q_elec, q_latent, aux_energy, q_lost, storage,
                         means.copy(), vv.copy(), ii.copy(), min_before_load)

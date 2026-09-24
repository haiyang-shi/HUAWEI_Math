"""Five-cell cold-start stack built on the calibrated Q1 finite-volume model.

The stack is left/right symmetric, so three through-plane states represent
cells (1,5), (2,4), and 3.  A seven-node thermal network holds two endplates
and all five cell means.  Q1 through-plane heat solves precede an implicit,
energy-conserving mean-temperature shift from the stack network.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '问题1'))
from model import (ColdStartModel, Parameters, R, F, MW, RHO_ICE,
                   RHO_LIQ, T_FREEZE, T_REF, ETH, LATENT_FREEZE)


AREA_M2 = 25e-4
AREA_CM2 = 25.0
QMAX_C_CM2 = 20.0
JMAX_A_CM2 = 0.5
VSAFE_V = 0.30
ENDPLATE_CAP_J_K = 7900.0 * 500.0 * 0.010 * AREA_M2
ENDPLATE_CONV_W_K = 40.0 * AREA_M2


def calibrated_parameters():
    values = json.loads((ROOT / 'result/metrics.json').read_text(encoding='utf-8'))['parameters']
    return Parameters(**values)


def policy_current(family, parameters, t):
    """Current density in A/cm2; switch values are right-continuous."""
    p = np.asarray(parameters, dtype=float)
    if family == 'constant':
        return float(p[0])
    if family == 'linear':
        j0, j1, ramp = p
        return float(j0 + (j1-j0)*min(max(t/ramp, 0.0), 1.0))
    if family == 'step':
        j1, j2, j3, tau1, tau2 = p
        return float(j1 if t < tau1 else (j2 if t < tau2 else j3))
    raise ValueError(family)


def breakpoints(family, parameters):
    p = np.asarray(parameters, dtype=float)
    if family == 'linear':
        return [float(p[2])]
    if family == 'step':
        return [float(p[3]), float(p[4])]
    return []


def validate_policy(family, parameters):
    p = np.asarray(parameters, dtype=float)
    if not np.all(np.isfinite(p)):
        return False
    if family == 'constant':
        return len(p) == 1 and 0 <= p[0] <= JMAX_A_CM2
    if family == 'linear':
        return len(p) == 3 and 0 <= p[0] <= p[1] <= JMAX_A_CM2 and p[2] > 0
    if family == 'step':
        return (len(p) == 5 and 0 <= p[0] <= p[1] <= p[2] <= JMAX_A_CM2
                and 0 < p[3] < p[4])
    return False


@dataclass
class StackResult:
    family: str
    parameters: tuple
    initial_c: float
    success: bool
    reason: str
    time_s: np.ndarray
    current_acm2: np.ndarray
    charge_c_cm2: np.ndarray
    temperature_c: np.ndarray  # columns are cells 1..5
    endplate_temperature_c: np.ndarray  # left/right
    voltage_v: np.ndarray
    ice_fraction: np.ndarray
    min_voltage_v: float
    max_ice_fraction: float
    min_gas_molm3: float
    max_pore_occupancy: float
    max_water_balance_kg_m2: float
    energy_balance_j: float
    heat_generated_j: float
    latent_heat_j: float
    heat_lost_j: float
    heat_stored_j: float

    @property
    def startup_s(self):
        return float(self.time_s[-1]) if self.success else None

    @property
    def used_charge(self):
        return float(self.charge_c_cm2[-1])


class StackModel:
    def __init__(self, mesh_factor=1, params=None, end_concentration_factor=10.0,
                 link_resistance_scale=1.0):
        self.cell = ColdStartModel(mesh_factor=mesh_factor)
        self.params = calibrated_parameters() if params is None else params
        self.end_concentration_factor = float(end_concentration_factor)
        self.link_resistance_scale = float(link_resistance_scale)
        self.capacity_areal = self.cell.c0 + self.params.plate_capacity_scale * self.cell.plate_capacity_areal / self.cell.length
        self.cell_capacity_j_k = AREA_M2 * float(np.dot(self.capacity_areal, self.cell.dx))
        # Q1 MEA layers in series plus two 2 mm bipolar plates between cell means.
        resistance_m2k_w = float(np.sum(self.cell.dx / self.cell.k0) + 0.004 / 95.0)
        self.link_g_w_k = AREA_M2 / (resistance_m2k_w * self.link_resistance_scale)
        # Half a cell and half the 10 mm endplate between their mean temperatures.
        self.plate_g_w_k = AREA_M2 / ((0.5*resistance_m2k_w + 0.005/15.0)*self.link_resistance_scale)

    def _voltage(self, temperature_k, water, ice, lam, current_am2, factor):
        m = self.cell
        _, liquid, eps_ice, eps_g = m.phases(temperature_k, water, ice)
        raw, gas = m.voltage(temperature_k, eps_ice, eps_g, current_am2, self.params, lam)
        if factor != 1:
            _, c_o2, dgas = m.gas_concentrations(temperature_k, eps_g, current_am2)
            c_mean = float(np.mean(c_o2[m.ccl]))
            resist = float(np.sum(m.dx[m.names == 'cGDL'] / np.maximum(dgas[m.names == 'cGDL'], 1e-16)))
            resist += 0.5 * float(np.sum(m.dx[m.ccl] / np.maximum(dgas[m.ccl], 1e-16)))
            jlim = 4*F*max(c_mean, 1e-8)/max(resist, 1e-12)
            t_mean = float(np.dot(temperature_k, m.dx)/m.length)
            eta_con = -R*t_mean/(4*F)*np.log(max(1-current_am2/max(jlim, 1e-9), 1e-8))
            raw -= (factor-1)*eta_con
        occupancy = (eps_ice + liquid/RHO_LIQ)/np.maximum(m.eps0, 1e-8)
        return (float(raw), float(gas), float(np.max(eps_ice[m.porous])),
                float(np.max(occupancy[m.porous])))

    def _intrinsic_heat_step(self, temp, water, ice, ice_rate, dt, current_am2, voltage):
        """Q1 heat equation without the single-cell ambient boundary terms."""
        m = self.cell
        _, liquid, eps_ice, _ = m.phases(temp, water, ice)
        k = m.k0 + eps_ice*(2.3-0.024) + liquid/RHO_LIQ*(0.6-0.024)
        diag = self.capacity_areal.copy()
        lower = np.zeros(m.n-1)
        upper = np.zeros(m.n-1)
        rhs = self.capacity_areal*temp + dt*(current_am2*(ETH-voltage)/m.length + LATENT_FREEZE*ice_rate)
        for i in range(m.n-1):
            g = 2/(m.dx[i]/k[i]+m.dx[i+1]/k[i+1])
            diag[i] += dt*g/m.dx[i]
            upper[i] -= dt*g/m.dx[i]
            lower[i] -= dt*g/m.dx[i+1]
            diag[i+1] += dt*g/m.dx[i+1]
        return m._tridiagonal(diag, lower, upper, rhs)

    def _stack_heat_step(self, intrinsic_profiles, old_plate_k, ambient_k, dt):
        m = self.cell
        means = np.array([float(np.dot(row, m.dx)/m.length) for row in intrinsic_profiles])
        before = np.array([old_plate_k[0], means[0], means[1], means[2],
                           means[1], means[0], old_plate_k[1]])
        caps = np.array([ENDPLATE_CAP_J_K, self.cell_capacity_j_k, self.cell_capacity_j_k,
                         self.cell_capacity_j_k, self.cell_capacity_j_k, self.cell_capacity_j_k,
                         ENDPLATE_CAP_J_K])
        mat = np.diag(caps.copy())
        rhs = caps*before
        for edge in range(6):
            g = self.plate_g_w_k if edge in (0, 5) else self.link_g_w_k
            mat[edge, edge] += dt*g
            mat[edge+1, edge+1] += dt*g
            mat[edge, edge+1] -= dt*g
            mat[edge+1, edge] -= dt*g
        for edge in (0, 6):
            mat[edge, edge] += dt*ENDPLATE_CONV_W_K
            rhs[edge] += dt*ENDPLATE_CONV_W_K*ambient_k
        after = np.linalg.solve(mat, rhs)
        for z, pos in enumerate((1, 2, 3)):
            intrinsic_profiles[z] += after[pos]-means[z]
        lost = dt*ENDPLATE_CONV_W_K*(after[0]+after[6]-2*ambient_k)
        return intrinsic_profiles, after[[0, 6]], float(lost)

    def simulate(self, family, parameters, initial_c=-10.0, ambient_c=None,
                 dt_max=0.5, time_cap=800.0, capture_interval=1.0,
                 voltage_margin=0.0):
        if not validate_policy(family, parameters):
            raise ValueError('Invalid current-policy parameters')
        if ambient_c is None:
            ambient_c = initial_c
        m = self.cell
        params = self.params
        temperature = np.full((3, m.n), initial_c+T_FREEZE)
        water = np.zeros((3, m.n))
        ice = np.zeros((3, m.n))
        lam = np.full(3, params.membrane_lambda)
        plate = np.full(2, initial_c+T_FREEZE)
        escaped = np.zeros(3)
        produced = np.zeros(3)
        sorbed = np.zeros(3)
        weights = np.array([2.0, 2.0, 1.0])
        water_per_lambda = 2150.0*MW*12e-6
        family_params = tuple(float(v) for v in parameters)
        factors = (self.end_concentration_factor, 1.0, 1.0)
        charge = 0.0
        t = 0.0
        q_elec = q_latent = q_lost = 0.0
        min_v = float('inf')
        max_ice = 0.0
        min_gas = float('inf')
        max_occ = 0.0
        max_water_error = 0.0
        times, currents, charges, ts, pts, vs, iss = [], [], [], [], [], [], []
        last_capture = -float('inf')
        reason = 'time_cap'
        success = False

        def observe(current, force=False):
            nonlocal min_v, max_ice, min_gas, max_occ, last_capture, reason
            stats = [self._voltage(temperature[z], water[z], ice[z], lam[z],
                                   current*1e4, factors[z]) for z in range(3)]
            vv = np.array([v[0] for v in stats])
            ii = np.array([v[2] for v in stats])
            gas = min(v[1] for v in stats)
            occ = max(v[3] for v in stats)
            min_v = min(min_v, float(np.min(vv)))
            max_ice = max(max_ice, float(np.max(ii)))
            min_gas = min(min_gas, gas)
            max_occ = max(max_occ, occ)
            cell_temp = (temperature @ m.dx)/m.length-T_FREEZE
            if force or t-last_capture >= capture_interval-1e-9:
                times.append(t)
                currents.append(current)
                charges.append(charge)
                ts.append(cell_temp[[0, 1, 2, 1, 0]].copy())
                pts.append(plate-T_FREEZE)
                vs.append(vv[[0, 1, 2, 1, 0]].copy())
                iss.append(ii[[0, 1, 2, 1, 0]].copy())
                last_capture = t
            if min_v < VSAFE_V+voltage_margin-1e-9:
                reason = 'voltage_below_0.30V'
                return False, cell_temp
            return True, cell_temp

        valid, means = observe(policy_current(family, family_params, 0), True)
        if valid and np.min(means)>0.0 and max_ice<0.99:
            success, reason = True, 'startup'
        while valid and not success and t < time_cap-1e-12:
            step = min(dt_max, time_cap-t)
            for point in breakpoints(family, family_params):
                if t+1e-10 < point < t+step-1e-10:
                    step = point-t
            jmid = policy_current(family, family_params, t+step/2)
            if jmid > 0:
                step = min(step, max((QMAX_C_CM2-charge)/jmid, 0.0))
            if step < 1e-10:
                reason = 'charge_limit'
                break
            # At a jump check the voltage just after the new load, before state evolution.
            valid, _ = observe(policy_current(family, family_params, t+1e-9))
            if not valid:
                observe(policy_current(family, family_params, t), True)
                break
            current_am2 = 1e4*jmid
            intrinsic = np.empty_like(temperature)
            for z in range(3):
                old_temp = temperature[z]
                old_water = water[z]
                old_ice = ice[z]
                production = current_am2*MW/(2*F)
                available = max(14.0-lam[z], 0.0)
                uptake = min(production*params.uptake_fraction*available/max(14.0-params.membrane_lambda, 1e-6),
                             available*water_per_lambda/step)
                new_water, new_ice, outflow, ice_rate = m._water_step(old_temp, old_water, old_ice,
                                                                       step, current_am2, uptake, params)
                new_lam = lam[z] + uptake*step/water_per_lambda
                vv, _, _, _ = self._voltage(old_temp, new_water, new_ice, new_lam,
                                             current_am2, factors[z])
                intrinsic[z] = self._intrinsic_heat_step(old_temp, new_water, new_ice,
                                                           ice_rate, step, current_am2, vv)
                q_elec += weights[z]*AREA_M2*step*current_am2*(ETH-vv)
                q_latent += weights[z]*AREA_M2*step*LATENT_FREEZE*float(np.dot(ice_rate, m.dx))
                escaped[z] += outflow*step
                produced[z] += production*step
                sorbed[z] += uptake*step
                water[z] = new_water
                ice[z] = new_ice
                lam[z] = new_lam
            temperature, plate, lost = self._stack_heat_step(intrinsic, plate,
                                                               ambient_c+T_FREEZE, step)
            q_lost += lost
            t += step
            charge += jmid*step
            errors = produced - np.array([float(np.dot(w, m.dx)) for w in water]) - escaped - sorbed
            max_water_error = max(max_water_error, float(np.max(np.abs(errors))))
            valid, means = observe(policy_current(family, family_params, t), False)
            if valid and np.min(means)>0.0 and max_ice<0.99 and charge <= QMAX_C_CM2+1e-8:
                success, reason = True, 'startup'
            elif charge >= QMAX_C_CM2-1e-10:
                reason = 'charge_limit'
                break
        observe(policy_current(family, family_params, t), True)
        stack_means = (temperature @ m.dx)/m.length
        storage = (AREA_M2*float(np.sum(weights*np.sum((temperature-(initial_c+T_FREEZE))
                                                    *self.capacity_areal*m.dx, axis=1)))
                   + ENDPLATE_CAP_J_K*float(np.sum(plate-(initial_c+T_FREEZE))))
        return StackResult(family, family_params, initial_c, success, reason,
                           np.array(times), np.array(currents), np.array(charges),
                           np.array(ts), np.array(pts), np.array(vs), np.array(iss),
                           min_v, max_ice, min_gas, max_occ, max_water_error,
                           q_elec+q_latent-q_lost-storage, q_elec, q_latent, q_lost, storage)

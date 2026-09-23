"""One-dimensional finite-volume reduced PEMFC cold-start model for Q1.

The five MEA layers are spatially resolved.  H2/O2 are quasi-steady because
their diffusion time is much shorter than the 0.2 s observation interval.
Pore water, ice, temperature and mean membrane hydration are transient.
Membrane uptake is an explicitly parameterized reduction of distributed
product water; the two observed curves cannot identify its microscopic
transport independently. Bipolar-plate heat capacity is attached to the MEA
thermal field.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded

from data import Experiment


F = 96485.0
R = 8.314
MW = 0.018
RHO_ICE = 920.0
RHO_LIQ = 990.0
P0 = 101325.0
T_REF = 298.15
T_FREEZE = 273.15
LATENT_FREEZE = 333600.0
ETH = 1.48


@dataclass(frozen=True)
class Parameters:
    j0_ref_am2: float = 0.01  # PDF remark 1 initial calibration value
    plate_capacity_scale: float = 1.0
    freeze_rate_s: float = 0.20  # unobserved sensitivity parameter
    melt_rate_s: float = 0.20
    liquid_diffusivity_scale: float = 1.0  # unobserved sensitivity parameter
    ice_area_exponent: float = 3.5  # attachment 1
    membrane_lambda: float = 3.0  # attachment 1 initial lambda
    uptake_fraction: float = 0.75  # fraction of generated water sorbed while membrane is dry
    hydration_activity_exponent: float = 2.0  # effective CL proton-accessibility response
    contact_ohm_m2: float = 1.0e-6  # 0.01 ohm cm2, PDF remark 1
    dry_interface_ohm_m2: float = 0.0  # additional hydration-dependent CL/PEM resistance


@dataclass(frozen=True)
class Simulation:
    time_s: np.ndarray
    temperature_c: np.ndarray
    voltage_v: np.ndarray
    max_ice_fraction: np.ndarray
    max_pore_ice_saturation: np.ndarray
    temperature_field_c: np.ndarray
    water_field_kgm3: np.ndarray
    ice_field_kgm3: np.ndarray
    water_escaped_kgm2: np.ndarray
    generated_water_kgm2: np.ndarray
    mass_balance_error_kgm2: np.ndarray
    min_gas_concentration_molm3: np.ndarray
    max_pore_occupancy: np.ndarray
    membrane_lambda: np.ndarray
    membrane_sorbed_increment_kgm2: np.ndarray
    heat_generated_jm2: np.ndarray
    heat_lost_jm2: np.ndarray
    latent_released_jm2: np.ndarray
    heat_storage_jm2: np.ndarray
    energy_balance_error_jm2: np.ndarray
    solver_success: bool
    solver_message: str
    nfev: int


class ColdStartModel:
    def __init__(self, mesh_factor: int = 1):
        layers = [
            ('aGDL', 150e-6, 8, 0.8, 0.30, 185 * 545, 1.10e-4, 8.69e-5, 2.0e-9),
            ('aCL', 3.4e-6, 3, 0.3916, 0.27, 970 * 240, 1.10e-4, 8.69e-5, 2.0e-10),
            ('PEM', 12e-6, 4, 0.0, 0.24, 2150 * 1050, 0.0, 0.0, 0.0),
            ('cCL', 11.3e-6, 5, 0.4207, 0.27, 970 * 240, 2.20e-5, 2.48e-5, 2.0e-10),
            ('cGDL', 150e-6, 8, 0.8, 0.30, 185 * 545, 2.20e-5, 2.48e-5, 2.0e-9),
        ]
        props = []
        for name, thickness, count, eps, k, cap, dgas, dv, dl in layers:
            props += [(name, thickness / (count * mesh_factor), eps, k, cap, dgas, dv, dl)] * (count * mesh_factor)
        self.names = np.array([r[0] for r in props])
        self.dx = np.array([r[1] for r in props])
        self.eps0 = np.array([r[2] for r in props])
        self.k0 = np.array([r[3] for r in props])
        self.c0 = np.array([r[4] for r in props])
        self.dgas0 = np.array([r[5] for r in props])
        self.dv0 = np.array([r[6] for r in props])
        self.dl0 = np.array([r[7] for r in props])
        self.n = len(props)
        self.length = sum(self.dx)
        self.porous = self.eps0 > 0
        self.ccl = self.names == 'cCL'
        self.acl = self.names == 'aCL'
        self.pem = self.names == 'PEM'
        self.h = 40.0  # W m-2 K-1, attachment 1
        self.plate_capacity_areal = 2 * 0.002 * 1980 * 766  # two bipolar plates

    @staticmethod
    def psat_pa(temperature_k: np.ndarray) -> np.ndarray:
        tc = np.asarray(temperature_k) - T_FREEZE
        above = 611.21 * np.exp((18.678 - tc / 234.5) * tc / (257.14 + tc))
        below = 611.15 * np.exp((23.036 - tc / 333.7) * tc / (279.82 + tc))
        return np.where(tc >= 0, above, below)

    def phases(self, temperature_k: np.ndarray, water: np.ndarray, ice: np.ndarray):
        eps_ice = np.clip(ice, 0, None) / RHO_ICE
        pore_free = np.maximum(self.eps0 - eps_ice, 1e-8)
        msat = pore_free * MW * self.psat_pa(temperature_k) / (R * temperature_k)
        mobile = np.maximum(water - ice, 0)
        vapor = np.where(self.porous, np.minimum(mobile, msat), 0)
        liquid = np.where(self.porous, np.maximum(mobile - vapor, 0), 0)
        eps_g = np.maximum(self.eps0 - eps_ice - liquid / RHO_LIQ, 1e-8)
        return vapor, liquid, eps_ice, eps_g

    def gas_concentrations(self, temperature_k, eps_g, current_am2):
        dgas = self.dgas0 * (temperature_k / T_REF) ** 1.75 * (eps_g ** 1.5)
        c_h2 = np.full(self.n, np.nan)
        c_o2 = np.full(self.n, np.nan)
        # Dry hydrogen enters at the aGDL outer boundary.
        flux = current_am2 / (2 * F)
        c = P0 / (R * temperature_k[0])
        for i in np.flatnonzero((self.names == 'aGDL') | self.acl):
            d = max(dgas[i], 1e-16)
            c_h2[i] = c - flux * self.dx[i] / (2 * d)
            c -= flux * self.dx[i] / (2 * d)
            if self.acl[i]:
                flux -= current_am2 * self.dx[i] / (2 * F * self.dx[self.acl].sum())
            c -= max(flux, 0) * self.dx[i] / (2 * d)
        # Dry air enters at the cGDL outer boundary.  Mass fractions 0.233/0.767
        # give an oxygen mole fraction of approximately 0.210.
        y_o2 = (0.233 / 32) / (0.233 / 32 + 0.767 / 28)
        flux = current_am2 / (4 * F)
        c = y_o2 * P0 / (R * temperature_k[-1])
        for i in np.flatnonzero(self.ccl | (self.names == 'cGDL'))[::-1]:
            d = max(dgas[i], 1e-16)
            c_o2[i] = c - flux * self.dx[i] / (2 * d)
            c -= flux * self.dx[i] / (2 * d)
            if self.ccl[i]:
                flux -= current_am2 * self.dx[i] / (4 * F * self.dx[self.ccl].sum())
            c -= max(flux, 0) * self.dx[i] / (2 * d)
        return c_h2, c_o2, dgas

    def voltage(self, temperature_k, eps_ice, eps_g, current_am2, params, membrane_lambda=None):
        c_h2, c_o2, dgas = self.gas_concentrations(temperature_k, eps_g, current_am2)
        tmean = float(np.dot(temperature_k, self.dx) / self.length)
        ch = float(np.mean(c_h2[self.acl]))
        co = float(np.mean(c_o2[self.ccl]))
        ph = max(ch * R * tmean, 1.0)
        po = max(co * R * tmean, 1.0)
        erev = 1.229 - 8.5e-4 * (tmean - T_REF) + R * tmean / (2 * F) * np.log((ph / P0) * np.sqrt(po / P0))
        pore_ice = np.clip(eps_ice[self.ccl] / self.eps0[self.ccl], 0, 0.999999)
        if membrane_lambda is None:
            membrane_lambda = params.membrane_lambda
        ice_area = float(np.mean((1 - pore_ice) ** params.ice_area_exponent))
        hydration_area = (max(membrane_lambda, 0.1) / max(params.membrane_lambda, 0.1)) ** params.hydration_activity_exponent
        area_factor = max(ice_area * hydration_area, 1e-6)
        j0 = params.j0_ref_am2 * np.exp(-67000 / R * (1 / tmean - 1 / T_REF))
        eta_act = R * tmean / (0.5 * F) * np.arcsinh(current_am2 / max(2 * j0 * area_factor, 1e-20))
        kappa = (0.5139 * membrane_lambda - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / tmean))
        dry_interface = params.dry_interface_ohm_m2 * (params.membrane_lambda / max(membrane_lambda, 0.1)) ** 2
        eta_ohm = current_am2 * (12e-6 / max(kappa, 0.01) + params.contact_ohm_m2 + dry_interface)
        # Same limiting-current relationship as PDF remark 1, evaluated at cCL.
        resist = np.sum(self.dx[self.names == 'cGDL'] / np.maximum(dgas[self.names == 'cGDL'], 1e-16))
        resist += 0.5 * np.sum(self.dx[self.ccl] / np.maximum(dgas[self.ccl], 1e-16))
        jlim = 4 * F * max(co, 1e-8) / max(resist, 1e-12)
        eta_con = -R * tmean / (4 * F) * np.log(max(1 - current_am2 / max(jlim, 1e-9), 1e-8))
        return float(erev - eta_act - eta_ohm - eta_con), float(np.nanmin([np.nanmin(c_h2), np.nanmin(c_o2)]))

    def _rhs(self, time_s, state, experiment, params):
        n = self.n
        t = state[:n]
        water = state[n:2*n]
        ice = state[2*n:3*n]
        j = float(np.interp(time_s, experiment.time_s, experiment.current_density_acm2)) * 1e4
        vapor, liquid, eps_ice, eps_g = self.phases(t, water, ice)
        v, _ = self.voltage(t, eps_ice, eps_g, j, params)

        # Local freezing/melting, with saturation limiting ice to the pore space.
        cold = np.clip((T_FREEZE - t) / 20, 0, 3)
        hot = np.clip((t - T_FREEZE) / 5, 0, 3)
        freeze = params.freeze_rate_s * cold * liquid * np.maximum(1 - eps_ice / np.maximum(self.eps0, 1e-8), 0)
        melt = params.melt_rate_s * hot * np.maximum(ice, 0)
        dice = np.where(self.porous, freeze - melt, 0)

        dwater = np.zeros(n)
        dwater[self.ccl] = j * MW / (2 * F * self.dx[self.ccl].sum())
        dv = self.dv0 * (t / T_REF) ** 1.75 * eps_g ** 1.5
        dl = self.dl0 * params.liquid_diffusivity_scale
        cv = vapor / np.maximum(eps_g, 1e-8)
        for i in range(n - 1):
            if not (self.porous[i] and self.porous[i+1]):
                continue
            dface_v = 2 * dv[i] * dv[i+1] / max(dv[i] + dv[i+1], 1e-20)
            dface_l = 2 * dl[i] * dl[i+1] / max(dl[i] + dl[i+1], 1e-20)
            separation = 0.5 * (self.dx[i] + self.dx[i+1])
            flux = -dface_v * (cv[i+1] - cv[i]) / separation - dface_l * (liquid[i+1] - liquid[i]) / separation
            dwater[i] -= flux / self.dx[i]
            dwater[i+1] += flux / self.dx[i+1]
        escaped_left = dv[0] * cv[0] / (self.dx[0] / 2)
        escaped_right = dv[-1] * cv[-1] / (self.dx[-1] / 2)
        dwater[0] -= escaped_left / self.dx[0]
        dwater[-1] -= escaped_right / self.dx[-1]

        # Bipolar-plate thermal mass is distributed over the resolved MEA field.
        heat_capacity = self.c0 + params.plate_capacity_scale * self.plate_capacity_areal / self.length
        k = self.k0 + eps_ice * (2.3 - 0.024) + liquid / RHO_LIQ * (0.6 - 0.024)
        dheat = np.full(n, j * (ETH - v) / self.length)
        for i in range(n - 1):
            conductance = 2 / (self.dx[i] / k[i] + self.dx[i+1] / k[i+1])
            flux = conductance * (t[i+1] - t[i])
            dheat[i] += flux / self.dx[i]
            dheat[i+1] -= flux / self.dx[i+1]
        ambient_k = experiment.ambient_c + T_FREEZE
        dheat[0] += self.h * (ambient_k - t[0]) / self.dx[0]
        dheat[-1] += self.h * (ambient_k - t[-1]) / self.dx[-1]
        dheat += LATENT_FREEZE * dice
        dt = dheat / heat_capacity
        return np.r_[dt, dwater, dice, escaped_left + escaped_right]

    def simulate_bdf(self, experiment: Experiment, params: Parameters, *, rtol=2e-5, atol=1e-7, method='BDF') -> Simulation:
        times = experiment.time_s
        state0 = np.r_[np.full(self.n, experiment.temperature_c[0] + T_FREEZE),
                       np.zeros(self.n), np.zeros(self.n), 0.0]
        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, experiment, params),
            (float(times[0]), float(times[-1])), state0, t_eval=times,
            method=method, rtol=rtol, atol=atol, max_step=0.4,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        tfield = sol.y[:self.n].T
        waterfield = sol.y[self.n:2*self.n].T
        icefield = sol.y[2*self.n:3*self.n].T
        escaped = sol.y[-1]
        temp = (tfield @ self.dx) / self.length - T_FREEZE
        voltage = []
        max_ice = []
        max_sat = []
        min_gas = []
        max_occupancy = []
        for t, row_t, row_w, row_i in zip(times, tfield, waterfield, icefield):
            vapor, liquid, eps_ice, eps_g = self.phases(row_t, row_w, row_i)
            j = float(np.interp(t, experiment.time_s, experiment.current_density_acm2)) * 1e4
            v, gas = self.voltage(row_t, eps_ice, eps_g, j, params)
            voltage.append(v)
            max_ice.append(float(np.max(eps_ice)))
            max_sat.append(float(np.max(eps_ice[self.porous] / self.eps0[self.porous])))
            min_gas.append(gas)
            occupancy = (eps_ice + liquid / RHO_LIQ) / np.maximum(self.eps0, 1e-8)
            max_occupancy.append(float(np.max(occupancy[self.porous])))
        j_am2 = experiment.current_density_acm2 * 1e4
        produced_rate = j_am2 * MW / (2 * F)
        generated = np.r_[0, np.cumsum(0.5 * (produced_rate[1:] + produced_rate[:-1]) * np.diff(times))]
        retained = waterfield @ self.dx
        balance = generated - retained - escaped
        return Simulation(
            time_s=times, temperature_c=temp, voltage_v=np.array(voltage),
            max_ice_fraction=np.array(max_ice), max_pore_ice_saturation=np.array(max_sat),
            temperature_field_c=tfield - T_FREEZE,
            water_field_kgm3=waterfield, ice_field_kgm3=icefield,
            water_escaped_kgm2=escaped, generated_water_kgm2=generated,
            mass_balance_error_kgm2=balance, min_gas_concentration_molm3=np.array(min_gas),
            max_pore_occupancy=np.array(max_occupancy),
            membrane_lambda=np.full(len(times), params.membrane_lambda),
            membrane_sorbed_increment_kgm2=np.zeros(len(times)),
            heat_generated_jm2=np.full(len(times), np.nan),
            heat_lost_jm2=np.full(len(times), np.nan),
            latent_released_jm2=np.full(len(times), np.nan),
            heat_storage_jm2=np.full(len(times), np.nan),
            energy_balance_error_jm2=np.full(len(times), np.nan), solver_success=sol.success,
            solver_message=sol.message, nfev=sol.nfev,
        )

    @staticmethod
    def _tridiagonal(diagonal, lower, upper, rhs):
        band = np.zeros((3, len(diagonal)))
        band[0, 1:] = upper
        band[1, :] = diagonal
        band[2, :-1] = lower
        return solve_banded((1, 1), band, rhs, check_finite=False)

    def _water_step(self, temperature_k, water, ice, dt, current_am2, uptake_rate_kgm2s, params):
        """Conservative implicit vapor/liquid transport with phase active sets."""
        n = self.n
        mobile_old = np.maximum(water - ice, 0)
        _, _, eps_ice, eps_g = self.phases(temperature_k, water, ice)
        msat = np.where(self.porous, (self.eps0 - eps_ice) * MW * self.psat_pa(temperature_k) / (R * temperature_k), 0)
        dv = self.dv0 * (temperature_k / T_REF) ** 1.75 * eps_g ** 1.5
        dl = self.dl0 * params.liquid_diffusivity_scale
        source = np.zeros(n)
        source[self.ccl] = (current_am2 * MW / (2 * F) - uptake_rate_kgm2s) / self.dx[self.ccl].sum()
        wet = mobile_old >= msat
        wet[~self.porous] = False
        mobile_new = mobile_old.copy()
        escaped_rate = 0.0
        for _ in range(12):
            av = (~wet & self.porous).astype(float)
            bv = np.where(wet & self.porous, msat, 0)
            al = (wet & self.porous).astype(float)
            bl = np.where(wet & self.porous, -msat, 0)
            diag = np.ones(n)
            lower = np.zeros(n-1)
            upper = np.zeros(n-1)
            rhs = mobile_old + dt * source
            for i in range(n-1):
                if not (self.porous[i] and self.porous[i+1]):
                    continue
                face_v = 2 * dv[i] * dv[i+1] / max(dv[i] + dv[i+1], 1e-20)
                face_l = 2 * dl[i] * dl[i+1] / max(dl[i] + dl[i+1], 1e-20)
                distance = (self.dx[i] + self.dx[i+1]) / 2
                ai = (face_v * av[i] / max(eps_g[i], 1e-8) + face_l * al[i]) / distance
                aj = -(face_v * av[i+1] / max(eps_g[i+1], 1e-8) + face_l * al[i+1]) / distance
                const = (face_v * (bv[i] / max(eps_g[i], 1e-8) - bv[i+1] / max(eps_g[i+1], 1e-8))
                         + face_l * (bl[i] - bl[i+1])) / distance
                diag[i] += dt * ai / self.dx[i]
                upper[i] += dt * aj / self.dx[i]
                rhs[i] -= dt * const / self.dx[i]
                lower[i] -= dt * ai / self.dx[i+1]
                diag[i+1] -= dt * aj / self.dx[i+1]
                rhs[i+1] += dt * const / self.dx[i+1]
            for i in (0, n-1):
                boundary = dv[i] / max(eps_g[i], 1e-8) / (self.dx[i] / 2)
                diag[i] += dt * boundary * av[i] / self.dx[i]
                rhs[i] -= dt * boundary * bv[i] / self.dx[i]
            candidate = self._tridiagonal(diag, lower, upper, rhs)
            new_wet = (candidate >= msat) & self.porous
            mobile_new = candidate
            if np.array_equal(new_wet, wet):
                break
            wet = new_wet
        else:
            raise RuntimeError('Water phase active set failed to converge')
        if np.min(mobile_new[self.porous]) < -1e-8:
            raise RuntimeError(f'Negative mobile water after phase convergence: {np.min(mobile_new[self.porous]):.4g}')
        mobile_new = np.maximum(mobile_new, 0)
        vapor = np.where(self.porous, np.minimum(mobile_new, msat), 0)
        liquid = np.where(self.porous, np.maximum(mobile_new - vapor, 0), 0)
        escaped_rate = dv[0] * vapor[0] / max(eps_g[0], 1e-8) / (self.dx[0] / 2)
        escaped_rate += dv[-1] * vapor[-1] / max(eps_g[-1], 1e-8) / (self.dx[-1] / 2)
        cold = np.clip((T_FREEZE - temperature_k) / 20, 0, 3)
        hot = np.clip((temperature_k - T_FREEZE) / 5, 0, 3)
        freeze_fraction = 1 - np.exp(-dt * params.freeze_rate_s * cold)
        melt_fraction = 1 - np.exp(-dt * params.melt_rate_s * hot)
        capacity = np.maximum((self.eps0 - eps_ice) * RHO_ICE, 0)
        freeze_mass = np.minimum(liquid * freeze_fraction, capacity)
        melt_mass = np.maximum(ice, 0) * melt_fraction
        new_ice = np.where(self.porous, ice + freeze_mass - melt_mass, 0)
        new_water = np.where(self.porous, ice + mobile_new, 0)
        return new_water, new_ice, escaped_rate, (freeze_mass - melt_mass) / dt

    def _heat_step(self, temperature_k, water, ice, ice_rate, dt, current_am2, ambient_k, params, membrane_lambda):
        _, liquid, eps_ice, eps_g = self.phases(temperature_k, water, ice)
        voltage, _ = self.voltage(temperature_k, eps_ice, eps_g, current_am2, params, membrane_lambda)
        heat_capacity = self.c0 + params.plate_capacity_scale * self.plate_capacity_areal / self.length
        k = self.k0 + eps_ice * (2.3 - 0.024) + liquid / RHO_LIQ * (0.6 - 0.024)
        diag = heat_capacity.copy()
        lower = np.zeros(self.n-1)
        upper = np.zeros(self.n-1)
        rhs = heat_capacity * temperature_k + dt * (current_am2 * (ETH - voltage) / self.length + LATENT_FREEZE * ice_rate)
        for i in range(self.n-1):
            conductance = 2 / (self.dx[i] / k[i] + self.dx[i+1] / k[i+1])
            diag[i] += dt * conductance / self.dx[i]
            upper[i] -= dt * conductance / self.dx[i]
            lower[i] -= dt * conductance / self.dx[i+1]
            diag[i+1] += dt * conductance / self.dx[i+1]
        for i in (0, self.n-1):
            diag[i] += dt * self.h / self.dx[i]
            rhs[i] += dt * self.h * ambient_k / self.dx[i]
        new_temperature = self._tridiagonal(diag, lower, upper, rhs)
        generated_flux = current_am2 * (ETH - voltage)
        latent_flux = LATENT_FREEZE * float(np.dot(ice_rate, self.dx))
        lost_flux = self.h * (new_temperature[0] - ambient_k + new_temperature[-1] - ambient_k)
        return new_temperature, generated_flux, latent_flux, lost_flux

    def simulate(self, experiment: Experiment, params: Parameters, *, max_step=0.10) -> Simulation:
        """Fast implicit finite-volume solution, evaluated at the 184 observations."""
        times = experiment.time_s
        nt, n = len(times), self.n
        tfield = np.empty((nt, n))
        waterfield = np.zeros((nt, n))
        icefield = np.zeros((nt, n))
        escaped = np.zeros(nt)
        generated = np.zeros(nt)
        sorbed = np.zeros(nt)
        heat_generated = np.zeros(nt)
        heat_lost = np.zeros(nt)
        latent_released = np.zeros(nt)
        heat_storage = np.zeros(nt)
        lambda_series = np.full(nt, params.membrane_lambda)
        water_per_lambda = 2150.0 * MW * 12e-6 / 1.0  # EW=1000 g/mol=1 kg/mol
        tfield[0] = experiment.temperature_c[0] + T_FREEZE
        for m in range(1, nt):
            temperature_k = tfield[m-1].copy()
            water = waterfield[m-1].copy()
            ice = icefield[m-1].copy()
            outflow = escaped[m-1]
            produced = generated[m-1]
            absorbed = sorbed[m-1]
            q_generated = heat_generated[m-1]
            q_lost = heat_lost[m-1]
            q_latent = latent_released[m-1]
            membrane_lambda = lambda_series[m-1]
            steps = int(np.ceil((times[m] - times[m-1]) / max_step))
            dt = (times[m] - times[m-1]) / steps
            for sub in range(steps):
                tmid = times[m-1] + (sub + 0.5) * dt
                j = float(np.interp(tmid, times, experiment.current_density_acm2)) * 1e4
                production = j * MW / (2 * F)
                available = max(14.0 - membrane_lambda, 0)
                uptake_rate = min(production * params.uptake_fraction * available / max(14.0 - params.membrane_lambda, 1e-6),
                                  available * water_per_lambda / dt)
                water, ice, flux, ice_rate = self._water_step(temperature_k, water, ice, dt, j, uptake_rate, params)
                membrane_lambda += uptake_rate * dt / water_per_lambda
                temperature_k, q_gen_flux, q_latent_flux, q_lost_flux = self._heat_step(
                    temperature_k, water, ice, ice_rate, dt, j,
                    experiment.ambient_c + T_FREEZE, params, membrane_lambda)
                q_generated += q_gen_flux * dt
                q_latent += q_latent_flux * dt
                q_lost += q_lost_flux * dt
                outflow += flux * dt
                produced += production * dt
                absorbed += uptake_rate * dt
            tfield[m] = temperature_k
            waterfield[m] = water
            icefield[m] = ice
            escaped[m] = outflow
            generated[m] = produced
            sorbed[m] = absorbed
            heat_generated[m] = q_generated
            heat_lost[m] = q_lost
            latent_released[m] = q_latent
            heat_capacity = self.c0 + params.plate_capacity_scale * self.plate_capacity_areal / self.length
            heat_storage[m] = float(np.dot(heat_capacity * self.dx, temperature_k - tfield[0]))
            lambda_series[m] = membrane_lambda
        temp = tfield @ self.dx / self.length - T_FREEZE
        voltage, max_ice, max_sat, min_gas, occupancy = [], [], [], [], []
        for sample_t, row_t, row_w, row_i, lam in zip(times, tfield, waterfield, icefield, lambda_series):
            _, liquid, eps_ice, eps_g = self.phases(row_t, row_w, row_i)
            j = float(np.interp(sample_t, times, experiment.current_density_acm2)) * 1e4
            v, gas = self.voltage(row_t, eps_ice, eps_g, j, params, lam)
            voltage.append(v)
            max_ice.append(float(np.max(eps_ice)))
            max_sat.append(float(np.max(eps_ice[self.porous] / self.eps0[self.porous])))
            min_gas.append(gas)
            occ = (eps_ice + liquid / RHO_LIQ) / np.maximum(self.eps0, 1e-8)
            occupancy.append(float(np.max(occ[self.porous])))
        balance = generated - waterfield @ self.dx - escaped - sorbed
        energy_balance = heat_generated + latent_released - heat_lost - heat_storage
        return Simulation(times, temp, np.asarray(voltage), np.asarray(max_ice), np.asarray(max_sat),
                          tfield - T_FREEZE, waterfield, icefield, escaped, generated, balance,
                          np.asarray(min_gas), np.asarray(occupancy), lambda_series, sorbed,
                          heat_generated, heat_lost, latent_released, heat_storage, energy_balance, True,
                          'Implicit finite-volume solve completed', nt-1)

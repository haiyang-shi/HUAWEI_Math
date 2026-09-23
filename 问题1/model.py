"""Conservative five-layer, one-dimensional PEMFC cold-start model.

Pore water/ice and heat are finite-volume states. The membrane is resolved in
the same mesh: dissolved water diffuses and migrates by electro-osmotic drag.
Two bipolar plates have separate heat capacities and temperatures. Gas
transport is quasi-steady at the time scale of the supplied measurements.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from data import Experiment


F = 96485.0
R = 8.314
MW = 0.018
RHO_ICE = 920.0
RHO_LIQ = 990.0
RHO_PEM = 2150.0
WATER_PER_LAMBDA = RHO_PEM * MW / 1.0  # EW = 1000 g/mol = 1 kg/mol
P0 = 101325.0
T_REF = 298.15
T_FREEZE = 273.15
LATENT_FREEZE = 333600.0
ETH = 1.48


@dataclass(frozen=True)
class Parameters:
    j0_ref_am2: float = 0.01
    plate_capacity_scale: float = 1.0
    membrane_sorption_s: float = 10.0  # fixed sub-second equilibration assumption; not identified by data
    interface_ohm_m2: float = 1e-4
    interface_hydration_exponent: float = 3.0
    freeze_rate_s: float = 0.20  # no observed ice: nuisance scenario parameter
    melt_rate_s: float = 0.20  # no measurements above freezing
    liquid_diffusivity_scale: float = 1.0  # nuisance scenario parameter
    ice_area_exponent: float = 3.5  # attachment 1
    initial_membrane_lambda: float = 3.0  # attachment 1
    contact_ohm_m2: float = 1e-6  # 0.01 ohm cm2 from the problem statement


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
    membrane_lambda_field: np.ndarray
    membrane_lambda: np.ndarray
    plate_temperature_c: np.ndarray
    generated_water_kgm2: np.ndarray
    retained_pore_water_kgm2: np.ndarray
    membrane_sorbed_increment_kgm2: np.ndarray
    water_escaped_kgm2: np.ndarray
    mass_balance_error_kgm2: np.ndarray
    heat_generated_jm2: np.ndarray
    latent_released_jm2: np.ndarray
    heat_lost_jm2: np.ndarray
    heat_storage_jm2: np.ndarray
    energy_balance_error_jm2: np.ndarray
    min_gas_concentration_molm3: np.ndarray
    max_pore_occupancy: np.ndarray
    solver_success: bool
    solver_message: str


class ColdStartModel:
    def __init__(self, mesh_factor: int = 1):
        if mesh_factor < 1:
            raise ValueError("mesh_factor must be positive")
        layers = [
            ("aGDL", 150e-6, 8, 0.8, 0.30, 185 * 545, 1.10e-4, 8.69e-5, 2.0e-9),
            ("aCL", 3.4e-6, 3, 0.3916, 0.27, 970 * 240, 1.10e-4, 8.69e-5, 2.0e-10),
            ("PEM", 12e-6, 4, 0.0, 0.24, 2150 * 1050, 0.0, 0.0, 0.0),
            ("cCL", 11.3e-6, 5, 0.4207, 0.27, 970 * 240, 2.20e-5, 2.48e-5, 2.0e-10),
            ("cGDL", 150e-6, 8, 0.8, 0.30, 185 * 545, 2.20e-5, 2.48e-5, 2.0e-9),
        ]
        props = []
        for name, thickness, count, eps, k, cap, dgas, dv, dl in layers:
            props += [(name, thickness / (count * mesh_factor), eps, k, cap, dgas, dv, dl)] * (count * mesh_factor)
        self.names = np.array([row[0] for row in props])
        self.dx = np.array([row[1] for row in props])
        self.eps0 = np.array([row[2] for row in props])
        self.k0 = np.array([row[3] for row in props])
        self.c0 = np.array([row[4] for row in props])
        self.dgas0 = np.array([row[5] for row in props])
        self.dv0 = np.array([row[6] for row in props])
        self.dl0 = np.array([row[7] for row in props])
        self.n = len(props)
        self.length = float(np.sum(self.dx))
        self.porous = self.eps0 > 0
        self.acl = self.names == "aCL"
        self.ccl = self.names == "cCL"
        self.pem = self.names == "PEM"
        self.pem_index = np.flatnonzero(self.pem)
        self.ccl_interface = self.pem_index[-1] + 1
        self.h = 40.0
        self.plate_capacity_each = 0.002 * 1980 * 766  # J m-2 K-1
        # A finite plate/MEA interface conductance resolves plate temperature.
        # BP conductivity and half thickness alone give 95,000 W m-2 K-1;
        # 20,000 leaves a conservative allowance for interfaces.
        self.plate_contact_wm2k = 20000.0

    @staticmethod
    def _tridiagonal(diagonal, lower, upper, rhs):
        band = np.zeros((3, len(diagonal)))
        band[0, 1:] = upper
        band[1] = diagonal
        band[2, :-1] = lower
        return solve_banded((1, 1), band, rhs, check_finite=False)

    @staticmethod
    def psat_pa(temperature_k):
        tc = np.asarray(temperature_k) - T_FREEZE
        above = 611.21 * np.exp((18.678 - tc / 234.5) * tc / (257.14 + tc))
        below = 611.15 * np.exp((23.036 - tc / 333.7) * tc / (279.82 + tc))
        return np.where(tc >= 0, above, below)

    @staticmethod
    def membrane_diffusivity(temperature_k, hydration):
        """Problem statement, remark 1, equation (24), in m2/s."""
        lam = np.clip(hydration, 0.2, 22.0)
        polynomial = 2.563 - 0.33 * lam + 0.0264 * lam**2 - 0.000671 * lam**3
        return np.maximum(1e-12, 1e-10 * np.exp(2416 * (1 / 303.15 - 1 / temperature_k)) * polynomial)

    def phases(self, temperature_k, water, ice):
        eps_ice = np.maximum(ice, 0) / RHO_ICE
        free_pore = np.maximum(self.eps0 - eps_ice, 1e-8)
        msat = free_pore * MW * self.psat_pa(temperature_k) / (R * temperature_k)
        mobile = np.maximum(water - ice, 0)
        vapor = np.where(self.porous, np.minimum(mobile, msat), 0)
        liquid = np.where(self.porous, np.maximum(mobile - vapor, 0), 0)
        eps_g = np.maximum(self.eps0 - eps_ice - liquid / RHO_LIQ, 1e-8)
        return vapor, liquid, eps_ice, eps_g

    def gas_concentrations(self, temperature_k, eps_g, current_am2):
        dgas = self.dgas0 * (temperature_k / T_REF) ** 1.75 * eps_g**1.5
        c_h2 = np.full(self.n, np.nan)
        c_o2 = np.full(self.n, np.nan)
        flux = current_am2 / (2 * F)
        c = P0 / (R * temperature_k[0])
        for i in np.flatnonzero((self.names == "aGDL") | self.acl):
            d = max(dgas[i], 1e-16)
            c_h2[i] = c - flux * self.dx[i] / (2 * d)
            c -= flux * self.dx[i] / (2 * d)
            if self.acl[i]:
                flux -= current_am2 * self.dx[i] / (2 * F * np.sum(self.dx[self.acl]))
            c -= max(flux, 0) * self.dx[i] / (2 * d)
        y_o2 = (0.233 / 32) / (0.233 / 32 + 0.767 / 28)
        flux = current_am2 / (4 * F)
        c = y_o2 * P0 / (R * temperature_k[-1])
        for i in np.flatnonzero(self.ccl | (self.names == "cGDL"))[::-1]:
            d = max(dgas[i], 1e-16)
            c_o2[i] = c - flux * self.dx[i] / (2 * d)
            c -= flux * self.dx[i] / (2 * d)
            if self.ccl[i]:
                flux -= current_am2 * self.dx[i] / (4 * F * np.sum(self.dx[self.ccl]))
            c -= max(flux, 0) * self.dx[i] / (2 * d)
        return c_h2, c_o2, dgas

    def voltage(self, temperature_k, eps_ice, eps_g, current_am2, membrane_lambda, params,
                *, ice_feedback=True):
        gas_porosity = eps_g if ice_feedback else np.minimum(eps_g + eps_ice, self.eps0)
        c_h2, c_o2, dgas = self.gas_concentrations(temperature_k, gas_porosity, current_am2)
        tmean = float(np.dot(temperature_k, self.dx) / self.length)
        ch = float(np.mean(c_h2[self.acl]))
        co = float(np.mean(c_o2[self.ccl]))
        ph = max(ch * R * tmean, 1.0)
        po = max(co * R * tmean, 1.0)
        erev = 1.229 - 8.5e-4 * (tmean - T_REF) + R * tmean / (2 * F) * np.log((ph / P0) * np.sqrt(po / P0))
        pore_ice = np.clip(eps_ice[self.ccl] / self.eps0[self.ccl], 0, 0.999999)
        ice_area = float(np.mean((1 - pore_ice) ** params.ice_area_exponent)) if ice_feedback else 1.0
        j0 = params.j0_ref_am2 * np.exp(-67000 / R * (1 / tmean - 1 / T_REF))
        eta_act = R * tmean / (0.5 * F) * np.arcsinh(current_am2 / max(2 * j0 * ice_area, 1e-20))
        lam = np.maximum(membrane_lambda[self.pem], 0.3)
        kappa = (0.5139 * lam - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / temperature_k[self.pem]))
        membrane_resistance = float(np.sum(self.dx[self.pem] / np.maximum(kappa, 0.01)))
        mean_lambda = float(np.average(lam, weights=self.dx[self.pem]))
        # One effective dry-interface law; both its reference resistance and
        # hydration exponent are fitted, at 253.15 K and lambda=3.
        interface = params.interface_ohm_m2 * (params.initial_membrane_lambda / mean_lambda) ** params.interface_hydration_exponent
        interface *= np.exp(5000 / R * (1 / tmean - 1 / 253.15))
        eta_ohm = current_am2 * (membrane_resistance + params.contact_ohm_m2 + interface)
        resist = np.sum(self.dx[self.names == "cGDL"] / np.maximum(dgas[self.names == "cGDL"], 1e-16))
        resist += 0.5 * np.sum(self.dx[self.ccl] / np.maximum(dgas[self.ccl], 1e-16))
        jlim = 4 * F * max(co, 1e-8) / max(resist, 1e-12)
        eta_con = -R * tmean / (4 * F) * np.log(max(1 - current_am2 / max(jlim, 1e-9), 1e-8))
        return float(erev - eta_act - eta_ohm - eta_con), float(np.nanmin([np.nanmin(c_h2), np.nanmin(c_o2)]))

    def _sorb_from_cathode(self, water, ice, membrane_lambda, dt, params):
        """Conservative first-order uptake from the thin cathode catalyst layer.

        The exact exponential transfer fraction avoids the hard rate cap that
        made the former interfacial coefficient unidentifiable. The finite
        membrane capacity is an explicit constitutive assumption.
        """
        edge_hydration = membrane_lambda[self.pem_index[-1]]
        capacity_factor = np.clip((14.0 - edge_hydration) / (14.0 - params.initial_membrane_lambda), 0, 1)
        fraction = (1 - np.exp(-params.membrane_sorption_s * dt)) * capacity_factor
        mobile = np.maximum(water[self.ccl] - ice[self.ccl], 0)
        transfer = mobile * fraction
        updated = water.copy()
        updated[self.ccl] -= transfer
        flux = float(np.dot(transfer, self.dx[self.ccl]) / dt)
        return updated, flux

    def _pore_step(self, temperature_k, water, ice, dt, current_am2, params):
        mobile_old = np.maximum(water - ice, 0)
        _, _, eps_ice, eps_g = self.phases(temperature_k, water, ice)
        msat = np.where(self.porous, (self.eps0 - eps_ice) * MW * self.psat_pa(temperature_k) / (R * temperature_k), 0)
        dv = self.dv0 * (temperature_k / T_REF) ** 1.75 * eps_g**1.5
        dl = self.dl0 * params.liquid_diffusivity_scale
        source = np.zeros(self.n)
        source[self.ccl] = current_am2 * MW / (2 * F * np.sum(self.dx[self.ccl]))
        wet = (mobile_old >= msat) & self.porous
        for _ in range(20):
            av = (~wet & self.porous).astype(float)
            bv = np.where(wet & self.porous, msat, 0)
            al = (wet & self.porous).astype(float)
            bl = np.where(wet & self.porous, -msat, 0)
            diag = np.ones(self.n)
            lower = np.zeros(self.n - 1)
            upper = np.zeros(self.n - 1)
            rhs = mobile_old + dt * source
            for i in range(self.n - 1):
                if not (self.porous[i] and self.porous[i + 1]):
                    continue
                face_v = 2 * dv[i] * dv[i + 1] / max(dv[i] + dv[i + 1], 1e-20)
                face_l = 2 * dl[i] * dl[i + 1] / max(dl[i] + dl[i + 1], 1e-20)
                distance = (self.dx[i] + self.dx[i + 1]) / 2
                ai = (face_v * av[i] / max(eps_g[i], 1e-8) + face_l * al[i]) / distance
                aj = -(face_v * av[i + 1] / max(eps_g[i + 1], 1e-8) + face_l * al[i + 1]) / distance
                const = (face_v * (bv[i] / max(eps_g[i], 1e-8) - bv[i + 1] / max(eps_g[i + 1], 1e-8))
                         + face_l * (bl[i] - bl[i + 1])) / distance
                diag[i] += dt * ai / self.dx[i]
                upper[i] += dt * aj / self.dx[i]
                rhs[i] -= dt * const / self.dx[i]
                lower[i] -= dt * ai / self.dx[i + 1]
                diag[i + 1] -= dt * aj / self.dx[i + 1]
                rhs[i + 1] += dt * const / self.dx[i + 1]
            for i in (0, self.n - 1):
                boundary = dv[i] / max(eps_g[i], 1e-8) / (self.dx[i] / 2)
                diag[i] += dt * boundary * av[i] / self.dx[i]
                rhs[i] -= dt * boundary * bv[i] / self.dx[i]
            candidate = self._tridiagonal(diag, lower, upper, rhs)
            new_wet = (candidate >= msat) & self.porous
            if np.array_equal(new_wet, wet):
                break
            wet = new_wet
        else:
            raise RuntimeError("Pore-water phase active set failed to converge")
        if np.min(candidate[self.porous]) < -1e-8:
            raise RuntimeError("Negative mobile pore water")
        mobile = np.maximum(candidate, 0)
        vapor = np.where(self.porous, np.minimum(mobile, msat), 0)
        liquid = np.where(self.porous, np.maximum(mobile - vapor, 0), 0)
        escaped_rate = dv[0] * vapor[0] / max(eps_g[0], 1e-8) / (self.dx[0] / 2)
        escaped_rate += dv[-1] * vapor[-1] / max(eps_g[-1], 1e-8) / (self.dx[-1] / 2)
        new_water = np.where(self.porous, ice + mobile, 0)
        return new_water, escaped_rate

    def _phase_change_step(self, temperature_k, water, ice, dt, params):
        _, liquid, eps_ice, _ = self.phases(temperature_k, water, ice)
        cold = np.clip((T_FREEZE - temperature_k) / 20, 0, 3)
        hot = np.clip((temperature_k - T_FREEZE) / 5, 0, 3)
        freeze_fraction = 1 - np.exp(-dt * params.freeze_rate_s * cold)
        melt_fraction = 1 - np.exp(-dt * params.melt_rate_s * hot)
        capacity = np.maximum((self.eps0 - eps_ice) * RHO_ICE, 0)
        freeze_mass = np.minimum(liquid * freeze_fraction, capacity)
        melt_mass = np.maximum(ice, 0) * melt_fraction
        new_ice = np.where(self.porous, ice + freeze_mass - melt_mass, 0)
        return new_ice, (freeze_mass - melt_mass) / dt

    def _membrane_step(self, temperature_k, membrane_lambda, dt, current_am2, uptake_rate, params):
        idx = self.pem_index
        lam = membrane_lambda[idx]
        q = lam * WATER_PER_LAMBDA
        d = self.membrane_diffusivity(temperature_k[idx], lam)
        diag = np.ones(len(idx))
        lower = np.zeros(len(idx) - 1)
        upper = np.zeros(len(idx) - 1)
        rhs = q.copy()
        rhs[-1] += dt * uptake_rate / self.dx[idx[-1]]
        # Electro-osmotic drag follows remark 1, eqs (19)-(20), on internal
        # PEM faces. External transfer uses the separate CL sorption boundary.
        for k in range(len(idx) - 1):
            face = 2 * d[k] * d[k + 1] / max(d[k] + d[k + 1], 1e-20)
            face /= (self.dx[idx[k]] + self.dx[idx[k + 1]]) / 2
            diag[k] += dt * face / self.dx[idx[k]]
            upper[k] -= dt * face / self.dx[idx[k]]
            lower[k] -= dt * face / self.dx[idx[k + 1]]
            diag[k + 1] += dt * face / self.dx[idx[k + 1]]
            n_drag = 2.5 * (lam[k] + lam[k + 1]) / (2 * 22)
            drag = n_drag * MW * current_am2 / F
            rhs[k] -= dt * drag / self.dx[idx[k]]
            rhs[k + 1] += dt * drag / self.dx[idx[k + 1]]
        updated = self._tridiagonal(diag, lower, upper, rhs) / WATER_PER_LAMBDA
        if np.min(updated) < 0.3 or np.max(updated) > 22:
            raise RuntimeError("Membrane hydration left physical range")
        result = membrane_lambda.copy()
        result[idx] = updated
        return result

    def _heat_step(self, temperature_k, plate_k, water, ice, ice_rate, membrane_lambda,
                   dt, current_am2, ambient_k, params, ice_feedback):
        _, liquid, eps_ice, eps_g = self.phases(temperature_k, water, ice)
        v, _ = self.voltage(temperature_k, eps_ice, eps_g, current_am2, membrane_lambda, params,
                            ice_feedback=ice_feedback)
        k = self.k0 + eps_ice * (2.3 - 0.024) + liquid / RHO_LIQ * (0.6 - 0.024)
        area_cap = np.r_[params.plate_capacity_scale * self.plate_capacity_each,
                         self.c0 * self.dx,
                         params.plate_capacity_scale * self.plate_capacity_each]
        old = np.r_[plate_k[0], temperature_k, plate_k[1]]
        diag = area_cap.copy()
        lower = np.zeros(self.n + 1)
        upper = np.zeros(self.n + 1)
        rhs = area_cap * old
        rhs[1:-1] += dt * (current_am2 * (ETH - v) * self.dx / self.length
                             + LATENT_FREEZE * ice_rate * self.dx)
        conductances = np.empty(self.n + 1)
        conductances[0] = 1 / (1 / self.plate_contact_wm2k + self.dx[0] / (2 * k[0]))
        conductances[-1] = 1 / (1 / self.plate_contact_wm2k + self.dx[-1] / (2 * k[-1]))
        for i in range(self.n - 1):
            conductances[i + 1] = 1 / (self.dx[i] / (2 * k[i]) + self.dx[i + 1] / (2 * k[i + 1]))
        for i, g in enumerate(conductances):
            diag[i] += dt * g
            upper[i] -= dt * g
            lower[i] -= dt * g
            diag[i + 1] += dt * g
        for i in (0, self.n + 1):
            diag[i] += dt * self.h
            rhs[i] += dt * self.h * ambient_k
        updated = self._tridiagonal(diag, lower, upper, rhs)
        generated_flux = current_am2 * (ETH - v)
        latent_flux = LATENT_FREEZE * float(np.dot(ice_rate, self.dx))
        lost_flux = self.h * (updated[0] + updated[-1] - 2 * ambient_k)
        return updated[1:-1], updated[[0, -1]], generated_flux, latent_flux, lost_flux

    def simulate(self, experiment: Experiment, params: Parameters, *, max_step=0.05,
                 freezing=True, ice_feedback=True) -> Simulation:
        times = experiment.time_s
        nt, n = len(times), self.n
        temp = np.empty((nt, n))
        water = np.zeros((nt, n))
        ice = np.zeros((nt, n))
        hydration = np.zeros((nt, n))
        plates = np.empty((nt, 2))
        generated = np.zeros(nt)
        escaped = np.zeros(nt)
        sorbed = np.zeros(nt)
        q_generated = np.zeros(nt)
        q_latent = np.zeros(nt)
        q_lost = np.zeros(nt)
        q_storage = np.zeros(nt)
        temp[0] = experiment.temperature_c[0] + T_FREEZE
        plates[0] = temp[0, 0]
        hydration[0, self.pem] = params.initial_membrane_lambda
        initial_membrane_mass = float(np.dot(hydration[0, self.pem] * WATER_PER_LAMBDA, self.dx[self.pem]))
        ambient_k = experiment.ambient_c + T_FREEZE
        for m in range(1, nt):
            current_t = temp[m - 1].copy()
            current_w = water[m - 1].copy()
            current_i = ice[m - 1].copy()
            current_lam = hydration[m - 1].copy()
            current_plate = plates[m - 1].copy()
            steps = int(np.ceil((times[m] - times[m - 1]) / max_step))
            dt = (times[m] - times[m - 1]) / steps
            for sub in range(steps):
                tmid = times[m - 1] + (sub + 0.5) * dt
                j = float(np.interp(tmid, times, experiment.current_density_acm2)) * 1e4
                production = j * MW / (2 * F)
                current_w, outflow = self._pore_step(
                    current_t, current_w, current_i, dt, j, params)
                current_w, uptake = self._sorb_from_cathode(current_w, current_i, current_lam, dt, params)
                if freezing:
                    current_i, ice_rate = self._phase_change_step(current_t, current_w, current_i, dt, params)
                else:
                    ice_rate = np.zeros(n)
                current_lam = self._membrane_step(current_t, current_lam, dt, j, uptake, params)
                current_t, current_plate, gen_flux, latent_flux, lost_flux = self._heat_step(
                    current_t, current_plate, current_w, current_i, ice_rate, current_lam,
                    dt, j, ambient_k, params, ice_feedback)
                generated[m] += production * dt
                escaped[m] += outflow * dt
                sorbed[m] += uptake * dt
                q_generated[m] += gen_flux * dt
                q_latent[m] += latent_flux * dt
                q_lost[m] += lost_flux * dt
            for series in (generated, escaped, sorbed, q_generated, q_latent, q_lost):
                series[m] += series[m - 1]
            temp[m], water[m], ice[m], hydration[m], plates[m] = (
                current_t, current_w, current_i, current_lam, current_plate)
            cap = np.r_[params.plate_capacity_scale * self.plate_capacity_each,
                        self.c0 * self.dx,
                        params.plate_capacity_scale * self.plate_capacity_each]
            q_storage[m] = float(np.dot(cap, np.r_[current_plate[0] - plates[0, 0],
                                                      current_t - temp[0], current_plate[1] - plates[0, 1]]))
        avg_temp = temp @ self.dx / self.length - T_FREEZE
        avg_lambda = hydration[:, self.pem] @ self.dx[self.pem] / np.sum(self.dx[self.pem])
        voltage = np.empty(nt)
        max_ice = np.empty(nt)
        max_sat = np.empty(nt)
        min_gas = np.empty(nt)
        occupancy = np.empty(nt)
        for m in range(nt):
            _, liquid, eps_ice, eps_g = self.phases(temp[m], water[m], ice[m])
            j = experiment.current_density_acm2[m] * 1e4
            voltage[m], min_gas[m] = self.voltage(temp[m], eps_ice, eps_g, j, hydration[m],
                                                   params, ice_feedback=ice_feedback)
            max_ice[m] = np.max(eps_ice)
            max_sat[m] = np.max(eps_ice[self.porous] / self.eps0[self.porous])
            occupancy[m] = np.max((eps_ice[self.porous] + liquid[self.porous] / RHO_LIQ) / self.eps0[self.porous])
        retained = water @ self.dx
        membrane_increment = hydration[:, self.pem] @ (self.dx[self.pem] * WATER_PER_LAMBDA) - initial_membrane_mass
        balance = generated - retained - escaped - membrane_increment
        energy_balance = q_generated + q_latent - q_lost - q_storage
        return Simulation(times, avg_temp, voltage, max_ice, max_sat, temp - T_FREEZE,
                          water, ice, hydration, avg_lambda, plates - T_FREEZE,
                          generated, retained, membrane_increment, escaped, balance,
                          q_generated, q_latent, q_lost, q_storage, energy_balance,
                          min_gas, occupancy, True, "Implicit finite-volume solve completed")

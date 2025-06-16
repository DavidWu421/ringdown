from .coefficients import *
from ..kerr import model as kerrmod
from ... import qnms
import numpyro.distributions as dist
import numpyro

jax.config.update("jax_enable_x64", True)

mode_to_index = {
    (1, -2, 2, 0, 0): 0,
    (1, -2, 2, 0, 1): 1,
    (1, -2, 2, 1, 0): 2,
    (1, -2, 2, 1, 1): 3,
    (1, -2, 2, 2, 0): 4,
    (1, -2, 2, 2, 1): 5,
    (1, -2, 2, 2, 2): 6,
    (1, -2, 3, 2, 0): 7,
    (1, -2, 3, 2, 1): 8,
    (1, -2, 3, 3, 0): 9,
    (1, -2, 3, 3, 1): 10,
    (1, -2, 4, 2, 0): 11,
    (1, -2, 4, 4, 0): 12
}


class KerrNewman:

    def __init__(self, modes):
        self.modes = modes
        self.kerr_modes = kerrmod.Kerr(modes)
        self.indices = []
        for mode in self.modes:
            if mode in mode_to_index.keys():
                self.indices.append(mode_to_index[mode])
            else:
                raise ValueError(
                    f"Don't have computed KerrNewman modes for mode {mode}")

        self.aij_omega = jnp.stack([a_omega[i] for i in self.indices])
        self.aij_gamma = jnp.stack([a_gamma[i] for i in self.indices])
        self.bij_omega = jnp.stack([b_omega[i] for i in self.indices])
        self.bij_gamma = jnp.stack([b_gamma[i] for i in self.indices])

        self.prior_kwargs = {
            'm_min': 40,
            'm_max': 200,
            'chi_min': 0.0,
            'chi_max': 1.0,
            'Q_min': 0.0,
            'Q_max': 1.0,
            'phi_chiQ_min': 0.0,
            'phi_chiQ_max': jnp.pi / 2,
            'half_r_squared_chiQ_min': 0.0,
            'half_r_squared_chiQ_max': 1 / 2
        }

    @property
    def prior_parameters(self):
        return self.prior_kwargs.keys()

    def get_key_from_value(self, value):
        for k, v in mode_to_index.items():
            if v == value:
                return k
        raise ValueError(f"Value {value} not found in the dictionary.")

    def get_kerr_omega(self, chi):
        omega = 2 * jnp.pi * kerrmod.chi_factors(chi, self.kerr_modes.fcoeffs)
        gamma = kerrmod.chi_factors(chi, self.kerr_modes.gcoeffs)
        return omega, gamma

    def get_charged_omega(self, chi, Q):
        omega_kerr, gamma_kerr = self.get_kerr_omega(chi)
        omega = omega_kerr + Q**2 * compute_shifts(chi, self.aij_omega,
                                                   self.bij_omega).flatten()
        gamma = (gamma_kerr - Q**2 *
                 compute_shifts(chi, self.aij_gamma, self.bij_gamma).flatten())

        return omega, gamma

    def get_charged_ftau(self, M, chi, Q):
        FREF = 2985.668287014743
        MREF = 68.0
        f0 = FREF * MREF / M
        omega0, gamma0 = self.get_charged_omega(chi, Q)
        omega_c = f0 * self.get_charged_omega(chi, Q)
        freqs = f0 * omega0 / (2 * jnp.pi)
        taus = -1.0 / gamma0
        return freqs, taus

    def r2_phi_to_Q_chi(self, half_r_squared_Qchi, phi_Qchi):
        r = jnp.sqrt(2 * half_r_squared_Qchi)
        chi = r * jnp.cos(phi_Qchi)
        Q = r * jnp.sin(phi_Qchi)
        return chi, Q

    def get_freqs_and_gammas(self, m, chi, Q):
        f0 = 1 / (m * qnms.T_MSUN)
        f_gr_0, g_gr_0 = self.get_charged_omega(chi, Q)
        f_gr = f0 * f_gr_0 / (2 * jnp.pi)
        g_gr = f0 * g_gr_0
        return f_gr, g_gr

    def prior_sample(self):
        variables = {}
        for var in ['m', 'half_r_squared_chiQ', 'phi_chiQ']:
            variable_prior_distribution = dist.Uniform(
                self.prior_kwargs[f'{var}_min'],
                self.prior_kwargs[f'{var}_max'])
            variables[var] = numpyro.sample(var, variable_prior_distribution)

        m = variables['m']
        half_r_squared_chiQ = variables['half_r_squared_chiQ']
        phi_chiQ = variables['phi_chiQ']

        r_chiQ = jnp.sqrt(2 * half_r_squared_chiQ)
        chi_value = r_chiQ * jnp.cos(phi_chiQ)
        Q_value = r_chiQ * jnp.sin(phi_chiQ)

        chi = numpyro.deterministic("chi", chi_value)
        Q = numpyro.deterministic("Q", Q_value)

        return {'m': m, 'chi': chi, 'Q': Q}

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
    (1, -2, 4, 4, 0): 12,
    (1, -2, -2, 0, 0): 13,
    (1, -2, -2, 0, 1): 14,
    (1, -2, -2, 1, 0): 15,
    (1, -2, -2, 1, 1): 16,
    (1, -2, -2, 2, 0): 17,
    (1, -2, -2, 2, 1): 18,
    (1, -2, -2, 2, 2): 19,
    (1, -2, -3, 2, 0): 20,
    (1, -2, -3, 2, 1): 21,
    (1, -2, -3, 3, 0): 22,
    (1, -2, -3, 3, 1): 23,
    (1, -2, -4, 2, 0): 24,
    (1, -2, -4, 4, 0): 25
}


class OddCubic:

    def __init__(self, modes):
        self.modes = modes
        self.kerr_modes = kerrmod.Kerr(modes)
        self.indices = []
        for mode in self.modes:
            if mode in mode_to_index.keys():
                self.indices.append(mode_to_index[mode])
            else:
                raise ValueError(
                    f"Don't have computed OddCubic modes for mode {mode}")

        self.aij_omega = jnp.stack([a_omega[i] for i in self.indices])
        self.aij_gamma = jnp.stack([a_gamma[i] for i in self.indices])

        self.prior_kwargs = {
            'm_min': 40,
            'm_max': 200,
            'chi_min': 0.0,
            'chi_max': 1.0,
            'alpha_min': 0.0,
            'alpha_max': 0.1,
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

    def get_odd_cubic_omega(self, chi, alpha):
        omega_kerr, gamma_kerr = self.get_kerr_omega(chi)
        omega = omega_kerr + alpha * compute_shifts(chi, self.aij_omega).flatten()
        gamma = (gamma_kerr + alpha *
                 compute_shifts(chi, self.aij_gamma).flatten())

        return omega, gamma

    def get_charged_ftau(self, M, chi, alpha):
        FREF = 2985.668287014743
        MREF = 68.0
        f0 = FREF * MREF / M
        omega0, gamma0 = self.get_odd_cubic_omega(chi, alpha)
        omega_c = f0 * self.get_odd_cubic_omega(chi, alpha)
        freqs = f0 * omega0 / (2 * jnp.pi)
        taus = -1.0 / gamma0
        return freqs, taus

    def get_freqs_and_gammas(self, m, chi, alpha):
        f0 = 1 / (m * qnms.T_MSUN)
        f_gr_0, g_gr_0 = self.get_odd_cubic_omega(chi, alpha)
        f_gr = f0 * f_gr_0 / (2 * jnp.pi)
        g_gr = f0 * g_gr_0
        return f_gr, g_gr

    def prior_sample(self):
        variables = {}
        for var in ['m', 'chi', 'alpha']:
            variable_prior_distribution = dist.Uniform(
                self.prior_kwargs[f'{var}_min'],
                self.prior_kwargs[f'{var}_max'])
            variables[var] = numpyro.sample(var, variable_prior_distribution)

        m = variables['m']
        chi = variables['chi']
        alpha = variables['alpha']
        return {'m': m, 'chi': chi, 'alpha': alpha}

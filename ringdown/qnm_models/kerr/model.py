import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from ... import qnms

def chi_factors(chi, coeffs):
    log_1m_chi = jnp.log1p(-chi)
    log_1m_chi_2 = log_1m_chi * log_1m_chi
    log_1m_chi_3 = log_1m_chi_2 * log_1m_chi
    log_1m_chi_4 = log_1m_chi_2 * log_1m_chi_2
    log_sqrt_1m_chi2 = 0.5 * jnp.log1p(-(chi**2))
    log_sqrt_1m_chi2_2 = log_sqrt_1m_chi2 * log_sqrt_1m_chi2
    log_sqrt_1m_chi2_3 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2
    log_sqrt_1m_chi2_4 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2_2
    log_sqrt_1m_chi2_5 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_2
    log_sqrt_1m_chi2_6 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_3

    v = jnp.stack(
        [
            1.0,
            log_1m_chi,
            log_1m_chi_2,
            log_1m_chi_3,
            log_1m_chi_4,
            log_sqrt_1m_chi2,
            log_sqrt_1m_chi2_2,
            log_sqrt_1m_chi2_3,
            log_sqrt_1m_chi2_4,
            log_sqrt_1m_chi2_5,
            log_sqrt_1m_chi2_6,
        ]
    )

    return jnp.dot(coeffs, v)

class Kerr:
	def __init__(self, modes):
		self.modes = modes
		self.fcoeffs = []
		self.gcoeffs = []
		for mode in self.modes:
		    c = qnms.KerrMode(mode).coefficients
		    self.fcoeffs.append(c[0])
		    self.gcoeffs.append(c[1])
		self.fcoeffs = jnp.array(self.fcoeffs)
		self.gcoeffs = jnp.array(self.gcoeffs)

		self.prior_kwargs = {'m_min' : 40, 'm_max' : 200, 
							 'chi_min' : 0.0,  'chi_max' : 1.0,
							 'Q_min' : 0.0, 'Q_max' : 1.0,
							 'phi_chiQ_min' : 0.0, 'phi_chiQ_max' : jnp.pi/2,
							 'half_r_squared_chiQ_min' : 0.0, 'half_r_squared_chiQ_max' : 1/2}

	@property
	def prior_parameters(self):
		return self.prior_kwargs.keys()

	def get_freqs_and_gammas(self, m, chi, Q):
		f0 = 1 / (m * qnms.T_MSUN)
		f_gr = f0 * chi_factors(chi, fcoeffs)
		g_gr = f0 * chi_factors(chi, gcoeffs)
		return f_gr, g_gr

	def prior_sample(self):
		variables = {}
		for var in ['m', 'chi']:
			variable_prior_distribution = dist.Uniform(self.prior_kwargs[f'{var}_min'], self.prior_kwargs[f'{var}_max'])
			variables[var] = numpyro.sample(var, variable_prior_distribution)
		
		m = variables['m']
		chi = variables['chi']
		return {'m' : m, 'chi' : chi}
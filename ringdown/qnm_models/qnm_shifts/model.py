import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from ... import qnms
from ... import config

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

def compute_coefficients_for_mode_shifts(chis, real_delta_omega, imag_delta_omega):
        #p, s, l, m, n = mode
        #sgn = p if m == 0 else p * np.sign(m)
        #q = qnm.modes_cache(s, l, p * abs(m), n)

        # Only use spins pre-computed by qnm package
        chis = np.array(chis)
        log_1m_chis = np.log1p(-chis)
        log_1m_chis_2 = log_1m_chis * log_1m_chis
        log_1m_chis_3 = log_1m_chis_2 * log_1m_chis
        log_1m_chis_4 = log_1m_chis_2 * log_1m_chis_2
        log_sqrt_1m_chis2 = 0.5 * np.log1p(-(chis**2))
        log_sqrt_1m_chis2_2 = log_sqrt_1m_chis2 * log_sqrt_1m_chis2
        log_sqrt_1m_chis2_3 = log_sqrt_1m_chis2_2 * log_sqrt_1m_chis2
        log_sqrt_1m_chis2_4 = log_sqrt_1m_chis2_2 * log_sqrt_1m_chis2_2
        log_sqrt_1m_chis2_5 = log_sqrt_1m_chis2_3 * log_sqrt_1m_chis2_2
        log_sqrt_1m_chis2_6 = log_sqrt_1m_chis2_3 * log_sqrt_1m_chis2_3

        M = np.column_stack(
            (
                np.ones_like(log_1m_chis),
                log_1m_chis,
                log_1m_chis_2,
                log_1m_chis_3,
                log_1m_chis_4,
                log_sqrt_1m_chis2,
                log_sqrt_1m_chis2_2,
                log_sqrt_1m_chis2_3,
                log_sqrt_1m_chis2_4,
                log_sqrt_1m_chis2_5,
                log_sqrt_1m_chis2_6,
            )
        )

        f = sgn * np.array([real_delta_omega[i] for i in range(len(chis))]) / (2 * np.pi)
        g = np.array([imag_delta_omega[i] for i in range(len(chis))])

        coeff_f = np.linalg.lstsq(M, f, rcond=None, **kws)[0]
        coeff_g = np.linalg.lstsq(M, g, rcond=None, **kws)[0]

        return coeff_f, coeff_g

class ShiftedKerrMode:
	def __init__(self, p,s,l,m,n, csv_file, 
					   dictionary_of_columns={'chis' : 'a',  
					   						  'real_delta_omega' : 'real_delta_omega',
					   						  'imag_delta_omega' : 'imag_delta_omega'}):
		self.mode = indexing.HarmonicIndex.construct(p=p,s=s,l=l,m=m,n=n);
		self.kerr_mode = qnms.KerrMode(p=p,s=s,l=l,m=m,n=n);
		self.dictionary_of_columns = dictionary_of_columns
		import pandas as pd
		kerr_shifts = pd.read_csv(csv_file);
		self.chis = kerr_shifts[self.dictionary_of_columns['chis']]
		self.real_delta_omega = kerr_shifts[self.dictionary_of_columns['real_delta_omega']]
		self.imag_delta_omega = kerr_shifts[self.dictionary_of_columns['imag_delta_omega']]
		self._coefficients = None

	@property
	def coefficients(self):
		if self._coefficients:
			return self._coefficients

		self._coefficients = compute_coefficients_for_mode_shifts(self.chis, self.real_delta_omega, self.imag_delta_omega)

	def dfdgamma(self, chi, m_msun=None):
            log_1m_chi = np.log1p(-chi)
            log_1m_chi_2 = log_1m_chi * log_1m_chi
            log_1m_chi_3 = log_1m_chi_2 * log_1m_chi
            log_1m_chi_4 = log_1m_chi_2 * log_1m_chi_2
            log_sqrt_1m_chi2 = 0.5 * np.log1p(-(chi**2))
            log_sqrt_1m_chi2_2 = log_sqrt_1m_chi2 * log_sqrt_1m_chi2
            log_sqrt_1m_chi2_3 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2
            log_sqrt_1m_chi2_4 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2_2
            log_sqrt_1m_chi2_5 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_2
            log_sqrt_1m_chi2_6 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_3

            v = np.stack(
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

            f, g = [np.dot(coeff, v) for coeff in self.coefficients]
        if m_msun is not None:
            f /= m_msun * config.T_MSUN
            g /= m_msun * config.T_MSUN
        return f, g

    def fgamma(self, chi, Q=0.0, m_msun=None):
    	f0, g0 = self.kerr_mode.fgamma(chi)
        df, dg = self.dfdgamma(chi)

        f = f0 + Q**2 * df;
        g = g0 + Q**2 * dg;

        if m_msun is not None:
            f /= m_msun * T_MSUN
            g /= m_msun * T_MSUN
        return f, g

	def __call__(self, chi, Q=0.0, m_msun=None):
        f,g = self.fgamma(self, chi, Q=0.0, m_msun=None)
        omega = 2 * np.pi * f - 1j * g
        return omega



class ShiftedQNMs:
	def __init__(self, modes):
		self.modes = modes
		self.fcoeffs = []
		self.gcoeffs = []
		for mode in self.modes:
		    c =  mode.kerr_mode.coefficients
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
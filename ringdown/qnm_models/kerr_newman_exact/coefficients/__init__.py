import jax.numpy as jnp
from jax import vmap

"""
Fitting Coefficients have been taken from Carulo et.al (April 2022)
https://arxiv.org/pdf/2109.13961.pdf
"""

def _to_jnp(arr): return jnp.array(arr, dtype=jnp.float32)

b_omega = [_to_jnp(x) for x in [
    [[1.0, 0.537583, -2.990402, 1.503421],
     [-1.899567, -2.128633, 6.626680, -2.903790],
     [1.015454, 2.147094, -4.672847, 1.891731],
     [-0.111430, -0.581706, 1.021061, -0.414517]],
    
    [[1.0, -2.918987, 2.866252, -0.944554],
     [-1.850299, 7.321955, -8.783456, 3.292966],
     [0.944088, -5.584876, 7.675096, -3.039132],
     [-0.088458, 1.198758, -1.973222, 0.838109]]
]]

c_omega = [_to_jnp(x) for x in [
    [[1.0, 0.548651, -3.141145, 1.636377],
     [-2.238461, -2.291933, 7.695570, -3.458474],
     [1.581677, 2.662938, -6.256090, 2.494264],
     [-0.341455, -0.930069, 1.688288, -0.612643]],
    
    [[1.0,-2.941138, 2.907859, -0.964407],
     [-2.250169, 8.425183, -9.852886, 3.660289],
     [1.611393, -7.869432, 9.999751, -3.737205],
     [-0.359285, 2.392321, -3.154979, 1.129776]]
]]

b_gamma = [_to_jnp(x) for x in [
    [[1.0, -2.721789, 2.472860, -0.750015],
     [-2.533958, 7.181110, -6.870324, 2.214689],
     [2.102750, -6.317887, 6.206452, -1.980749],
     [-0.568636, 1.857404, -1.820547, 0.554722]],
    
    [[1.0, -3.074983, 3.182195, -1.105297],
     [0.366066, 4.296285, -9.700146, 5.016955],
     [-3.290350, -0.844265, 9.999863, -5.818349],
     [1.927196, -0.401520, -3.537667, 2.077991]]
]]

c_gamma = [_to_jnp(x) for x in [
    [[1.0,-2.732346, 2.495049, -0.761581],
     [-2.498341, 7.089542, -6.781334, 2.181880],
     [2.056918, -6.149334, 6.010021, -1.909275],
     [-0.557557, 1.786783, -1.734461, 0.524997]],
    
    [[1.0, -3.079686, 3.191889, -1.110140],
     [0.388928, 4.159242, -9.474149, 4.904881],
     [-3.119527, -0.914668, 9.767356, -5.690517],
     [1.746957, -0.240680, -3.505359, 2.049254]]
]]

Y0_omega = jnp.array([0.37367168, 0.34671099], dtype=jnp.float32)
Y0_gamma = jnp.array([0.08896232, 0.27391488], dtype=jnp.float32)

Y0_bij_omega = jnp.stack([Y0_omega[i] * b_omega[i] for i in range(2)])
Y0_bij_gamma = jnp.stack([Y0_gamma[i] * b_gamma[i] for i in range(2)])
cij_omega = jnp.stack(c_omega)
cij_gamma = jnp.stack(c_gamma)

# Vectorized basis term computation
def get_chiQ_basis(chi, Q):
    chi_powers = chi ** jnp.arange(4)[:, None]
    Q_powers = Q ** jnp.arange(4)[None, :]
    return chi_powers * Q_powers  # shape (4, 4)

# Main function for computing omega/gamma values
def jnp_chiq_exact_factors(chi, Q, Y0_bij, cij):
    chi_Q = get_chiQ_basis(chi, Q)  # shape (4, 4)
    num = jnp.einsum('mij,ij->m', Y0_bij, chi_Q)
    den = jnp.einsum('mij,ij->m', cij, chi_Q)
    return num / den

def get_charged_omega(chi, Q):
    omega = jnp_chiq_exact_factors(chi, Q, Y0_bij_omega, cij_omega)
    gamma = jnp_chiq_exact_factors(chi, Q, Y0_bij_gamma, cij_gamma)
    return omega - 1j * gamma

def get_charged_ftau(M, chi, Q):
    FREF = 2985.668287014743
    MREF = 68.0
    f0 = FREF * MREF / M
    omega_c = f0 * get_charged_omega(chi, Q)
    freqs = jnp.real(omega_c) / (2 * jnp.pi)
    taus = -1.0 / jnp.imag(omega_c)
    return freqs, taus
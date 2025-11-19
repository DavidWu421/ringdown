import jax
import jax.numpy as jnp
from jax import vmap
import qnm
import os

jax.config.update("jax_enable_x64", True)


def _to_jnp(arr):
    return jnp.array(arr, dtype=jnp.float64)


# Fitting \omega = \Omega - i \gamma with the form: 
# \Omega= \sum_{j=1}^5 (a_{omega})_j*\sqrt{1-\chi^2}^j + (b_{omega})_j \ln(\sqrt{1-\chi^2})^j
# -\gamma= \sum_{j=1}^5 (a_{gamma})_j*\sqrt{1-\chi^2}^j + (b_{gamma})_j \ln(\sqrt{1-\chi^2})^j. 
# Coefficients are in groups of 5 where each group of 5 corresponds to each mode
#  in the order of [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], 
# [2, 2, 1], [2, 2, 2], [3, 2, 0], [3, 2, 1], [3, 3, 0], [3, 3, 1], [4, 2, 0], [4, 4, 0]]

mode_list = jnp.array([[2, 0, 0,"+"], [2, 0, 1,"+"], [2, 1, 0,"+"], [2, 1, 1,"+"], [2, 2, 0,"+"],
                       [2, 2, 1,"+"], [2, 2, 2,"+"], [3, 2, 0,"+"], [3, 2, 1,"+"], [3, 3, 0,"+"],
                       [3, 3, 1,"+"], [4, 2, 0,"+"], [4, 4, 0,"+"],
                       [2, 0, 0,"-"], [2, 0, 1,"-"], [2, 1, 0,"-"], [2, 1, 1,"-"], [2, 2, 0,"-"],
                       [2, 2, 1,"-"], [2, 2, 2,"-"], [3, 2, 0,"-"], [3, 2, 1,"-"], [3, 3, 0,"-"],
                       [3, 3, 1,"-"], [4, 2, 0,"-"], [4, 4, 0,"-"]])

folder = "../Fits"

files = os.listdir(folder)

def get_cubic_coeffs_from_file(filename):
    real_coeffs = []
    imag_coeffs = []
    with open(filename, "r") as f:
        for line in f:
            # skip blank lines
            if not line.strip():
                continue

            cols = line.split()   # split on whitespace (tabs or spaces)
            real_coeffs.append(float(cols[1]))   # second column (index 1)
            imag_coeffs.append(float(cols[2]))   # third column (index 2)
    return real_coeffs, imag_coeffs

def get_cubic_mode_coeffs(mode):
    l=mode[0]
    m=mode[1]
    n=mode[2]
    paritysign=mode[3]
    if paritysign=="+":
        parity="even"
    elif paritysign=="-":
        parity="odd"
    filename="../Fits/cubic_even"+parity+"l"+str(l)+"_m"+str(m)+"_n"+str(n)+".txt"
    return get_cubic_coeffs_from_file(filename)

real_mode_coeffs=[]
imag_mode_coeffs=[]
for mode in mode_list:
    real_coeffs,imag_coeffs = get_cubic_mode_coeffs(mode)
    real_mode_coeffs.append(real_coeffs)
    imag_mode_coeffs.append(imag_coeffs)

# KN Shift fits
a_omega = [
    _to_jnp(x)
    for x in [real_mode_coeffs]
]
a_gamma = [
    _to_jnp(x)
    for x in [imag_mode_coeffs]
]

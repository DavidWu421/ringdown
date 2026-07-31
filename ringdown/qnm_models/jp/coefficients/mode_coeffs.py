import jax
import jax.numpy as jnp
from jax import vmap
import qnm
import os
from pathlib import Path

jax.config.update("jax_enable_x64", True)


def _to_jnp(arr):
    return jnp.array(arr, dtype=jnp.float64)


# Fitting \omega = \Omega - i \gamma with the form: 
# \Omega= \sum_{j=1}^16 (Re[a_{omega}])_j*\chi^j and -\gamma= \sum_{j=1}^16 (Im[a_{omega}])_j*\chi^j. 
# Coefficients are in groups of 16 where each group of 16 corresponds to each mode
#  in the order of [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], 
# [2, 2, 1], [2, 2, 2], [3, 2, 0], [3, 2, 1], [3, 3, 0], [3, 3, 1], [4, 2, 0], [4, 4, 0]].
# The last number is 0 for even parity and 1 for odd parity.

mode_list = jnp.array([[2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0],
                       [2, 2, 1], [2, 2, 2], [3, 2, 0], [3, 2, 1], [3, 3, 0],
                       [3, 3, 1], [4, 2, 0], [4, 4, 0],
                       [-2, 0, 0], [-2, 0, 1], [-2, 1, 0], [-2, 1, 1], [-2, 2, 0],
                       [-2, 2, 1], [-2, 2, 2], [-3, 2, 0], [-3, 2, 1], [-3, 3, 0],
                       [-3, 3, 1], [-4, 2, 0], [-4, 4, 0]])

# 

# Robust reference to the 'Fits' directory relative to this module
folder = Path(__file__).parent.parent / "Fits"

# Optional: check it exists
if not folder.exists():
    raise FileNotFoundError(f"Fits directory not found at {folder}")

# JP fit coefficient file
filename = Path(__file__).parent.parent / "Fits" / "JPFitsCoeffs.csv"

mode_fit_dict={}
real_coeffs_even = []
imag_coeffs_even = []
real_coeffs_odd = []
imag_coeffs_odd = []
with open(filename, "r") as f:
    next(f)  # skip header
    for i, line in enumerate(f):
        entries = [float(x) for x in line.strip().split(",")[5:]]
        key = "".join(line.strip().split(",")[:5])
        if i % 4 == 0:
            # real_coeffs_even.append(entries)
            mode_fit_dict[key]=entries
        elif i % 4 == 1:
            # imag_coeffs_even.append(entries)
            mode_fit_dict[key]=entries
        elif i % 4 == 2:
            # real_coeffs_odd.append(entries)
            mode_fit_dict[key]=entries
        elif i % 4 == 3:
            # imag_coeffs_odd.append(entries)
            mode_fit_dict[key]=entries


def get_jp_mode_coeffs(mode):
    """
    mode: [l, m, n]
    parity: l>0 for even, l<0 for odd

    """
    l, m, n= mode

    # Convert parity sign to file naming
    if l > 0:
        parity = "Even"
    elif l< 1:
        parity = "Odd"
    else:
        raise ValueError(f"Invalid parity: {l}, expected l>0 or l<0")

    real_key= str(abs(l))+str(m)+str(n)+parity+"Re"
    imag_key= str(abs(l))+str(m)+str(n)+parity+"Im"

    # Call the file-reading function
    return mode_fit_dict[real_key], mode_fit_dict[imag_key]

real_mode_coeffs=[]
imag_mode_coeffs=[]
for mode in mode_list:
    real_coeffs,imag_coeffs = get_jp_mode_coeffs(mode)
    real_mode_coeffs.append(real_coeffs)
    imag_mode_coeffs.append(imag_coeffs)

# odd cubic Shift fits
a_omega = [
    _to_jnp(x)
    for x in real_mode_coeffs
]
a_gamma = [
    _to_jnp(x)
    for x in imag_mode_coeffs
]

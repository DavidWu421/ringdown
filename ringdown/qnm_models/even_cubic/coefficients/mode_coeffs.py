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

# List all files in the Fits directory
files = list(folder.iterdir())  # returns Path objects

def get_even_cubic_coeffs_from_file(filename: Path):
    """
    Read real and imaginary coefficients from a file.
    Pads the lists with zeros up to length 16 if needed.
    """
    real_coeffs = []
    imag_coeffs = []

    # Ensure filename is a Path object
    filename = Path(filename)

    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.split()
            real_coeffs.append(float(cols[1]))   # second column
            imag_coeffs.append(float(cols[2]))   # third column

    # Pad with zeros if fewer than 16 entries
    while len(real_coeffs) < 16:
        real_coeffs.append(0.0)
        imag_coeffs.append(0.0)

    return real_coeffs, imag_coeffs

def get_even_cubic_mode_coeffs(mode):
    """
    mode: [l, m, n]
    parity: l>0 for even, l<0 for odd

    """
    l, m, n= mode

    # Convert parity sign to file naming
    if l > 0:
        parity = "plus"
    elif l< 1:
        parity = "minus"
    else:
        raise ValueError(f"Invalid parity: {l}, expected l>0 or l<0")

    # Construct full path to the file using FITS_DIR
    filename = folder / f"cubic_even_{parity}_l{abs(l)}_m{m}_n{n}.txt"

    # Call the file-reading function
    return get_even_cubic_coeffs_from_file(filename)

real_mode_coeffs=[]
imag_mode_coeffs=[]
for mode in mode_list:
    real_coeffs,imag_coeffs = get_even_cubic_mode_coeffs(mode)
    real_mode_coeffs.append(real_coeffs)
    imag_mode_coeffs.append(imag_coeffs)

# Even cubic Shift fits
a_omega = [
    _to_jnp(x)
    for x in real_mode_coeffs
]
a_gamma = [
    _to_jnp(x)
    for x in imag_mode_coeffs
]

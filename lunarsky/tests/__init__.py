
import numpy as np
from astropy.coordinates import SkyCoord, Angle

def get_catalog():
    # Generate a fake catalog for tests.
    Nangs = 30
    ras = np.linspace(0, 360, Nangs)
    decs = np.linspace(-90, 90, Nangs)
    ras, decs = map(np.ndarray.flatten, np.meshgrid(ras, decs))

    stars = SkyCoord(ra=ras, dec=decs, unit='deg', frame='icrs')
    return stars

def positions_close(fr0, fr1, tol):
    # Check that astropy star positions are close.
    # tol = Angle object or angle in rad

    vecs0 = fr0.cartesian.xyz.value
    vecs1 = fr1.cartesian.xyz.value
    N = vecs0.shape[-1] # last axis is number of objects
    dots = np.array([np.dot(vecs0[:, mi], vecs1[:, mi]) for mi in range(N)])
    invalid = np.abs(dots)>1.0

    # Floating errors may push some dot products to be larger than 1.
    # Check these are within floating precision of 1.
    check_inv = np.isclose(np.abs(dots[invalid]),1.0)
    dev_angs = Angle(np.arccos(dots), 'rad')
    return np.all(dev_angs[~invalid] < tol) and np.all(check_inv)


import numpy as np
from lunarsky import SkyCoord
from astropy.coordinates import Angle
import pytest


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
    N = vecs0.shape[-1]     # last axis is number of objects
    dots = np.array([np.dot(vecs0[:, mi], vecs1[:, mi]) for mi in range(N)])
    invalid = np.abs(dots) > 1.0

    # Floating errors may push some dot products to be larger than 1.
    # Check these are within floating precision of 1.
    check_inv = np.isclose(np.abs(dots[invalid]), 1.0)
    dev_angs = Angle(np.arccos(dots[~invalid]), 'rad')
    return np.all(dev_angs < tol) and np.all(check_inv)


def assert_raises_message(exception_type, message, func, *args, **kwargs):
    """
    Check that the correct error message is raised.
    """
    nocatch = kwargs.pop('nocatch', False)
    if nocatch:
        func(*args, **kwargs)

    with pytest.raises(exception_type) as err:
        func(*args, **kwargs)

    try:
        assert message in str(err.value)
    except AssertionError as excp:
        print("{} not in {}".format(message, str(err.value)))
        raise excp

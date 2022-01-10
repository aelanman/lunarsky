import numpy as np
from astropy.coordinates import SkyCoord
import pytest
import warnings

# Ignore deprecation warnings from spiceypy
@pytest.fixture(autouse=True)
def ignore_representation_deprecation():
    warnings.filterwarnings(
        "ignore",
        message="Using or importing the ABCs from"
        "'collections' instead of from"
        "'collections.abc' is deprecated",
        category=DeprecationWarning,
    )


@pytest.fixture
def grcat():
    Nangs = 30
    ras = np.linspace(0, 360, Nangs)
    decs = np.linspace(-90 + 180 / Nangs, 90, Nangs, endpoint=False)
    ras, decs = map(np.ndarray.flatten, np.meshgrid(ras, decs))
    decs = np.insert(np.array([-90, 90]), 1, decs)
    ras = np.insert(np.array([0, 0]), 1, ras)

    stars = SkyCoord(ra=ras, dec=decs, unit="deg", frame="icrs")
    return stars

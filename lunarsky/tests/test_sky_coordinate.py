

import lunarsky.tests as ltest

from astropy.coordinates import ICRS, GCRS, EarthLocation, AltAz
from astropy.time import Time
import pytest

# Check that the changes to SkyCoord don't cause unexpected behavior.


def test_skycoord_transforms():
    # An EarthLocation object should still get copied over
    # under transformations.

    eloc = EarthLocation.from_geodetic(0.0, 10.0)
    coords = ltest.get_catalog()

    altaz = coords.transform_to(AltAz(location=eloc, obstime=Time.now()))

    assert altaz.location == eloc

    gcrs = altaz.transform_to(GCRS())

    assert gcrs.location == eloc

    icrs = altaz.transform_to(ICRS())

    assert icrs.location == eloc

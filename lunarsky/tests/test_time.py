import numpy as np
from astropy.coordinates import ICRS
from astropy.time import TimeDelta
import pytest

from lunarsky import MoonLocation, SkyCoord, Time


@pytest.mark.parametrize("lat", np.linspace(-89, 89, 5))
@pytest.mark.parametrize("lon", np.linspace(0, 360, 5))
def test_sidereal_time_calculation(lat, lon):
    # Confirm that the ra of the zenith is close to the calculated LST.

    t0 = Time.now()
    loc = MoonLocation.from_selenodetic(lon, lat, 0)
    t0.location = loc

    Ntimes = 200
    Ndays = 28
    times = t0 + TimeDelta(np.linspace(0, Ndays, Ntimes) * 3600 * 24, format="sec")

    for tt in times:
        src = SkyCoord(alt="90d", az="0d", frame="lunartopo", obstime=tt, location=loc)
        lst = tt.sidereal_time("mean")
        assert np.isclose(lst.deg, src.transform_to(ICRS()).ra.deg, atol=1e-4)

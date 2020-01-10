
from astropy.time import Time
import astropy.coordinates as ascoord
import lunarsky
import lunarsky.tests as ltests
import numpy as np
import pytest


Ntimes = 5
Nangs = 3
latitudes = np.linspace(0, 90, Nangs)
longitudes = np.linspace(0, 360, Nangs)
latlons = [(lat, lon) for lon in longitudes for lat in latitudes]

# Ten years of time.
times = Time(lunarsky.topo._J2000.jd + np.linspace(0, 10 * 365.25, Ntimes), format='jd')


@pytest.mark.parametrize('time', times)
@pytest.mark.parametrize('lat,lon', latlons)
def test_icrs_to_mcmf(time, lat, lon):
    # Check that the following transformation paths are equivalent:
    #   ICRS -> MCMF -> TOPO
    #   ICRS -> TOPO

    stars = ltests.get_catalog()

    loc = lunarsky.MoonLocation.from_selenodetic(lon, lat)

    topo0 = stars.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    mcmf = stars.transform_to(lunarsky.MCMF(obstime=time))
    topo1 = mcmf.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    assert ltests.positions_close(topo0, topo1, ascoord.Angle(10.0, 'arcsec'))

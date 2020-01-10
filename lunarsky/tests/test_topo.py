
from astropy.time import Time
import astropy.coordinates as ascoord
import lunarsky
import numpy as np
import pytest


Ntimes = 5
Nangs = 3
latitudes = np.linspace(0, 90, Nangs)
longitudes = np.linspace(0, 360, Nangs)
latlons = [(lat, lon) for lon in longitudes for lat in latitudes]

# Ten years of time.
times = Time(lunarsky.topo._J2000.jd + np.linspace(0, 10*365.25, Ntimes), format='jd')


@pytest.mark.parametrize('time', times)
@pytest.mark.parametrize('lat,lon', latlons)
def test_icrs_to_mcmf(time, lat, lon):
    # Check that the following transformation paths are equivalent:
    #   ICRS -> MCMF -> TOPO
    #   ICRS -> TOPO

    # TODO -- Replace with a small actual catalog.
    Nangs = 30
    ras = np.linspace(0, 360, Nangs)
    decs = np.linspace(-90, 90, Nangs)
    ras, decs = map(np.ndarray.flatten, np.meshgrid(ras, decs))

    stars = ascoord.SkyCoord(ra=ras, dec=decs, unit='deg', frame='icrs')

    loc = lunarsky.MoonLocation.from_selenodetic(lon, lat)

    topo0 = stars.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    mcmf = stars.transform_to(lunarsky.MCMF(obstime=time))
    topo1 = mcmf.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))

    assert np.all(topo0.alt == topo1.alt)
    assert np.all(topo0.az == topo1.az)


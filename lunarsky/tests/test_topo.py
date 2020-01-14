
from astropy.time import Time, TimeDelta
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


@pytest.mark.parametrize('time', times)
@pytest.mark.parametrize('lat,lon', latlons)
def test_topo_transform_loop(time, lat, lon):
    # Testing remaining transformations
    height = 10.0     # m
    stars = ltests.get_catalog()
    loc = lunarsky.MoonLocation.from_selenodetic(lon, lat, height)
    topo0 = stars.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    icrs0 = topo0.transform_to(ascoord.ICRS())
    assert ltests.positions_close(stars, icrs0, ascoord.Angle(5.0, 'arcsec'))

    mcmf0 = topo0.transform_to(lunarsky.MCMF(obstime=time))
    mcmf1 = stars.transform_to(lunarsky.MCMF(obstime=time))
    assert ltests.positions_close(mcmf0, mcmf1, ascoord.Angle(5.0, 'arcsec'))


def test_earth_from_moon():
    # Look at the position of the Earth from the Moon over time.
    Ntimes = 100
    ets = np.linspace(0, 4 * 28 * 24 * 3600., Ntimes)    # Four months
    times_jd = Time.now() + TimeDelta(ets, format='sec')

    # Minimum and maximum, respectively, over the year.
    # The lunar apogee nad perigee vary over time. These are
    # chosen from a table of minimum/maxmium perigees over a century.
    # http://astropixels.com/ephemeris/moon/moonperap2001.html
    lunar_perigee = 356425.0    # km, Dec 6 2052
    lunar_apogee = 406709.0     # km, Dec 12 2061

    lat, lon = 0, 0  # deg
    loc = lunarsky.MoonLocation.from_selenodetic(lat, lon)
    zaaz_deg = np.zeros((Ntimes, 2))
    for ti, tim in enumerate(times_jd):
        mcmf = lunarsky.spice_utils.earth_pos_mcmf(tim)
        dist = np.linalg.norm(mcmf.cartesian.xyz.to('km').value)
        assert lunar_perigee < dist < lunar_apogee
        top = mcmf.transform_to(lunarsky.LunarTopo(location=loc, obstime=tim))
        zaaz_deg[ti, :] = [top.zen.deg, top.az.deg]

    assert np.all(zaaz_deg[:, 0] < 10)  # All zenith angles should be less than 10 degrees

    # Check that the periodicity of the Earth's motion around the zenith
    # is consistent with the Moon's orbit.
    moonfreq = 1 / (28. * 24. * 3600.)    # Hz, frequency of the moon's orbit
    _az = np.fft.fft(zaaz_deg[:, 1])
    ks = np.fft.fftfreq(Ntimes, d=np.diff(ets)[0])
    sel = ks > 0

    closest = np.argmin(np.abs(moonfreq - ks[sel]))
    peakloc = np.argmax(np.abs(_az[sel]))
    assert peakloc == closest

from lunarsky.time import Time, TimeDelta
import astropy.coordinates as ac
from astropy import units as un
from astropy.tests.helper import assert_quantity_allclose
import lunarsky
import numpy as np
import pytest


# Lunar station positions
Nangs = 3
latitudes = np.linspace(0, 90, Nangs)
longitudes = np.linspace(0, 360, Nangs)
latlons = [(lat, lon) for lon in longitudes for lat in latitudes]

# Times
t0 = Time("2010-10-28T15:30:00")
_J2000 = Time("J2000")

jd_10yr = Time(t0.jd + np.linspace(0, 10 * 365.25, 5), format="jd")
et_10yr = (jd_10yr - _J2000).sec

jd_4mo = t0 + TimeDelta(np.linspace(0, 4 * 28 * 24 * 3600.0, 100), format="sec")
et_4mo = (jd_4mo - _J2000).sec


@pytest.mark.parametrize("time", jd_10yr)
@pytest.mark.parametrize("lat,lon", latlons)
def test_icrs_to_mcmf(time, lat, lon, grcat):
    # Check that the following transformation paths are equivalent:
    #   ICRS -> MCMF -> TOPO
    #   ICRS -> TOPO
    loc = lunarsky.MoonLocation.from_selenodetic(lon, lat)
    topo0 = grcat.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    mcmf = grcat.transform_to(lunarsky.MCMF(obstime=time))
    topo1 = mcmf.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    assert np.all(topo0.separation(topo1) < ac.Angle("10arcsec"))


@pytest.mark.parametrize(
    "obj",
    [
        ac.get_sun(Time("2025-05-30T03:30:19.00")).transform_to(ac.ICRS),
        ac.SkyCoord(ra="30d", dec="-70d", frame="icrs"),
    ],
)
@pytest.mark.parametrize("path", [["lunartopo"], ["mcmf"], ["lunartopo", "mcmf"]])
def test_transform_loops(obj, path):
    # Tests from ICRS -> path -> ICRS
    t0 = Time("2025-05-30T03:30:19.00")
    loc = lunarsky.MoonLocation.from_selenodetic(10, 87)
    fdict = {
        "lunartopo": lunarsky.LunarTopo(location=loc, obstime=t0),
        "mcmf": lunarsky.MCMF(obstime=t0),
    }
    orig_pos = obj.cartesian.xyz.copy()

    for fr in path:
        obj = obj.transform_to(fdict[fr])
    # Lastly, back to ICRS
    obj = obj.transform_to(ac.ICRS())
    tol = 1e-4 if (obj.spherical.distance.unit == un.one) else 1 * un.m
    assert_quantity_allclose(obj.cartesian.xyz, orig_pos, atol=tol)


@pytest.mark.parametrize("time", jd_10yr)
@pytest.mark.parametrize("lat,lon", latlons)
def test_topo_transform_loop(time, lat, lon, grcat):
    # Testing remaining transformations
    height = 10.0  # m
    loc = lunarsky.MoonLocation.from_selenodetic(lon, lat, height)
    topo0 = grcat.transform_to(lunarsky.LunarTopo(location=loc, obstime=time))
    icrs0 = topo0.transform_to(ac.ICRS())
    assert np.all(grcat.separation(icrs0) < ac.Angle("5arcsec"))

    mcmf0 = topo0.transform_to(lunarsky.MCMF(obstime=time))
    mcmf1 = grcat.transform_to(lunarsky.MCMF(obstime=time))
    assert np.all(mcmf0.separation(mcmf1) < ac.Angle("5arcsec"))


def test_earth_from_moon():
    # Look at the position of the Earth in lunar reference frames.

    # The lunar apo/perigee vary over time. These are
    # chosen from a table of minimum/maxmium perigees over a century.
    # http://astropixels.com/ephemeris/moon/moonperap2001.html
    lunar_perigee = 356425.0  # km, Dec 6 2052
    lunar_apogee = 406709.0  # km, Dec 12 2061

    mcmf = lunarsky.spice_utils.earth_pos_mcmf(jd_4mo)
    dists = mcmf.spherical.distance.to_value("km")
    assert all((lunar_perigee < dists) & (dists < lunar_apogee))

    # # Compare position of Earth in MCMF frame from astropy and from spice
    epos_icrs = ac.SkyCoord(ac.get_body_barycentric("earth", jd_4mo), frame="icrs")
    mcmf_ap = epos_icrs.transform_to(lunarsky.MCMF(obstime=jd_4mo)).frame
    assert_quantity_allclose(mcmf_ap.cartesian.xyz, mcmf.cartesian.xyz, atol="5km")
    #   TODO Lower this tolerance ^

    # Now test LunarTopo frame positions
    lat, lon = 0, 0  # deg
    loc = lunarsky.MoonLocation.from_selenodetic(lat, lon)
    topo = mcmf.transform_to(lunarsky.LunarTopo(location=loc, obstime=jd_4mo))

    # The Earth should stay within 10 deg of zenith of lat=lon=0 position
    assert np.all(topo.zen.deg < 10)

    # Check that the periodicity of the Earth's motion around the zenith
    # is consistent with the Moon's orbit.
    moonfreq = 1 / (28.0 * 24.0 * 3600.0)  # Hz, frequency of the moon's orbit
    _az = np.fft.fft(topo.az.deg)
    ks = np.fft.fftfreq(jd_4mo.size, d=np.diff(et_4mo)[0])
    sel = ks > 0

    closest = np.argmin(np.abs(moonfreq - ks[sel]))
    peakloc = np.argmax(np.abs(_az[sel]))
    assert peakloc == closest


def test_multi_times():
    # Check vectorization over time axis for LunarTopo transformations
    lat, lon = latlons[6]
    loc = lunarsky.MoonLocation.from_selenodetic(lon, lat, height=10)
    star = lunarsky.SkyCoord(ra=[35], dec=[17], unit="deg", frame="icrs")
    topo0 = star.transform_to(lunarsky.LunarTopo(location=loc, obstime=jd_4mo))
    icrs0 = topo0.transform_to(ac.ICRS())

    assert np.all(star.separation(icrs0) < ac.Angle("5arcsec"))

    mcmf0 = topo0.transform_to(lunarsky.MCMF(obstime=jd_4mo))
    mcmf1 = star.transform_to(lunarsky.MCMF(obstime=jd_4mo))
    assert np.all(mcmf0.separation(mcmf1) < ac.Angle("5arcsec"))


def test_sidereal_vs_solar_day():
    # Compute sidereal vs solar day length from time between transits of
    # the Sun and an arbitrary star.

    # 200 time steps over one month. Enough time to cover a full period
    ntimes = 200
    ts = Time("2025-01-01T00:00:00") + TimeDelta(np.linspace(0, 30, ntimes), format="jd")
    loc = lunarsky.MoonLocation.from_selenodetic(0, 56, 10.0)

    # Get Sun position and arbitrary star position
    sun_icrs = ac.get_sun(ts).transform_to("icrs")
    obj_icrs = ac.SkyCoord(ra="15d", dec="20d", frame="icrs")

    ltf = lunarsky.LunarTopo(location=loc, obstime=ts)
    sun_topo = sun_icrs.transform_to(ltf)
    obj_topo = obj_icrs.transform_to(ltf)

    # Measure period for each as double time between peak and trough
    sun_per = 2 * np.abs(
        ts[np.argmax(sun_topo.alt.deg)] - ts[np.argmin(sun_topo.alt.deg)]
    )
    obj_per = 2 * np.abs(
        ts[np.argmax(obj_topo.alt.deg)] - ts[np.argmin(obj_topo.alt.deg)]
    )

    assert np.isclose(sun_per.jd, 29.55, atol=0.1)
    assert np.isclose(obj_per.jd, 27.13, atol=0.1)


@pytest.mark.parametrize(
    "loc",
    [
        lunarsky.MoonLocation.from_selenodetic(lat, lon, 10.0)
        for lat, lon in [(27, 53), (10, 8)]
    ],
)
@pytest.mark.parametrize("toframe", ["lunartopo", "mcmf"])
def test_nearby_obj_transforms(toframe, loc):
    sun_gcrs = ac.get_sun(jd_4mo)

    if toframe == "lunartopo":
        frame = lunarsky.LunarTopo(location=loc, obstime=jd_4mo)
    elif toframe == "mcmf":
        frame = lunarsky.MCMF(obstime=jd_4mo)

    res = sun_gcrs.transform_to(frame).transform_to(ac.GCRS(obstime=jd_4mo))
    assert_quantity_allclose(sun_gcrs.cartesian.xyz, res.cartesian.xyz)

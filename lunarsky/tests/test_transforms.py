import numpy as np
from copy import deepcopy
from lunarsky.time import Time, TimeDelta
import astropy.coordinates as ac
from astropy import units as un
from astropy.utils import IncompatibleShapeError, exceptions
from astropy.tests.helper import assert_quantity_allclose
import lunarsky
from lunarsky.moon import SELENOIDS
import pytest

try:
    from astropy.coordinates.angles.utils import angular_separation
except ImportError:
    from astropy.coordinates.angle_utilities import angular_separation

# Lunar station positions
Nangs = 7
latitudes = np.linspace(-90, 90, Nangs)
longitudes = np.linspace(0, 360, Nangs)
latlons_grid = [(lat, lon) for lon in longitudes for lat in latitudes]

# Avoiding poles:
latlons_grid_nopole = [
    (lat, lon)
    for lon in np.linspace(0.1, 360, Nangs, endpoint=False)
    for lat in np.linspace(-70.0, 70.0, Nangs, endpoint=False)
]

# Times
t0 = Time("2020-10-28T15:30:00")
_J2000 = Time("J2000")

jd_10yr = Time(t0.jd + np.linspace(0, 10 * 365.25, 5), format="jd")
jd_4mo = t0 + TimeDelta(np.linspace(0, 4 * 28 * 24 * 3600.0, 100), format="sec")


@pytest.mark.parametrize("time", jd_10yr)
@pytest.mark.parametrize("lat,lon", latlons_grid)
@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
def test_icrs_to_topo_long_time(time, lat, lon, grcat):
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
        ac.get_sun(t0).transform_to(ac.ICRS),
        ac.SkyCoord(ra="30d", dec="-70d", frame="icrs"),
    ],
)
@pytest.mark.parametrize("path", [["lunartopo"], ["mcmf"], ["lunartopo", "mcmf"]])
@pytest.mark.parametrize(
    "time, lat, lon",
    [
        (t0, 11.2, 1.4),
        (t0, [10.3, 11.2], [0.0, 1.4]),
        (jd_4mo[:2], 10.3, 0.0),
        (jd_4mo[:2], [10.3, 11.2], [0.0, 1.4]),
    ],
)
@pytest.mark.parametrize("ell", SELENOIDS)
def test_transform_loops(obj, path, time, lat, lon, ell):
    # Tests from ICRS -> path -> ICRS
    obj = lunarsky.SkyCoord(obj)  # Ensure we're working with lunarsky-compatible SkyCoord
    loc = lunarsky.MoonLocation.from_selenodetic(lat, lon, ellipsoid=ell)
    fdict = {
        "lunartopo": lunarsky.LunarTopo(location=loc, obstime=time),
        "mcmf": lunarsky.MCMF(obstime=time),
    }
    orig_pos = obj.cartesian.xyz.copy()

    for fr in path:
        obj = obj.transform_to(fdict[fr])

    # Lastly, back to ICRS
    obj = obj.transform_to(ac.ICRS())
    if obj.ndim == 1:
        obj = obj[0]
    tol = 1e-4 if (obj.spherical.distance.unit == un.one) else 1 * un.m
    assert_quantity_allclose(obj.cartesian.xyz, orig_pos, atol=tol)


@pytest.mark.parametrize("ell", SELENOIDS)
def test_topo_to_topo(ell):
    # Check that zenith source transforms properly
    loc0 = lunarsky.MoonLocation.from_selenodetic(lat=0, lon=90, ellipsoid=ell)
    loc1 = lunarsky.MoonLocation.from_selenodetic(lat=0, lon=0)

    src = lunarsky.SkyCoord(alt="90d", az="0d", frame="lunartopo", location=loc0)
    new = src.transform_to(lunarsky.LunarTopo(location=loc1))
    assert new.az.deg == 90


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
def test_mcmf_to_mcmf():
    # Transform MCMF positions to MCMF frame half a lunar sidereal day later.
    # Assert that the new positions are roughly close to 180 deg from the original.
    src = lunarsky.SkyCoord(ra="0d", dec="0d", frame="icrs")
    src = src.transform_to(lunarsky.MCMF(obstime=jd_10yr))
    orig_pos = deepcopy(src)
    src = src.transform_to(
        lunarsky.MCMF(obstime=jd_10yr + TimeDelta(27.322 / 2, format="jd"))
    )
    sph0 = src.spherical
    sph1 = orig_pos.spherical
    res = angular_separation(sph0.lon, sph0.lat, sph1.lon, sph1.lat).to("deg")
    assert_quantity_allclose(res, 177 * un.deg, atol=5 * un.deg)
    # Tolerance to allow for lunar precession / nutation.


@pytest.mark.parametrize("ell", SELENOIDS.keys())
def test_earth_from_moon(ell):
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
    epos_icrs_ap = ac.SkyCoord(ac.get_body_barycentric("earth", jd_4mo), frame="icrs")
    epos_icrs_sp = ac.SkyCoord(mcmf).transform_to(ac.ICRS())
    mcmf_ap = epos_icrs_ap.transform_to(lunarsky.MCMF(obstime=jd_4mo)).frame
    assert_quantity_allclose(mcmf_ap.cartesian.xyz, mcmf.cartesian.xyz, atol="6km")
    #   TODO Lower this tolerance ^

    # Whatever difference exists between the two positions should be the same in both ICRS
    # and MCMF frames:
    assert_quantity_allclose(
        np.linalg.norm(epos_icrs_ap.cartesian.xyz - epos_icrs_sp.cartesian.xyz, axis=0),
        np.linalg.norm(mcmf_ap.cartesian.xyz - mcmf.cartesian.xyz, axis=0),
    )

    # Now test LunarTopo frame positions
    lat, lon = 0, 0  # deg
    loc = lunarsky.MoonLocation.from_selenodetic(lat, lon, ellipsoid=ell)
    topo = mcmf.transform_to(lunarsky.LunarTopo(location=loc, obstime=jd_4mo))

    # The Earth should stay within 10 deg of zenith of lat=lon=0 position
    assert np.all(topo.zen.deg < 10)

    # Check that the periodicity of the Earth's motion around the zenith
    # is consistent with the Moon's orbit.
    moonfreq = 1 / (28.0 * 24.0 * 3600.0)  # Hz, frequency of the moon's orbit
    _az = np.fft.fft(topo.az.deg)
    ks = np.fft.fftfreq(jd_4mo.size, d=np.diff((jd_4mo - _J2000).sec)[0])
    sel = ks > 0

    closest = np.argmin(np.abs(moonfreq - ks[sel]))
    peakloc = np.argmax(np.abs(_az[sel]))
    assert peakloc == closest


def test_multi_times():
    # Check vectorization over time axis for LunarTopo transformations
    lat, lon = latlons_grid[6]
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
    sun_gcrs = lunarsky.SkyCoord(ac.get_sun(jd_4mo))

    if toframe == "lunartopo":
        frame = lunarsky.LunarTopo(location=loc, obstime=jd_4mo)
    elif toframe == "mcmf":
        frame = lunarsky.MCMF(obstime=jd_4mo)

    res = sun_gcrs.transform_to(frame).transform_to(ac.GCRS(obstime=jd_4mo))
    assert_quantity_allclose(sun_gcrs.cartesian.xyz, res.cartesian.xyz)


@pytest.mark.parametrize(
    "Nt, Nl, success",
    [
        (2, 1, True),
        (1, 2, True),
        (2, 2, True),
        (2, 3, False),
        (3, 2, False),
    ],
)
def test_incompatible_shape_error(Nt, Nl, success):
    # Transforming to lunartopo

    latlons_sel = np.array(latlons_grid[:Nl]).T
    locs = lunarsky.MoonLocation.from_selenodetic(
        lat=latlons_sel[0] * un.deg, lon=latlons_sel[1] * un.deg
    )

    times = Time.now() + TimeDelta(np.linspace(0, 10, Nt), format="sec")

    if success:
        lunarsky.LunarTopo(location=locs, obstime=times)
    else:
        with pytest.raises(ValueError, match="non-scalar data and/or attributes"):
            lunarsky.LunarTopo(location=locs, obstime=times)


@pytest.mark.parametrize("fromframe", ["icrs", "mcmf"])
def test_incompatible_transform(fromframe):
    Nl = 3
    Nt = 1
    Ns = 5
    latlons_sel = np.array(latlons_grid[:Nl]).T
    locs = lunarsky.MoonLocation.from_selenodetic(
        lat=latlons_sel[0] * un.deg, lon=latlons_sel[1] * un.deg
    )

    times = Time.now() + TimeDelta(np.linspace(0, 10, Nt), format="sec")
    ltop = lunarsky.LunarTopo(location=locs, obstime=times)

    fromframe = ac.frame_transform_graph.lookup_name(fromframe)
    coo = fromframe().realize_frame(
        ac.SphericalRepresentation(
            lat=np.linspace(-10, 10, Ns) * un.deg,
            lon=np.linspace(30, 40, Ns) * un.deg,
            distance=1,
        )
    )
    src = lunarsky.SkyCoord(coo)
    with pytest.raises(IncompatibleShapeError):
        src.transform_to(ltop)


@pytest.mark.parametrize("ell", SELENOIDS)
def test_finite_vs_spherical(ell):
    # Transform MCMF coordinates with distance and without units
    # Check consistency with ellipsoid equatorial radius
    # Assumes infinite distance if no unit given, as astropy does.

    R0 = 404789  # km
    xyz = np.array([[R0, -R0], [0, 0], [0, 0]])
    with_units = lunarsky.SkyCoord(lunarsky.MCMF(*(xyz * un.km)))
    sans_units = lunarsky.SkyCoord(lunarsky.MCMF(*(xyz)))

    loc = lunarsky.MoonLocation.from_selenodetic(
        lon=180 * un.deg, lat=0, height=0, ellipsoid=ell
    )
    altaz_with_units = with_units.transform_to(lunarsky.LunarTopo(location=loc))
    with pytest.warns(exceptions.AstropyUserWarning, match="Coordinates do not "):
        altaz_sans_units = sans_units.transform_to(lunarsky.LunarTopo(location=loc))

    dists = R0 * un.km + np.array([1, -1]) * SELENOIDS[ell]._equatorial_radius
    assert np.all(altaz_with_units.distance == dists)
    assert_quantity_allclose(altaz_sans_units.distance, 1.0)


@pytest.mark.parametrize("ell", SELENOIDS)
@pytest.mark.parametrize("lat,lon", latlons_grid_nopole)
def test_topo_zenith_shift(ell, lat, lon):
    # Verify that a given source at zenith in one topo frame shifts
    # predictably when viewed from a topo frame with the same lat/lon but
    # different ellipsoid

    # Checking that the ellipsoid is interpreted correctly

    # This test is a little sketchy... will need to review this later.
    #   Some discrepancies for GRAIL23 selenoid when the source distance is large.
    #       Choosing 1000 km for now.
    #   Also fails near poles.

    # Comparing against the SPHERE ellipsoid. Test fails for this due to divide by zero
    if ell == "SPHERE":
        return

    lat *= un.deg
    lon *= un.deg

    loc0 = lunarsky.MoonLocation.from_selenodetic(
        lon, lat, ellipsoid="SPHERE"
    )  # For reference.
    loc1 = lunarsky.MoonLocation.from_selenodetic(
        lon, lat, ellipsoid=ell
    )  # Same lat/lon = different place for different ellipsoid

    # Zenith source at finite distance over loc0.
    src0 = lunarsky.SkyCoord(
        alt="90d",
        az="0d",
        distance=1e3 * un.km,
        frame=lunarsky.LunarTopo(location=loc0, obstime=Time.now()),
    )
    src1 = src0.transform_to(lunarsky.LunarTopo(location=loc1))

    # Law of sines:
    #   Geometry among zenith angle, mcmf station vector,
    #   and selenocentric vs. selenodetic latitudes

    R1 = loc1.mcmf.cartesian.norm()
    lat_cen = loc1.mcmf.spherical.lat
    lat_det = loc1.lat
    assert_quantity_allclose(
        src1.distance / np.sin((np.abs(lat_det - lat_cen)).rad),
        R1 / np.sin(src1.zen.rad),
        atol=1 * un.km,
    )

import numpy as np

from astropy.coordinates import ICRS, GCRS, EarthLocation, AltAz
from astropy.time import Time

from lunarsky import MoonLocation, SkyCoord, LunarTopo, MCMF

# Check that the changes to SkyCoord don't cause unexpected behavior.


def test_skycoord_transforms(grcat):
    # An EarthLocation object should still get copied over
    # under transformations.

    eloc = EarthLocation.from_geodetic(0.0, 10.0)

    altaz = grcat.transform_to(AltAz(location=eloc, obstime=Time.now()))

    assert altaz.location == eloc

    gcrs = altaz.transform_to(GCRS())

    assert gcrs.location == eloc

    icrs = altaz.transform_to(ICRS())

    assert icrs.location == eloc


def test_skycoord_with_lunar_frames():
    # Check that defining a SkyCoord with frames
    # lunartopo and mcmf works correctly.

    Nsrcs = 10
    alts = np.random.uniform(0, np.pi / 2, Nsrcs)
    azs = np.random.uniform(0, 2 * np.pi, Nsrcs)
    t0 = Time.now()
    loc = MoonLocation.from_selenodetic(0, 0)
    src = SkyCoord(
        alt=alts, az=azs, unit="rad", frame="lunartopo", obstime=t0, location=loc
    )

    assert src.location == loc
    assert isinstance(src.frame, LunarTopo)
    x, y, z = src.transform_to("mcmf").cartesian.xyz
    src2 = SkyCoord(x=x, y=y, z=z, frame="mcmf", obstime=t0, location=loc)

    assert isinstance(src2.frame, MCMF)

    icrs2 = src2.transform_to("icrs")
    icrs1 = src.transform_to("icrs")
    assert np.allclose(icrs2.ra.deg, icrs1.ra.deg, atol=1e-5)
    assert np.allclose(icrs2.dec.deg, icrs1.dec.deg, atol=1e-5)


def test_earth_and_moon():
    # Check that as I do transforms with both Earth and Moon positions,
    # the transform graph doesn't break.

    Nsrcs = 10
    alts = np.random.uniform(0, np.pi / 2, Nsrcs)
    azs = np.random.uniform(0, 2 * np.pi, Nsrcs)
    t0 = Time.now()
    loc = MoonLocation.from_selenodetic(0, 0)
    src = SkyCoord(
        alt=alts, az=azs, unit="rad", frame="lunartopo", obstime=t0, location=loc
    )

    eloc = EarthLocation.from_geodetic(0, 0)
    src2 = SkyCoord(
        alt=alts, az=azs, unit="rad", frame="altaz", obstime=t0, location=eloc
    )

    src.transform_to("icrs")
    src2.transform_to("icrs")

import pytest
import copy
import numpy as np
import astropy.units as unit
from astropy.coordinates import Longitude, Latitude
from lunarsky import MoonLocation, MoonLocationAttribute, MCMF


class TestsWithObject:
    # The following three functions have been adapted from corresponding
    # tests in astropy/coordinates/tests/test_earth.py
    def setup(self):
        # Check that the setup from different input formats works as expected.
        self.lon = Longitude(
            [0.0, 45.0, 90.0, 135.0, 180.0, -180, -90, -45],
            unit.deg,
            wrap_angle=180 * unit.deg,
        )
        self.lat = Latitude([+0.0, 30.0, 60.0, +90.0, -90.0, -60.0, -30.0, 0.0], unit.deg)
        self.h = unit.Quantity([0.1, 0.5, 1.0, -0.5, -1.0, +4.2, -11.0, -0.1], unit.m)
        self.location = MoonLocation.from_selenodetic(self.lon, self.lat, self.h)
        self.x, self.y, self.z = self.location.to_selenocentric()

    def test_input(self):
        cartesian = MoonLocation(self.x, self.y, self.z)
        assert np.all(cartesian == self.location)
        cartesian = MoonLocation(self.x.value, self.y.value, self.z.value, self.x.unit)
        assert np.all(cartesian == self.location)
        spherical = MoonLocation(self.lon.deg, self.lat.deg, self.h.to(unit.m))
        assert np.all(spherical == self.location)

    def test_invalid(self):
        # incomprehensible by either raises TypeError
        # Check error cases in setup.

        # TODO Include error messages in the check
        with pytest.raises(TypeError):
            MoonLocation(self.lon, self.y, self.z)

        # wrong units
        with pytest.raises(unit.UnitsError):
            MoonLocation.from_selenocentric(self.lon, self.lat, self.lat)
        # inconsistent units
        with pytest.raises(unit.UnitsError):
            MoonLocation.from_selenocentric(self.h, self.lon, self.lat)
        # floats without a unit
        with pytest.raises(TypeError):
            MoonLocation.from_selenocentric(self.x.value, self.y.value, self.z.value)
        # inconsistent shape
        with pytest.raises(ValueError):
            MoonLocation.from_selenocentric(self.x, self.y, self.z[:5])

        # inconsistent shape
        with pytest.raises(ValueError):
            MoonLocation.from_selenodetic(self.lon, self.lat[:5])

        # inconsistent units
        with pytest.raises(unit.UnitsError):
            MoonLocation.from_selenodetic(self.x, self.y, self.z)
        # inconsistent shape
        with pytest.raises(ValueError):
            MoonLocation.from_selenodetic(self.lon, self.lat, self.h[:5])

    def test_attributes(self):
        assert np.allclose(self.location.height, self.h)
        assert np.allclose(self.location.lon, self.lon)
        assert np.allclose(self.location.lat, self.lat)

    def test_mcmf_attr(self):
        mcmf = MCMF(x=self.x, y=self.y, z=self.z)
        assert np.allclose(mcmf.x, self.location.mcmf.x)
        assert np.allclose(mcmf.y, self.location.mcmf.y)
        assert np.allclose(mcmf.z, self.location.mcmf.z)


def test_moonlocation_attribute():
    # Make a MoonLocationAttribute in three ways
    # Confirm each looks like a typical MoonLocation

    testObj = TestsWithObject()
    testObj.setup()
    moonloc = testObj.location

    mlattr = MoonLocationAttribute()
    attr1, boo = mlattr.convert_input(None)
    assert attr1 is None
    assert not boo

    # If not None, this will look for a "transform_to"
    # method, which of course a string doesn't have.
    with pytest.raises(ValueError, match="was passed into a MoonLocationAttribute"):
        mlattr.convert_input("string")

    attr2, boo = mlattr.convert_input(moonloc.mcmf)
    assert np.all(attr2 == moonloc)


def test_moonlocation_copy():
    # Check that station_ids are copied properly
    loc0 = MoonLocation.from_selenodetic(lat=["-15d", "25d"], lon=["97d", "0d"])
    lcop = loc0.copy()
    assert lcop.station_ids == loc0.station_ids
    lcop2 = copy.copy(loc0)
    assert lcop2.station_ids == loc0.station_ids


def test_moonlocation_delete():
    # Check that making multiple instances of the same location raises the _ref_count
    # appropriately, and deleting them removes them correctly.
    before = copy.copy(MoonLocation._existing_stat_ids)
    s0 = MoonLocation._new_stat_id

    locs = []
    for ii in range(5):
        locs.append(MoonLocation.from_selenodetic(lat=["-15d", "25d"], lon=["97d", "0d"]))

    check0 = copy.copy(MoonLocation._existing_stat_ids)
    for ii in range(5, 0, -1):
        assert MoonLocation._ref_count[-1] == ii
        assert MoonLocation._existing_stat_ids[-1] == check0[-1]
        locs.pop(ii - 1)

    check1 = copy.copy(MoonLocation._existing_stat_ids)

    exp = before + [s0, s0 + 1]

    assert exp == check0
    assert check1 == before


def test_station_ids():
    # Check that when a MoonLocations are made, the appropriate station_ids are assigned.

    # Get whatever are already recorded in the class.
    orig_statids = copy.copy(MoonLocation._existing_stat_ids)

    # Random positions with some repeats, in five groups
    lonlatheights = [
        [
            (294.67, -67.68, 15.23),
            (78.31, 51.60, 16.59),
            (335.27, 57.45, 27.32),
        ],
        [
            (335.27, 57.45, 27.32),
            (133.29, -80.63, 25.92),
            (323.28, -2.46, 4.74),
            (45.94, -25.17, 20.22),
            (326.40, 29.21, 9.67),
            (197.94, 3.92, 16.34),
        ],
        [(197.94, 3.92, 16.34)],
        [
            (197.94, 3.92, 16.34),
            (242.63, -20.86, 4.90),
            (70.30, -40.22, 29.91),
            (187.70, -85.44, 9.35),
            (129.55, 14.11, 0.24),
            (140.46, 4.19, 28.03),
            (232.22, 56.89, 15.22),
            (51.31, -30.92, 11.53),
            (55.59, -61.90, 3.98),
        ],
        [
            (307.76, 32.31, 29.88),
            (248.07, 2.89, 22.41),
            (349.94, -77.10, 14.27),
            (315.41, -72.71, 13.26),
            (173.39, 14.43, 3.37),
            (183.19, -83.01, 16.13),
            (258.53, -56.66, 29.75),
            (115.36, -44.42, 11.34),
            (270.61, 57.85, 26.16),
            (320.07, -75.06, 21.15),
            (216.74, 54.75, 1.92),
        ],
    ]

    all_pos = np.array(sum(lonlatheights, []))
    n_unique = np.unique(all_pos, axis=0).shape[0]

    locs = []
    locstrs = []

    for gp in lonlatheights:
        lons, lats, heights = np.array(gp).T
        locs.append(MoonLocation.from_selenodetic(lat=lats, lon=lons, height=heights))

    # Check that only unique positions got added
    added = len(MoonLocation._existing_stat_ids) - len(orig_statids)
    assert added == n_unique

    statids = [loc.station_ids for loc in locs]

    locstrs = []
    for inst in locs:
        llhs = []
        if inst.isscalar:
            llh_arr = [
                (
                    inst.lon.deg.item(),
                    inst.lat.deg.item(),
                    inst.height.to_value("km").item(),
                )
            ]
        else:
            llh_arr = zip(inst.lon.deg, inst.lat.deg, inst.height.to_value("km"))
        for llh in llh_arr:
            llhs.append("_".join(["{:.4f}".format(ll) for ll in llh]))
        locstrs.append(llhs)

    # Check that saved location strings correspond with station IDs in each instance
    for gi, gp in enumerate(statids):
        for sti, sid in enumerate(gp):
            ind = MoonLocation._existing_stat_ids.index(sid)
            assert locstrs[gi][sti] == MoonLocation._existing_locs[ind]

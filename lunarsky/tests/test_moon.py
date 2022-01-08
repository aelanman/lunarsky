import pytest
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

import numpy as np
import copy
from astropy import units as u
from astropy.units.quantity import QuantityInfoBase
from astropy.coordinates.angles import Longitude, Latitude
from astropy.coordinates.earth import GeodeticLocation
from astropy.coordinates.representation.geodetic import BaseGeodeticRepresentation
from astropy.coordinates.representation import (
    CartesianRepresentation,
)
from astropy.coordinates.attributes import Attribute

from .spice_utils import remove_topo

LUNAR_RADIUS = 1737.1e3  # m

__all__ = ["MoonLocation", "MoonLocationAttribute"]


class SPHERESelenodeticRepresentation(BaseGeodeticRepresentation):
    """Lunar ellipsoid as a sphere

    Radius defined by lunarsky.spice_utils.LUNAR_RADIUS
    """

    _equatorial_radius = LUNAR_RADIUS * u.m
    _flattening = 0.0


class GSFCSelenodeticRepresentation(BaseGeodeticRepresentation):
    """Lunar ellipsoid from NASA/GSFC "Planetary Fact Sheet"

    https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
    """

    _equatorial_radius = 1738.1e3 * u.m
    _flattening = 0.0012


class GRAIL23SelenodeticRepresentation(BaseGeodeticRepresentation):
    """Lunar ellipsoid defined by gravimetry of GRAIL data.

    https://doi.org/10.1007/s40328-023-00415-w
    """

    _equatorial_radius = 1737576.6 * u.m
    _flattening = 0.000305


class CE1LAM10SelenodeticRepresentation(BaseGeodeticRepresentation):
    """Lunar ellipsoid from Chang'e 1 laser altimetry.

    Rotation ellipsoid = CE-1-LAM-GEO

    https://doi.org/10.1007/s11430-010-4060-6
    """

    _equatorial_radius = 1737.632 * u.km
    _flattening = 1 / 973.463


# Define reference ellipsoids
SELENOIDS = {
    "SPHERE": SPHERESelenodeticRepresentation,
    "GSFC": GSFCSelenodeticRepresentation,
    "GRAIL23": GRAIL23SelenodeticRepresentation,
    "CE-1-LAM-GEO": CE1LAM10SelenodeticRepresentation,
}


class SelenodeticLocation(GeodeticLocation):
    """Rename GeodeticLocation class for clarity"""


def _check_ellipsoid(ellipsoid=None, default="SPHERE"):
    """Defaulting lunar ellipsoid"""
    if ellipsoid is None:
        ellipsoid = default
    if ellipsoid not in SELENOIDS:
        raise ValueError(f"Ellipsoid {ellipsoid} not among known ones ({SELENOIDS})")
    return ellipsoid


class MoonLocationInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """

    _represent_as_dict_attrs = ("x", "y", "z", "ellipsoid")

    def _construct_from_dict(self, map):
        ellipsoid = map.pop("ellipsoid")
        out = self._parent_cls(**map)
        out.ellipsoid = ellipsoid
        return out

    def new_like(self, cols, length, metadata_conflicts="warn", name=None):
        """
        Return a new MoonLocation instance which is consistent with the
        input ``cols`` and has ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        cols : list
            List of input columns
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : MoonLocation (or subclass)
            Empty instance of this class consistent with ``cols``
        """
        # Very similar to QuantityInfo.new_like, but the creation of the
        # map is different enough that this needs its own routine.
        # Get merged info attributes shape, dtype, format, description.
        attrs = self.merge_cols_attributes(
            cols, metadata_conflicts, name, ("meta", "format", "description")
        )
        # The above raises an error if the dtypes do not match, but returns
        # just the string representation, which is not useful, so remove.
        attrs.pop("dtype")
        # Make empty MoonLocation using the dtype and unit of the last column.
        # Use zeros so we do not get problems for possible conversion to
        # selenodetic coordinates.
        shape = (length,) + attrs.pop("shape")
        data = u.Quantity(
            np.zeros(shape=shape, dtype=cols[0].dtype), unit=cols[0].unit, copy=False
        )
        # Get arguments needed to reconstruct class
        map = {
            key: (data[key] if key in "xyz" else getattr(cols[-1], key))
            for key in self._represent_as_dict_attrs
        }
        out = self._construct_from_dict(map)
        # Set remaining info attributes
        for attr, value in attrs.items():
            setattr(out.info, attr, value)

        return out


class MoonLocation(u.Quantity):
    """
    Location on the Moon.

    There are two ``selenocentric'' coordinate systems in common use:
        - The Mean Axis / Polar axis (ME) system defines the z-axis as the mean rotational
    axis of the Moon, while the prime meridian is set by the mean Earth direction.
        - The Principal Axes (PA) frame is defined by the principal axes of the Moon.

    This class uses the ME frame.

    Positions may be defined in Cartesian (x, y, z) coordinates with respect to the
    center of mass of the Moon, or in ``selenodetic'' coordinates (longitude, latitude, height).

    Selenodetic coordinates are defined with respect to a reference ellipsoid. The default is a
    sphere of radius 1731.1e3 km, but other ellipsoids are available. See lunarsky.moon.SELENOIDS.

    See:
        "A Standardized Lunar Coordinate System for the Lunar Reconnaissance
        Orbiter and Lunar Datasets"
        LRO Project and LGCWG White Paper Version 5, 2008 October 1
        (https://lunar.gsfc.nasa.gov/library/LunCoordWhitePaper-10-08.pdf)

    Notes
    -----
    This class fits into the coordinates transformation framework in that it
    encodes a position on the `~astropy.coordinates.MCMF` frame.  To get a
    proper `~astropy.coordinates.MCMF` object from this object, use the ``mcmf``
    property.
    """

    _ellipsoid = "SPHERE"
    _location_dtype = np.dtype({"names": ["x", "y", "z"], "formats": [np.float64] * 3})
    _array_dtype = np.dtype((np.float64, (3,)))

    # Manage the set of defined ephemerides.
    # Class attributes only
    _inuse_stat_ids = []
    _avail_stat_ids = None
    _existing_locs = []
    _ref_count = []

    # This instance's station id(s)
    station_ids = []

    info = MoonLocationInfo()

    def __new__(cls, *args, **kwargs):
        # TODO: needs copy argument and better dealing with inputs.
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], MoonLocation):
            return args[0].copy()
        try:
            self = cls.from_selenocentric(*args, **kwargs)
        except (u.UnitsError, TypeError) as exc_selenocentric:
            try:
                self = cls.from_selenodetic(*args, **kwargs)
            except Exception as exc_selenodetic:
                raise TypeError(
                    "Coordinates could not be parsed as either "
                    "selenocentric or selenodetic, with respective "
                    'exceptions "{}" and "{}"'.format(exc_selenocentric, exc_selenodetic)
                )
        return self

    @classmethod
    def _set_site_id(cls, inst):
        """
        Set the station ID number and manage registry of defined stations.
        """
        if inst.isscalar:
            llh_arr = [
                (
                    inst.lon.deg.item(),
                    inst.lat.deg.item(),
                    inst.height.to_value("km").item(),
                )
            ]
            ncrds = 1
        else:
            llh_arr = zip(inst.lon.deg, inst.lat.deg, inst.height.to_value("km"))
            ncrds = inst.lon.size

        statids = []
        if cls._avail_stat_ids is None:
            cls._avail_stat_ids = list(range(999, 0, -1))

        if len(cls._avail_stat_ids) < ncrds:
            raise ValueError("Too many unique MoonLocation objects open at once.")

        for llh in llh_arr:
            lonlatheight = "_".join(
                ["{:.4f}".format(ll) for ll in llh] + [inst._ellipsoid]
            )
            if lonlatheight not in cls._existing_locs:
                new_stat_id = cls._avail_stat_ids.pop()
                cls._existing_locs.append(lonlatheight)
                statids.append(new_stat_id)
                cls._inuse_stat_ids.append(new_stat_id)
                cls._ref_count.append(1)
            else:
                ind = cls._existing_locs.index(lonlatheight)
                cls._ref_count[ind] += 1
                statids.append(cls._inuse_stat_ids[ind])
        inst.station_ids = statids
        return inst

    def _set_station_id(self):
        """
        Run classmethod for setting station IDs.

        Convenience function used for testing mostly
        """
        self.__class__._set_site_id(self)

    @classmethod
    def from_selenocentric(cls, x, y, z, unit=None):
        """
        Location on the Moon, initialized from selenocentric coordinates.

        Parameters
        ----------
        x, y, z : `~astropy.units.Quantity` or array_like
            Cartesian coordinates.  If not quantities, ``unit`` should be given.
        unit : `~astropy.units.UnitBase` object or None
            Physical unit of the coordinate values.  If ``x``, ``y``, and/or
            ``z`` are quantities, they will be converted to this unit.

        Raises
        ------
        astropy.units.UnitsError
            If the units on ``x``, ``y``, and ``z`` do not match or an invalid
            unit is given.
        ValueError
            If the shapes of ``x``, ``y``, and ``z`` do not match.
        TypeError
            If ``x`` is not a `~astropy.units.Quantity` and no unit is given.
        """
        if unit is None:
            try:
                unit = x.unit
            except AttributeError:
                raise TypeError(
                    "Selenocentric coordinates should be Quantities "
                    "unless an explicit unit is given."
                )
        else:
            unit = u.Unit(unit)

        if unit.physical_type != "length":
            raise u.UnitsError(
                "Selenocentric coordinates should be in " "units of length."
            )

        try:
            x = u.Quantity(x, unit, copy=False)
            y = u.Quantity(y, unit, copy=False)
            z = u.Quantity(z, unit, copy=False)
        except u.UnitsError:
            raise u.UnitsError(
                "Selenocentric coordinate units should all be " "consistent."
            )

        x, y, z = np.broadcast_arrays(x, y, z)
        struc = np.empty(x.shape, cls._location_dtype)
        struc["x"], struc["y"], struc["z"] = x, y, z
        inst = super().__new__(cls, struc, unit, copy=False)

        return inst

    @classmethod
    def from_selenodetic(cls, lon, lat, height=0.0, ellipsoid=None):
        """
        Location on the Moon, from latitude and longitude.

        Parameters
        ----------
        lon : `~astropy.coordinates.Longitude` or float
            Lunar East longitude.  Can be anything that initialises an
            `~astropy.coordinates.Angle` object (if float, in degrees).
        lat : `~astropy.coordinates.Latitude` or float
            Lunar latitude.  Can be anything that initialises an
            `~astropy.coordinates.Latitude` object (if float, in degrees).
        height : `~astropy.units.Quantity` or float, optional
            Height above reference sphere (if float, in meters; default: 0).
            The reference sphere is a sphere of radius 1737.1 kilometers,
            from the center of mass of the Moon.
        ellipsoid : str, optional
            Name of the reference ellipsoid to use (default: 'SPHERE').
            Available ellipsoids are:  'SPHERE', 'GRAIL23', 'CE-1-LAM-GEO'.
            See docstrings for classes in ELLIPSOIDS dictionary for references.

        Raises
        ------
        astropy.units.UnitsError
            If the units on ``lon`` and ``lat`` are inconsistent with angular
            ones, or that on ``height`` with a length.
        ValueError
            If ``lon``, ``lat``, and ``height`` do not have the same shape, or

        Notes
        -----

        latitude is defined relative to an equator 90 degrees
        off from the mean rotation axis. Longitude is defined
        relative to a prime meridian, which is itself given by
        the mean position of the "sub-Earth" point on the lunar surface.

        For the conversion to selenocentric coordinates, the ERFA routine
        ``gd2gce`` is used.  See https://github.com/liberfa/erfa

        """
        ellipsoid = _check_ellipsoid(ellipsoid, default=cls._ellipsoid)
        lon = Longitude(lon, u.degree, copy=False).wrap_at(180 * u.degree)
        lat = Latitude(lat, u.degree, copy=False)
        # don't convert to m by default, so we can use the height unit below.
        if not isinstance(height, u.Quantity):
            height = u.Quantity(height, u.m, copy=False)

        if not lon.shape == lat.shape:
            raise ValueError(
                "Inconsistent quantity shapes: {}, {}".format(
                    str(lon.shape), str(lat.shape)
                )
            )

        # get selenocentric coordinates. Have to give one-dimensional array.

        selenodetic = SELENOIDS[ellipsoid](lon, lat, height, copy=False)
        xyz = selenodetic.to_cartesian().get_xyz(xyz_axis=-1) << height.unit
        self = xyz.view(cls._location_dtype, cls).reshape(selenodetic.shape)
        self.ellipsoid = ellipsoid
        return self

    def __str__(self):
        return self.__repr__()

    def copy(self):
        # Necessary to preserve station_ids list
        c = super().copy()
        c.station_ids = self.station_ids
        return c

    def __copy__(self):
        # Ensure that the station_ids are copied as well under shallow copy
        obj = copy.copy(super())
        obj.station_ids = self.station_ids
        return obj

    def __del__(self):
        # Remove this MoonLocation's station_ids from _inuse_stat_ids and
        # locations from _existing_locs.
        # Also clear the corresponding frames from spice variable pool.
        for si, stat_id in enumerate(self.station_ids):
            try:
                ind = self.__class__._inuse_stat_ids.index(stat_id)
            except ValueError:
                continue
            count = self.__class__._ref_count[ind]
            if count <= 1:
                sid = self.__class__._inuse_stat_ids.pop(ind)
                self.__class__._existing_locs.pop(ind)
                self.__class__._avail_stat_ids.insert(0, sid)
                remove_topo(stat_id)
                self.__class__._ref_count.pop(ind)
            else:
                self.__class__._ref_count[ind] -= 1

    @property
    def selenodetic(self):
        """Convert to selenodetic coordinates."""
        return self.to_selenodetic()

    @property
    def ellipsoid(self):
        """The default ellipsoid used to convert to selenodetic coordinates."""
        return self._ellipsoid

    @ellipsoid.setter
    def ellipsoid(self, ellipsoid):
        self._ellipsoid = _check_ellipsoid(ellipsoid)

    def to_selenodetic(self, ellipsoid=None):
        """Convert to selenodetic coordinates (lat, lon, height).

        Height is in reference to the ellipsoid.

        Returns
        -------
        (lon, lat, height) : tuple
            The tuple contains instances of `~astropy.coordinates.Longitude`,
            `~astropy.coordinates.Latitude`, and `~astropy.units.Quantity`

        """
        ellipsoid = _check_ellipsoid(ellipsoid, default=self.ellipsoid)
        xyz = self.view(self._array_dtype, u.Quantity)
        llh = CartesianRepresentation(xyz, xyz_axis=-1, copy=False).represent_as(
            SELENOIDS[ellipsoid],
        )
        return SelenodeticLocation(
            Longitude(llh.lon, u.degree, wrap_angle=180.0 * u.degree, copy=False),
            Latitude(llh.lat, u.degree, copy=False),
            u.Quantity(llh.height, self.unit, copy=False),
        )

    @property
    def lon(self):
        """Longitude of the location"""
        return self.selenodetic[0]

    @property
    def lat(self):
        """Longitude of the location"""
        return self.selenodetic[1]

    @property
    def height(self):
        """Height of the location"""
        return self.selenodetic[2]

    # mostly for symmetry with selenodetic and to_selenodetic.
    @property
    def selenocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities"""
        return self.to_selenocentric()

    def to_selenocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities"""
        return (self.x, self.y, self.z)

    def get_mcmf(self, obstime=None):
        """
        Generates a `~lunarsky.mcmf.MCMF` object with the location of
        this object at the requested ``obstime``.

        Parameters
        ----------
        obstime : `~lunarsky.time.Time` or None
            The ``obstime`` to apply to the new `~lunarsky.mcmf.MCMF`, or
            if None, the default ``obstime`` will be used.

        Returns
        -------
        mcmf : `~lunarsky.mcmf.MCMF`
            The new object in the MCMF frame
        """
        # do this here to prevent a series of complicated circular imports
        from . import MCMF

        return MCMF(x=self.x, y=self.y, z=self.z, obstime=obstime)

    mcmf = property(
        get_mcmf,
        doc="""An `~astropy.coordinates.MCMF` object  with
                                     for the location of this object at the
                                     default ``obstime``.""",
    )

    def get_mcmf_posvel(self, obstime):
        """
        Calculate the MCMF position and velocity of this object at the
        requested ``obstime``.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the GCRS position/velocity at.

        Returns
        --------
        obsgeoloc : `~astropy.coordinates.CartesianRepresentation`
            The GCRS position of the object
        obsgeovel : `~astropy.coordinates.CartesianRepresentation`
            The GCRS velocity of the object
        """
        # MCMF position
        mcmf_data = self.get_mcmf(obstime).data
        obspos = mcmf_data.without_differentials()
        obsvel = mcmf_data.differentials["s"].to_cartesian()
        return obspos, obsvel

    @property
    def x(self):
        """The X component of the selenocentric coordinates."""
        return self["x"]

    @property
    def y(self):
        """The Y component of the selenocentric coordinates."""
        return self["y"]

    @property
    def z(self):
        """The Z component of the selenocentric coordinates."""
        return self["z"]

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result.dtype is self.dtype:
            return result.view(self.__class__)
        else:
            return result.view(u.Quantity)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if hasattr(obj, "_ellipsoid"):
            self._ellipsoid = obj._ellipsoid

    def __len__(self):
        if self.shape == ():
            raise IndexError("0-d MoonLocation arrays cannot be indexed")
        else:
            return super().__len__()

    def _to_value(self, unit, equivalencies=[]):
        """Helper method for to and to_value."""
        # Conversion to another unit in both ``to`` and ``to_value`` goes
        # via this routine. To make the regular quantity routines work, we
        # temporarily turn the structured array into a regular one.
        array_view = self.view(self._array_dtype, np.ndarray)
        if equivalencies == []:
            equivalencies = self._equivalencies
        new_array = self.unit.to(unit, array_view, equivalencies=equivalencies)
        return new_array.view(self.dtype).reshape(self.shape)


class MoonLocationAttribute(Attribute):
    """
    A frame attribute that can act as a `~lunarsky.MoonLocation`.
    It can be created as anything that can be transformed to the
    `~lunarsky.MCMF` frame, but always presents as an `MoonLocation`
    when accessed after creation.

    Parameters
    ----------
    default : object
        Default value for the attribute if not provided
    secondary_attribute : str
        Name of a secondary instance attribute which supplies the value if
        ``default is None`` and no value was supplied during initialization.
    """

    def convert_input(self, value):
        """
        Checks that the input is a Quantity with the necessary units (or the
        special value ``0``).

        Parameters
        ----------
        value : object
            Input value to be converted.

        Returns
        -------
        out, converted : correctly-typed object, boolean
            Tuple consisting of the correctly-typed object and a boolean which
            indicates if conversion was actually performed.

        Raises
        ------
        ValueError
            If the input is not valid for this attribute.
        """

        if value is None:
            return None, False
        elif isinstance(value, MoonLocation):
            return value, False
        else:
            # we have to do the import here because of some tricky circular deps
            from . import MCMF

            if not hasattr(value, "transform_to"):
                raise ValueError(
                    '"{}" was passed into a '
                    "MoonLocationAttribute, but it does not have "
                    '"transform_to" method'.format(value)
                )
            mcmfobj = value.transform_to(MCMF())
            return mcmfobj.moon_location, True

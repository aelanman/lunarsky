import numpy as np
import copy
from astropy import units as u
from astropy.units.quantity import QuantityInfoBase
from astropy.coordinates.angles import Longitude, Latitude
from astropy.coordinates.earth import GeodeticLocation
from astropy.coordinates.representation import (
    CartesianRepresentation,
    SphericalRepresentation,
)
from astropy.coordinates.attributes import Attribute

from .spice_utils import remove_topo

__all__ = ["MoonLocation", "MoonLocationAttribute"]


_DEFAULT_SITE_ID = 98


class MoonLocationInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """

    _represent_as_dict_attrs = ("x", "y", "z")

    def _construct_from_dict(self, map):
        out = self._parent_cls(**map)
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
        # map is different enough that this needs its own rouinte.
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
    center of mass of the Moon, or in ``selenodetic'' coordinates (longitude, latitude).
    In selenodetic coordinates, positions are on the surface exactly.

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

    _location_dtype = np.dtype({"names": ["x", "y", "z"], "formats": [np.float64] * 3})
    _array_dtype = np.dtype((np.float64, (3,)))

    _lunar_radius = 1737.1e3  # m

    # Manage the set of defined ephemerides.
    # Class attributes only
    _existing_stat_ids = []
    _existing_locs = []
    _ref_count = []
    _new_stat_id = _DEFAULT_SITE_ID  # Starting at 98

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
        self = cls._set_site_id(self)
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
        else:
            llh_arr = zip(inst.lon.deg, inst.lat.deg, inst.height.to_value("km"))
        statids = []

        for llh in llh_arr:
            lonlatheight = "_".join(["{:.4f}".format(ll) for ll in llh])
            if lonlatheight not in cls._existing_locs:
                cls._existing_locs.append(lonlatheight)
                statids.append(cls._new_stat_id)
                cls._existing_stat_ids.append(cls._new_stat_id)
                cls._new_stat_id += 1
                cls._ref_count.append(1)
                if cls._new_stat_id >= 999:
                    raise ValueError("Too many MoonLocation objects open at once. ")
            else:
                ind = cls._existing_locs.index(lonlatheight)
                cls._ref_count[ind] += 1
                statids.append(cls._existing_stat_ids[ind])
        inst.station_ids = statids
        return inst

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

        inst = cls._set_site_id(inst)

        return inst

    @classmethod
    def from_selenodetic(cls, lon, lat, height=0.0):
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

        """
        lon = Longitude(lon, u.degree, wrap_angle=180 * u.degree, copy=False)
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

        lunar_radius = u.Quantity(cls._lunar_radius, u.m, copy=False)

        Npts = lon.size
        xyz = np.zeros((Npts, 3))
        xyz[:, 0] = (lunar_radius + height) * np.cos(lat) * np.cos(lon)
        xyz[:, 1] = (lunar_radius + height) * np.cos(lat) * np.sin(lon)
        xyz[:, 2] = (lunar_radius + height) * np.sin(lat)

        xyz = np.squeeze(xyz)

        self = xyz.ravel().view(cls._location_dtype, cls).reshape(xyz.shape[:-1])
        self._unit = u.meter
        inst = self.to(height.unit)

        return cls._set_site_id(inst)

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
        # Remove this MoonLocation's station_ids from _existing_stat_ids and
        # locations from _existing_locs.
        # Also clear the corresponding frames from spice variable pool.

        for si, stat_id in enumerate(self.station_ids):
            try:
                ind = self.__class__._existing_stat_ids.index(stat_id)
            except ValueError:
                continue
            count = self.__class__._ref_count[ind]
            if count == 1:
                self.__class__._existing_stat_ids.pop(ind)
                self.__class__._existing_locs.pop(ind)
                remove_topo(stat_id)
                self.__class__._ref_count.pop(ind)
            else:
                self.__class__._ref_count[ind] -= 1

    @property
    def selenodetic(self):
        """Convert to selenodetic coordinates."""
        return self.to_selenodetic()

    def to_selenodetic(self):
        """Convert to selenodetic coordinates (lat, lon, height).

        Height is in reference to a sphere with radius `_lunar_radius`,
        centered at the center of mass.

        Returns
        -------
        (lon, lat, height) : tuple
            The tuple contains instances of `~astropy.coordinates.Longitude`,
            `~astropy.coordinates.Latitude`, and `~astropy.units.Quantity`

        """
        xyz = self.view(self._array_dtype, u.Quantity)
        lld = CartesianRepresentation(xyz, xyz_axis=-1, copy=False).represent_as(
            SphericalRepresentation
        )
        return GeodeticLocation(
            Longitude(lld.lon, u.degree, wrap_angle=180.0 * u.degree, copy=False),
            Latitude(lld.lat, u.degree, copy=False),
            u.Quantity(
                lld.distance - (self._lunar_radius * u.meter), self.unit, copy=False
            ),
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

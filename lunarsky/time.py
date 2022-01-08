"""Extension of astropy's Time class."""

import numpy as np
import astropy
from astropy.coordinates import EarthLocation, Longitude

from .moon import MoonLocation
import spiceypy as spice

__all__ = ["Time", "TimeDelta"]


class Time(astropy.time.Time):
    _appended_docstr = """
    Notes
    -----
    location can also be given as a `lunarsky.MoonLocation` object.
    """

    __doc__ = astropy.time.Time.__doc__ + _appended_docstr

    def __init__(
        self,
        val,
        val2=None,
        format=None,
        scale=None,
        precision=None,
        in_subfmt=None,
        out_subfmt=None,
        location=None,
        copy=False,
    ):

        super_loc = None
        if isinstance(location, EarthLocation):
            super_loc = location

        super().__init__(
            val,
            val2=val2,
            format=format,
            scale=scale,
            precision=precision,
            in_subfmt=in_subfmt,
            out_subfmt=out_subfmt,
            location=super_loc,
            copy=copy,
        )

        if isinstance(location, MoonLocation):
            self.location = location

    def sidereal_time(self, kind, longitude=None, model=None):
        # Currently returns the zenith RA as the LST.
        if self.location is None or self.location is EarthLocation:
            return super().sidereal_time(kind, longitude=longitude, model=model)

        if model is not None:
            raise ValueError(
                "The 'model' keyword is not supported" "'for MoonLocation sidereal_times."
            )

        # From here on, proceed assuming longitude or location refer
        # to the selenodetic coordinate system. "self.location" must
        # be defined in order to get here.

        et = np.atleast_1d((self - Time("J2000")).sec)
        mats = np.array([spice.pxform("MOON_ME", "J2000", t) for t in et])

        # Zenith vector
        zvec = self.location.mcmf.cartesian.xyz
        uzvec = zvec / np.linalg.norm(zvec)

        newvec = np.dot(mats, uzvec)
        return Longitude(np.arctan2(newvec[..., 1], newvec[..., 0]).squeeze(), "rad")

    sidereal_time.__doc__ = (
        astropy.time.Time.sidereal_time.__doc__
        + """
        Notes
        -----
        If the location attribute is a `~lunarsky.MoonLocation` instance, the 'kind' keyword
        is ignored and the calculated sidereal time is given as the ICRS frame right ascension
        of zenith over the MoonLocation.
        """
    )

    def light_travel_time(self, *args, **kwargs):
        if isinstance(self.location, MoonLocation):
            raise ValueError(
                "Light travel time calculations are not" "yet supported for MoonLocation"
            )

        return super().light_travel_time(*args, **kwargs)


# For convenience in import statements, TimeDelta is also included in this namespace.
TimeDelta = astropy.time.TimeDelta

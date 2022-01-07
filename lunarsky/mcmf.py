import numpy as np
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import (
    CartesianRepresentation,
    CartesianDifferential,
    UnitSphericalRepresentation,
)
from astropy.coordinates.baseframe import (
    BaseCoordinateFrame,
    base_doc,
    frame_transform_graph,
)
from astropy.coordinates.attributes import TimeAttribute
import astropy.units as un
from astropy.time import Time

from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.builtin_frames.icrs import ICRS

import spiceypy as spice

__all__ = ["MCMF"]

DEFAULT_OBSTIME = Time("J2000", scale="tt")


@format_doc(base_doc, components="", footer="")
class MCMF(BaseCoordinateFrame):
    """
    A coordinate or frame in the lunar "Mean Earth/ Mean Rotation frame". This is a
    "Moon-Centered/Moon-Fixed" frame, defined by an X axis through the mean position
    of the Earth-Moon direction and a Z axis through the mean rotational axis.

    In JPL ephemeris data, this is called MOON_ME.
    """

    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential

    obstime = TimeAttribute(default=DEFAULT_OBSTIME)

    @property
    def moon_location(self):
        """
        The data in this frame as an `~astropy.coordinates.MoonLocation` class.
        """
        from .moon import MoonLocation

        cart = self.represent_as(CartesianRepresentation)
        return MoonLocation(x=cart.x, y=cart.y, z=cart.z)


# Transforms


def icrs_to_mcmf_mat(ets):
    # Rotation matrix from ICRS to MOON_ME
    # time = single astropy Time object.

    # Ephemeris time = seconds since J2000
    mat = np.stack([spice.pxform("J2000", "MOON_ME", et) for et in ets], axis=0)

    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, MCMF)
def icrs_to_mcmf(icrs_coo, mcmf_frame):
    is_unitspherical = (
        isinstance(icrs_coo.data, UnitSphericalRepresentation)
        or icrs_coo.cartesian.x.unit == un.one
    )

    icrs_coo_cart = icrs_coo.cartesian
    ets = np.atleast_1d((mcmf_frame.obstime - Time("J2000")).sec)
    if not is_unitspherical:
        # For positions in the solar system.
        mcmf_posvel = (
            np.stack([spice.spkgeo(301, et, "J2000", 0)[0] for et in ets]) * un.km
        )
        icrs_coo_cart -= CartesianRepresentation((mcmf_posvel[:, :3]).T)

    mat = icrs_to_mcmf_mat(ets)
    newrepr = icrs_coo_cart.transform(mat)

    if ets.shape != mcmf_frame.obstime.shape:
        newrepr = newrepr.reshape(icrs_coo.shape)

    return mcmf_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, ICRS)
def mcmf_to_icrs(mcmf_coo, icrs_frame):
    is_unitspherical = (
        isinstance(mcmf_coo.data, UnitSphericalRepresentation)
        or mcmf_coo.cartesian.x.unit == un.one
    )

    ets = np.atleast_1d((mcmf_coo.obstime - Time("J2000")).sec)
    mat = icrs_to_mcmf_mat(ets)
    newrepr = mcmf_coo.cartesian.transform(matrix_transpose(mat))

    if not is_unitspherical:
        mcmf_posvel = (
            np.stack([spice.spkgeo(301, et, "J2000", 0)[0] for et in ets]) * un.km
        )
        newrepr += CartesianRepresentation((mcmf_posvel[:, :3]).T)

    if ets.shape != mcmf_coo.obstime.shape:
        newrepr = newrepr.reshape(mcmf_coo.shape)
    return icrs_frame.realize_frame(newrepr)

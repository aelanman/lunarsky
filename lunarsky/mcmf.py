
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc, frame_transform_graph
from astropy.coordinates.attributes import TimeAttribute
from astropy.time import Time

from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.builtin_frames.icrs import ICRS

import spiceypy as spice

__all__ = ['MCMF']

DEFAULT_OBSTIME = Time('J2000', scale='tt')


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

def icrs_to_mcmf_mat(time):
    # Rotation matrix from ICRS to MOON_ME
    # time = single astropy Time object.

    # Ephemeris time = seconds since J2000
    et = (time - Time('J2000')).sec
    mat = spice.pxform('J2000', 'MOON_ME', et)

    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, MCMF)
def icrs_to_mcmf(icrs_coo, mcmf_frame):
    mat = icrs_to_mcmf_mat(mcmf_frame.obstime)
    newrepr = icrs_coo.cartesian.transform(mat)

    return mcmf_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, ICRS)
def mcmf_to_icrs(mcmf_coo, icrs_frame):
    mat = icrs_to_mcmf_mat(mcmf_coo.obstime)
    newrepr = mcmf_coo.cartesian.transform(matrix_transpose(mat))

    return icrs_frame.realize_frame(newrepr)

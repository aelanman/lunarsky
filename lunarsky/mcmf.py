import numpy as np
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import (
    CartesianRepresentation,
    CartesianDifferential,
    UnitSphericalRepresentation,
)
from astropy.utils import check_broadcast
from astropy.coordinates.baseframe import (
    BaseCoordinateFrame,
    base_doc,
    frame_transform_graph,
)
from astropy.coordinates.attributes import TimeAttribute
import astropy.units as un
from astropy.time import Time

from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.builtin_frames.icrs import ICRS

import spiceypy as spice

__all__ = ["MCMF"]

_J2000 = Time("J2000", scale="tt")


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

    obstime = TimeAttribute(default=_J2000)

    @property
    def moon_location(self):
        """
        The data in this frame as an `~astropy.coordinates.MoonLocation` class.
        """
        from .moon import MoonLocation

        cart = self.represent_as(CartesianRepresentation)
        return MoonLocation(x=cart.x, y=cart.y, z=cart.z)


# -----------------
# Helper functions
# -----------------


def make_transform(coo, toframe):

    ap_to_spice = {"icrs": ("J2000", 0), "mcmf": ("MOON_ME", 301)}

    # Get target frame and source frame names
    from_name, from_id = ap_to_spice[coo.name]
    to_name, to_id = ap_to_spice[toframe.name]

    to_mcmf = isinstance(toframe, MCMF)
    obstime = toframe.obstime if to_mcmf else coo.obstime

    # Make arrays
    ets = np.atleast_1d((obstime - _J2000).sec)
    shape_out = check_broadcast(coo.shape, ets.shape)

    coo_cart = coo.cartesian

    mats = np.stack([spice.pxform("J2000", "MOON_ME", et) for et in ets], axis=0)
    if not to_mcmf:
        mats = np.linalg.inv(mats)

    # Check for unitspherical
    is_unitspherical = (
        isinstance(coo.data, UnitSphericalRepresentation)
        or coo.cartesian.x.unit == un.one
    )

    # If not unitspherical, shift by origin vector before rotating.
    if not is_unitspherical:
        # Make origin vector(s) in coo's frame.
        orig_posvel = (
            np.asarray([spice.spkgeo(to_id, et, from_name, from_id)[0] for et in ets])
            * un.km
        )
        coo_cart -= CartesianRepresentation((orig_posvel.T)[:3])

    newrepr = coo_cart.transform(mats).reshape(shape_out)

    return toframe.realize_frame(newrepr)


# -----------------
# Transforms
# -----------------


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, MCMF)
def icrs_to_mcmf(icrs_coo, mcmf_frame):
    return make_transform(icrs_coo, mcmf_frame)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, ICRS)
def mcmf_to_icrs(mcmf_coo, icrs_frame):
    return make_transform(mcmf_coo, icrs_frame)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, MCMF)
def mcmf_to_mcmf(mcmf_coo, mcmf_frame):
    if np.all(mcmf_coo.obstime == mcmf_frame.obstime):
        return mcmf_frame.realize_frame(mcmf_coo.data)
    else:
        # Go through ICRS to ensure new time is accounted for.
        return mcmf_coo.transform_to(ICRS()).transform_to(mcmf_frame)

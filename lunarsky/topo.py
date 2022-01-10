import numpy as np
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import (
    SphericalRepresentation,
    SphericalCosLatDifferential,
    UnitSphericalRepresentation,
    CartesianRepresentation,
)
from astropy.coordinates.baseframe import (
    BaseCoordinateFrame,
    base_doc,
    frame_transform_graph,
)
from astropy.coordinates.attributes import TimeAttribute
from astropy.time import Time
from astropy import units as un
from astropy.coordinates.baseframe import RepresentationMapping
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.builtin_frames.icrs import ICRS

from .mcmf import MCMF
from .moon import MoonLocationAttribute
from .spice_utils import check_is_loaded, topo_frame_def, lunar_surface_ephem

import spiceypy as spice


_90DEG = 90 * un.deg
__all__ = ["LunarTopo"]

_J2000 = Time("J2000")
DEFAULT_OBSTIME = Time("J2000", scale="tt")

doc_components = """
    az : `~astropy.coordinates.Angle`, optional, must be keyword
        The Azimuth for this object (``alt`` must also be given and
        ``representation`` must be None).
    alt : `~astropy.coordinates.Angle`, optional, must be keyword
        The Altitude for this object (``az`` must also be given and
        ``representation`` must be None).
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight."""

doc_footer = """
    Other parameters
    ----------------
    obstime : `~astropy.time.Time`
        The time at which the observation is taken.  Used for determining the
        position and orientation of the Earth.
    location : `~lunarsky.MoonLocation`
        The location on the Moon.
    """


@format_doc(base_doc, components=doc_components, footer=doc_footer)
class LunarTopo(BaseCoordinateFrame):
    """
    An "East/North/Up" coordinate frame on the lunar surface, analogous to the
    AltAz frame in astropy.
    """

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "az"),
            RepresentationMapping("lat", "alt"),
        ]
    }

    default_representation = SphericalRepresentation
    default_differential = SphericalCosLatDifferential

    obstime = TimeAttribute(default=DEFAULT_OBSTIME)
    location = MoonLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def zen(self):
        """
        The zenith angle for this coordinate
        """
        return _90DEG.to(self.alt.unit) - self.alt


# Helper functions


def _make_mats(ets, frame0, frame1):
    return np.stack([spice.pxform(frame0, frame1, et) for et in ets], axis=0)


def _spice_setup(latitude, longitude):
    if not isinstance(latitude, (int, float)):
        latitude = latitude[0]
    if not isinstance(longitude, (int, float)):
        longitude = longitude[0]

    loadnew = True
    frameloaded = check_is_loaded("*LUNAR-TOPO*")
    if frameloaded:
        latlon = spice.gcpool("TOPO_LAT_LON", 0, 8)
        loadnew = not latlon == ["{:.4f}".format(ll) for ll in [latitude, longitude]]
    if loadnew:
        lunar_surface_ephem(latitude, longitude)  # Furnishes SPK for lunar surface point
        station_name, idnum, frame_specs, latlon = topo_frame_def(
            latitude, longitude, moon=True
        )
        spice.pcpool("TOPO_LAT_LON", latlon)
        frame_strs = ["{}={}".format(k, v) for (k, v) in frame_specs.items()]
        spice.lmpool(frame_strs)


# Transformations
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, LunarTopo)
def icrs_to_lunartopo(icrs_coo, topo_frame):

    _spice_setup(topo_frame.location.lat.deg, topo_frame.location.lon.deg)

    is_unitspherical = (
        isinstance(icrs_coo.data, UnitSphericalRepresentation)
        or icrs_coo.cartesian.x.unit == un.one
    )

    icrs_coo_cart = icrs_coo.cartesian
    ets = np.atleast_1d((topo_frame.obstime - _J2000).sec)
    if not is_unitspherical:
        # For positions in the solar system.
        # ICRS position of lunar surface point wrt solar system barycenter.
        lsp_icrs_posvel = (
            np.stack([spice.spkgeo(301098, et, "J2000", 0)[0] for et in ets]) * un.km
        )
        icrs_coo_cart -= CartesianRepresentation(
            (lsp_icrs_posvel[:, :3]).T
        )  # 0-2=pos, 3-5=vel

    mats = _make_mats(ets, "J2000", "LUNAR-TOPO")
    newrepr = icrs_coo_cart.transform(mats)

    # If a time axis was added, remove it:
    if ets.shape != topo_frame.obstime.shape:
        newrepr = newrepr.reshape(icrs_coo.shape)
    return topo_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, ICRS)
def lunartopo_to_icrs(topo_coo, icrs_frame):

    _spice_setup(topo_coo.location.lat.deg, topo_coo.location.lon.deg)

    is_unitspherical = (
        isinstance(topo_coo.data, UnitSphericalRepresentation)
        or topo_coo.cartesian.x.unit == un.one
    )

    ets = np.atleast_1d((topo_coo.obstime - _J2000).sec)
    mats = _make_mats(ets, "LUNAR-TOPO", "J2000")
    newrepr = topo_coo.cartesian.transform(mats)
    if not is_unitspherical:
        # For positions in the solar system.
        # ICRS position of lunar surface point wrt solar system barycenter.
        lsp_icrs_posvel = (
            np.stack([spice.spkgeo(301098, et, "J2000", 0)[0] for et in ets]) * un.km
        )
        newrepr += CartesianRepresentation((lsp_icrs_posvel[:, :3]).T)  # 0-2=pos, 3-5=vel

    # If a time axis was added, remove it:
    if ets.shape != topo_coo.obstime.shape:
        newrepr = newrepr.reshape(topo_coo.shape)
    return icrs_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, LunarTopo)
def mcmf_to_lunartopo(mcmf_coo, topo_frame):

    # TODO:
    #   > What if mcmf_coo and topo_frame have different obstimes?
    #   > What if location has obstime?
    _spice_setup(topo_frame.location.lat.deg, topo_frame.location.lon.deg)

    is_unitspherical = (
        isinstance(mcmf_coo.data, UnitSphericalRepresentation)
        or mcmf_coo.cartesian.x.unit == un.one
    )

    mcmf_coo_cart = mcmf_coo.cartesian
    ets = np.atleast_1d((topo_frame.obstime - _J2000).sec)
    if not is_unitspherical:
        # For positions in the solar system.
        # ICRS position of lunar surface point wrt selenocenter
        lsp_mcmf_posvel = (
            np.stack([spice.spkgeo(301098, et, "MOON_ME", 301)[0] for et in ets]) * un.km
        )
        #        import IPython; IPython.embed()
        mcmf_coo_cart -= CartesianRepresentation(
            (lsp_mcmf_posvel[:, :3]).T
        )  # 0-2=pos, 3-5=vel

    mats = _make_mats(ets, "MOON_ME", "LUNAR-TOPO")
    newrepr = mcmf_coo_cart.transform(mats)

    # If a time axis was added, remove it:
    if ets.shape != topo_frame.obstime.shape:
        newrepr = newrepr.reshape(mcmf_coo.shape)

    return topo_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, MCMF)
def lunartopo_to_mcmf(topo_coo, mcmf_frame):

    _spice_setup(topo_coo.location.lat.deg, topo_coo.location.lon.deg)

    is_unitspherical = (
        isinstance(topo_coo.data, UnitSphericalRepresentation)
        or topo_coo.cartesian.x.unit == un.one
    )

    mat = spice.pxform("LUNAR-TOPO", "MOON_ME", 0)  # Not time-dependent
    newrepr = topo_coo.cartesian.transform(mat)

    ets = np.atleast_1d((topo_coo.obstime - _J2000).sec)
    if not is_unitspherical:
        # For positions in the solar system.
        # Shift back to be relative to the selenocenter
        lsp_mcmf_posvel = (
            np.stack([spice.spkgeo(301098, et, "MOON_ME", 301)[0] for et in ets]) * un.km
        )
        newrepr += CartesianRepresentation((lsp_mcmf_posvel[:, :3]).T)  # 0-2=pos, 3-5=vel

    if ets.shape != topo_coo.obstime.shape:
        newrepr = newrepr.reshape(topo_coo.shape)
    return mcmf_frame.realize_frame(newrepr)

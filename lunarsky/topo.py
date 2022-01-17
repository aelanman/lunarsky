import numpy as np
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import (
    SphericalRepresentation,
    SphericalCosLatDifferential,
    UnitSphericalRepresentation,
    CartesianRepresentation,
)
from astropy.utils import check_broadcast
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

_J2000 = Time("J2000", scale="tt")

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

    obstime = TimeAttribute(default=_J2000)
    location = MoonLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def zen(self):
        """
        The zenith angle for this coordinate
        """
        return _90DEG.to(self.alt.unit) - self.alt


# -----------------
# Helper functions
# -----------------


def _spice_setup(latitude, longitude, station_id):

    latlonids = np.stack(
        [np.atleast_1d(latitude), np.atleast_1d(longitude), station_id]
    ).T
    if latlonids.ndim == 1:
        latlonids = latlonids[None, :]

    for lat, lon, sid in latlonids:
        sid = int(sid)  # Station IDs must be ints, but are converted to float above.
        frameloaded = check_is_loaded(f"*LUNAR-TOPO-{sid}*")
        if not frameloaded:
            lunar_surface_ephem(
                lat, lon, station_num=sid
            )  # Furnishes SPK for lunar surface point
            station_name, idnum, frame_specs, latlon = topo_frame_def(
                lat, lon, moon=True, station_num=sid
            )
            frame_strs = ["{}={}".format(k, v) for (k, v) in frame_specs.items()]
            spice.lmpool(frame_strs)


def make_transform(coo, toframe):

    ap_to_spice = {"icrs": ("J2000", 0), "mcmf": ("MOON_ME", 301)}
    # Get target frame and source frame names
    if isinstance(coo, LunarTopo):
        totopo = False
        frame_spice_name, frame_id = ap_to_spice[toframe.name]
        obstime = coo.obstime
        location = coo.location
    elif isinstance(toframe, LunarTopo):
        totopo = True
        frame_spice_name, frame_id = ap_to_spice[coo.name]
        obstime = toframe.obstime
        location = toframe.location

    # Make arrays
    ets = (obstime - _J2000).sec
    stat_ids = np.asarray(location.station_ids)
    shape_out = check_broadcast(coo.shape, ets.shape, location.shape)

    # Set up SPICE ephemerides and frame details
    _spice_setup(location.lat.deg, location.lon.deg, stat_ids)

    ets_ids = np.atleast_2d(np.stack(np.broadcast_arrays(ets, stat_ids)).T)

    coo_cart = coo.cartesian

    # Make rotation matrices
    mats = np.asarray(
        [
            spice.pxform(f"LUNAR-TOPO-{int(stat_id)}", frame_spice_name, et)
            for (et, stat_id) in ets_ids
        ]
    )
    if totopo:
        mats = np.linalg.inv(mats)

    # Check for unitspherical
    is_unitspherical = (
        isinstance(coo.data, UnitSphericalRepresentation)
        or coo.cartesian.x.unit == un.one
    )

    # If not unitspherical, shift by origin vector before rotating.
    if not is_unitspherical:
        # Make origin vector(s) in coo's frame.
        if totopo:
            origin_id = lambda n: int(frame_id)  # MCMF or ICRS frame origin
            target_id = lambda n: int(n) + 301000  # Station ID
            frame_name = lambda n: frame_spice_name

        else:
            origin_id = lambda n: int(n) + 301000
            target_id = lambda n: int(frame_id)
            frame_name = lambda n: f"LUNAR-TOPO-{int(n)}"

        orig_posvel = (
            np.asarray(
                [
                    spice.spkgeo(
                        target_id(stat_id), et, frame_name(stat_id), origin_id(stat_id)
                    )[0]
                    for (et, stat_id) in ets_ids
                ]
            )
            * un.km
        )

        coo_cart -= CartesianRepresentation((orig_posvel.T)[:3])

    newrepr = coo_cart.transform(mats).reshape(shape_out)

    return toframe.realize_frame(newrepr)


# -----------------
# Transformations
# -----------------
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, LunarTopo)
def icrs_to_lunartopo(icrs_coo, topo_frame):

    return make_transform(icrs_coo, topo_frame)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, ICRS)
def lunartopo_to_icrs(topo_coo, icrs_frame):

    return make_transform(topo_coo, icrs_frame)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, LunarTopo)
def mcmf_to_lunartopo(mcmf_coo, topo_frame):
    # TODO:
    #   > What if mcmf_coo and topo_frame have different obstimes?
    #   > What if location has obstime?

    return make_transform(mcmf_coo, topo_frame)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, MCMF)
def lunartopo_to_mcmf(topo_coo, mcmf_frame):

    return make_transform(topo_coo, mcmf_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference, LunarTopo, LunarTopo
)
def lunartopo_to_lunartopo(topo_coo, topo_frame):
    # TODO Should be possible to directly transform between different lunartopo frames.
    # For now, just go through MCMF
    return topo_coo.transform_to(MCMF(obstime=topo_coo.obstime)).transform_to(topo_frame)

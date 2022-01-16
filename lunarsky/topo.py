import numpy as np
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import (
    SphericalRepresentation,
    SphericalCosLatDifferential,
    UnitSphericalRepresentation,
    CartesianRepresentation,
)
from astropy.utils.shapes import check_broadcast
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


# Helper functions


def _make_mats(ets, frame0, frame1):
    return np.stack([spice.pxform(frame0, frame1, et) for et in ets], axis=0)


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


# Transformations
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, LunarTopo)
def icrs_to_lunartopo(icrs_coo, topo_frame):
    ets = (topo_frame.obstime - _J2000).sec
    stat_ids = np.asarray(topo_frame.location.station_ids)
    shape_out = check_broadcast(icrs_coo.shape, ets.shape, topo_frame.location.shape)

    _spice_setup(
        topo_frame.location.lat.deg,
        topo_frame.location.lon.deg,
        topo_frame.location.station_ids,
    )

    is_unitspherical = (
        isinstance(icrs_coo.data, UnitSphericalRepresentation)
        or icrs_coo.cartesian.x.unit == un.one
    )

    icrs_coo_cart = icrs_coo.cartesian

    ets_ids = np.atleast_2d(np.stack(np.broadcast_arrays(ets, stat_ids)).T)

    if not is_unitspherical:
        # For positions in the solar system.

        # ICRS position of lunar surface point wrt solar system barycenter.
        lsp_icrs_posvel = (
            np.asarray(
                [
                    spice.spkgeo(int(stat_id) + 301000, et, "J2000", 0)[0]
                    for (et, stat_id) in ets_ids
                ]
            )
            * un.km
        )
        icrs_coo_cart -= CartesianRepresentation(
            (lsp_icrs_posvel.T)[:3]  # 0-2=pos, 3-5=vel
        )

    mats = np.asarray(
        [
            spice.pxform("J2000", f"LUNAR-TOPO-{int(stat_id)}", et)
            for (et, stat_id) in ets_ids
        ]
    )
    newrepr = icrs_coo_cart.transform(mats)

    newrepr = newrepr.reshape(shape_out)

    return topo_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, ICRS)
def lunartopo_to_icrs(topo_coo, icrs_frame):

    ets = (topo_coo.obstime - _J2000).sec
    stat_ids = np.asarray(topo_coo.location.station_ids)
    shape_out = check_broadcast(topo_coo.shape, ets.shape, topo_coo.location.shape)

    _spice_setup(
        topo_coo.location.lat.deg,
        topo_coo.location.lon.deg,
        topo_coo.location.station_ids,
    )

    is_unitspherical = (
        isinstance(topo_coo.data, UnitSphericalRepresentation)
        or topo_coo.cartesian.x.unit == un.one
    )

    ets_ids = np.atleast_2d(np.stack(np.broadcast_arrays(ets, stat_ids)).T)

    mats = np.asarray(
        [
            spice.pxform(f"LUNAR-TOPO-{int(stat_id)}", "J2000", et)
            for (et, stat_id) in ets_ids
        ]
    )
    newrepr = topo_coo.cartesian.transform(mats)
    if not is_unitspherical:
        # For positions in the solar system.

        lsp_icrs_posvel = (
            np.asarray(
                [
                    spice.spkgeo(int(stat_id) + 301000, et, "J2000", 0)[0]
                    for (et, stat_id) in ets_ids
                ]
            )
            * un.km
        )
        newrepr += CartesianRepresentation((lsp_icrs_posvel.T)[:3])  # 0-2=pos, 3-5=vel

    newrepr = newrepr.reshape(shape_out)

    return icrs_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, LunarTopo)
def mcmf_to_lunartopo(mcmf_coo, topo_frame):

    # TODO:
    #   > What if mcmf_coo and topo_frame have different obstimes?
    #   > What if location has obstime?

    ets = (topo_frame.obstime - _J2000).sec
    stat_ids = np.asarray(topo_frame.location.station_ids)
    shape_out = check_broadcast(mcmf_coo.shape, ets.shape, topo_frame.location.shape)

    _spice_setup(topo_frame.location.lat.deg, topo_frame.location.lon.deg, stat_ids)

    is_unitspherical = (
        isinstance(mcmf_coo.data, UnitSphericalRepresentation)
        or mcmf_coo.cartesian.x.unit == un.one
    )

    ets_ids = np.atleast_2d(np.stack(np.broadcast_arrays(ets, stat_ids)).T)

    mcmf_coo_cart = mcmf_coo.cartesian
    if not is_unitspherical:
        # For positions in the solar system.
        # MCMF position of lunar surface point wrt selenocenter
        lsp_mcmf_posvel = (
            np.asarray(
                [
                    spice.spkgeo(int(stat_id) + 301000, et, "MOON_ME", 301)[0]
                    for (et, stat_id) in ets_ids
                ]
            )
            * un.km
        )
        mcmf_coo_cart -= CartesianRepresentation(
            (lsp_mcmf_posvel.T)[:3]  # 0-2=pos, 3-5=vel
        )

    mats = np.asarray(
        [
            spice.pxform("MOON_ME", f"LUNAR-TOPO-{int(stat_id)}", et)
            for (et, stat_id) in ets_ids
        ]
    )

    newrepr = mcmf_coo_cart.transform(mats).reshape(shape_out)

    return topo_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, MCMF)
def lunartopo_to_mcmf(topo_coo, mcmf_frame):

    ets = (topo_coo.obstime - _J2000).sec
    stat_ids = np.asarray(topo_coo.location.station_ids)
    shape_out = check_broadcast(topo_coo.shape, ets.shape, topo_coo.location.shape)

    _spice_setup(topo_coo.location.lat.deg, topo_coo.location.lon.deg, stat_ids)

    is_unitspherical = (
        isinstance(topo_coo.data, UnitSphericalRepresentation)
        or topo_coo.cartesian.x.unit == un.one
    )

    ets_ids = np.atleast_2d(np.stack(np.broadcast_arrays(ets, stat_ids)).T)

    mats = np.asarray(
        [
            spice.pxform(f"LUNAR-TOPO-{int(stat_id)}", "MOON_ME", et)
            for (et, stat_id) in ets_ids
        ]
    )
    newrepr = topo_coo.cartesian.transform(mats)

    if not is_unitspherical:
        # For positions in the solar system.

        # Shift back to be relative to the selenocenter
        lsp_mcmf_posvel = (
            np.asarray(
                [
                    spice.spkgeo(int(stat_id) + 301000, et, "MOON_ME", 301)[0]
                    for (et, stat_id) in ets_ids
                ]
            )
            * un.km
        ).squeeze()
        newrepr += CartesianRepresentation((lsp_mcmf_posvel.T)[:3])  # 0-2=pos, 3-5=vel

    newrepr = newrepr.reshape(shape_out)
    return mcmf_frame.realize_frame(newrepr)


# Enable Topo -> Topo transformations (such that the obstime or location can change)
frame_transform_graph._add_merged_transform(LunarTopo, MCMF, LunarTopo)

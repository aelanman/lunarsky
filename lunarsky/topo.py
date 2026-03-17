import numpy as np
import warnings
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import (
    SphericalRepresentation,
    SphericalCosLatDifferential,
    UnitSphericalRepresentation,
    CartesianRepresentation,
)
from astropy.utils import exceptions
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
from .spice_utils import (
    j2000_to_moon_me,
    moon_me_to_j2000,
    station_pos_ssb_j2000,
    topo_rotation_matrix,
)

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


def _get_topo_data(location):
    """
    Compute topo rotation matrices and MOON_ME positions for each location element.

    Returns
    -------
    topo_mats : ndarray, shape (M, 3, 3)
        ME-to-topo rotation matrices.
    pos_mes : ndarray, shape (M, 3)
        Station positions in MOON_ME, in km.
    """
    locations = np.atleast_1d(location)
    topo_mats = []
    pos_mes = []
    for loc in locations:
        topo_mats.append(topo_rotation_matrix(loc.lat.deg, loc.lon.deg))
        pos_mes.append([loc.x.to_value("km"), loc.y.to_value("km"), loc.z.to_value("km")])
    return np.array(topo_mats), np.array(pos_mes)


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

    if location is None:
        raise ValueError("location must be defined for LunarTopo transformations")

    # Make arrays
    ets = (obstime - _J2000).sec
    topo_mats, pos_mes = _get_topo_data(location)
    loc_indices = np.arange(len(topo_mats))
    shape_out = np.broadcast_shapes(coo.shape, ets.shape, location.shape)

    ets_locs = np.atleast_2d(np.stack(np.broadcast_arrays(ets, loc_indices)).T)

    coo_cart = coo.cartesian

    # Make rotation matrices
    # TOPO->ME is topo_matrix.T (constant); TOPO->J2000 = R(ME->J2000) @ topo_matrix.T
    mats_list = []
    for et, loc_idx in ets_locs:
        loc_idx = int(loc_idx)
        topo_mat = topo_mats[loc_idx]
        if frame_spice_name == "MOON_ME":
            mats_list.append(topo_mat.T)
        else:
            me_to_j2000 = moon_me_to_j2000(np.array([et]))[0]
            mats_list.append(me_to_j2000 @ topo_mat.T)
    mats = np.asarray(mats_list)
    if totopo:
        mats = np.linalg.inv(mats)

    # Check for unitspherical
    is_unitspherical = (
        isinstance(coo.data, UnitSphericalRepresentation)
        or coo.cartesian.x.unit == un.one
    )

    # If not unitspherical, shift by origin vector before rotating.
    if not is_unitspherical:
        orig_pos_list = []
        for et, loc_idx in ets_locs:
            loc_idx = int(loc_idx)
            pos_me = pos_mes[loc_idx]
            topo_mat = topo_mats[loc_idx]
            et_arr = np.array([et])

            if totopo:
                # Origin = station position relative to frame origin, in coo's frame
                if frame_id == 0:  # SSB / ICRS
                    orig_pos_list.append(station_pos_ssb_j2000(pos_me, et_arr)[0])
                else:  # Moon center / MCMF
                    orig_pos_list.append(pos_me)
            else:
                # Origin = frame origin relative to station, in coo's frame (topo)
                if frame_id == 0:  # SSB
                    station_ssb = station_pos_ssb_j2000(pos_me, et_arr)[0]
                    ssb_station_j2000 = -station_ssb
                    # Rotate to topo
                    j2000_me = j2000_to_moon_me(et_arr)[0]
                    ssb_station_me = j2000_me @ ssb_station_j2000
                    orig_pos_list.append(topo_mat @ ssb_station_me)
                else:  # Moon center
                    orig_pos_list.append(-(topo_mat @ pos_me))

        orig_pos = np.asarray(orig_pos_list) * un.km
        coo_cart -= CartesianRepresentation(orig_pos.T)

    newrepr = coo_cart.transform(mats).reshape(shape_out)

    if is_unitspherical:
        if not np.allclose(newrepr.norm(), 1.0):
            warnings.warn(
                "Coordinates do not all have unit magnitude, but will be treated as unit spherical,"
                "Define coordinates as Quantity or normalize to remove this warning.",
                exceptions.AstropyUserWarning,
            )
        newrepr = newrepr.represent_as(UnitSphericalRepresentation)

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

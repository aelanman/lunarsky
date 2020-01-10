
from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import SphericalRepresentation, SphericalCosLatDifferential
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc
from astropy.coordinates.attributes import TimeAttribute
from astropy.time import Time
from astropy.units import deg
from astropy.coordinates.baseframe import frame_transform_graph, RepresentationMapping
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.builtin_frames.icrs import ICRS

from .mcmf import MCMF
from .moon import MoonLocationAttribute
from .kernel_manager import check_is_loaded, topo_frame_def

import spiceypy as spice


_90DEG = 90 * deg
__all__ = ['LunarTopo']

_J2000 = Time("J2000")

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
    location : `~astropy.coordinates.EarthLocation`
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame.
    """


@format_doc(base_doc, components=doc_components, footer=doc_footer)
class LunarTopo(BaseCoordinateFrame):
    """
    An "East/North/Up" coordinate frame on the lunar surface, analogous to the
    AltAz frame in astropy.
    """

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping('lon', 'az'),
            RepresentationMapping('lat', 'alt')
        ]
    }

    default_representation = SphericalRepresentation
    default_differential = SphericalCosLatDifferential

    obstime = TimeAttribute(default=None)
    location = MoonLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def secz(self):
        """
        Secant if the zenith angle for this coordinate, a common estimate of the
        airmass.
        """
        return 1/np.sin(self.alt)

    @property
    def zen(self):
        """
        The zenith angle for this coordinate
        """
        return _90DEG.to(self.alt.unit) - self.alt


# Transformations

def _spice_setup(latitude, longitude):
    if not check_is_loaded('*LUNAR-TOPO*'):
        station_name, idnum, frame_specs = topo_frame_def(latitude, longitude, moon=True)
        frame_strs = ["{}={}".format(k,v) for (k,v) in frame_specs.values()]
        spice.lmpool(frame_specs)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, LunarTopo)
def icrs_to_lunartopo(icrs_coo, topo_frame):

    _spice_setup(topo_frame.location.lat.deg, topo_frame.location.lon.deg)

    et = (topo_frame.obstime - _J2000).sec
    mat = spice.pxform('J2000', 'LUNAR-TOPO', et)
    newrepr = icrs_coo.cartesian.transform(mat)

    return topo_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, ICRS)
def lunartopo_to_icrs(topo_coo, icrs_frame):

    _spice_setup(topo_coo.location.lat.deg, topo_coo.location.lon.deg)

    et = (topo_coo.obstime - _J2000).sec
    mat = spice.pxform('LUNAR-TOPO', 'J2000', et)
    newrepr = topo_coo.cartesian.transform(mat)

    return icrs_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, MCMF, LunarTopo)
def mcmf_to_lunartopo(mcmf_coo, topo_frame):

    _spice_setup(topo_frame.location.lat.deg, topo_frame.location.lon.deg)

    mat = spice.pxform('MOON_ME', 'LUNAR-TOPO', 0)
    newrepr = mcmf_coo.cartesian.transform(mat)

    return topo_frame.realize_frame(newrepr)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, LunarTopo, MCMF)
def lunartopo_to_mcmf(topo_coo, mcmf_frame):

    _spice_setup(topo_coo.location.lat.deg, topo_coo.location.lon.deg)

    mat = spice.pxform('LUNAR-TOPO', 'MOON_ME', 0)   # Not time-dependent

    newrepr = topo_coo.cartesian.transform(mat)

    return mcmf_frame.realize_frame(newrepr)

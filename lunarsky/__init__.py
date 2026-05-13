from .moon import *  # noqa
from .mcmf import *  # noqa
from .topo import *  # noqa
from .time import *  # noqa
from .sky_coordinate import SkyCoord  # noqa

from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.attributes import EarthLocationAttribute

if "location" in frame_transform_graph.frame_attributes:
    frame_transform_graph.frame_attributes["location"] = EarthLocationAttribute(
        default=None
    )

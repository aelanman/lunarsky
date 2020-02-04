
from .moon import *
from .mcmf import *
from .topo import *
from .sky_coordinate import SkyCoord

from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.attributes import Attribute, EarthLocationAttribute

if 'location' in frame_transform_graph.frame_attributes:
    frame_transform_graph.frame_attributes['location'] = EarthLocationAttribute(default=None)

# Adjust the SkyCoord object to work with either MoonLocation or EarthLocation

from numpy import sqrt
from astropy.coordinates import (
    EarthLocationAttribute,
    EarthLocation,
    SkyCoord as aSkyCoord,
)
from astropy.coordinates import UnitSphericalRepresentation
from astropy.coordinates.baseframe import (
    BaseCoordinateFrame,
    frame_transform_graph,
    GenericFrame,
)
from astropy.coordinates.sky_coordinate_parsers import _get_frame_class
from astropy.constants import c as speed_of_light

from .moon import MoonLocation, MoonLocationAttribute
from .topo import LunarTopo


class SkyCoord(aSkyCoord):
    def __init__(self, *args, **kwargs):
        loc = kwargs.get("location", None)
        if isinstance(loc, MoonLocation):
            frame_transform_graph.frame_attributes["location"] = MoonLocationAttribute(
                default=None
            )
        elif isinstance(loc, EarthLocation):
            frame_transform_graph.frame_attributes["location"] = EarthLocationAttribute(
                default=None
            )
        super().__init__(*args, **kwargs)
        # Set the graph to its default
        frame_transform_graph.frame_attributes["location"] = EarthLocationAttribute(
            default=None
        )

    def transform_to(self, frame, merge_attributes=True):
        # a modified version of the corresponding astropy function.
        from astropy.coordinates.errors import ConvertError

        frame_kwargs = {}

        # Frame name (string) or frame class?  Coerce into an instance.
        try:
            frame = _get_frame_class(frame)()
        except Exception:
            pass

        if isinstance(frame, SkyCoord):
            frame = frame.frame  # Change to underlying coord frame instance

        if isinstance(frame, BaseCoordinateFrame):
            new_frame_cls = frame.__class__
            # Get frame attributes, allowing defaults to be overridden by
            # explicitly set attributes of the source if ``merge_attributes``.
            for attr in frame_transform_graph.frame_attributes:
                self_val = getattr(self, attr, None)
                frame_val = getattr(frame, attr, None)
                if frame_val is not None and not (
                    merge_attributes and frame.is_frame_attr_default(attr)
                ):
                    frame_kwargs[attr] = frame_val
                elif self_val is not None and not self.is_frame_attr_default(attr):
                    frame_kwargs[attr] = self_val
                elif frame_val is not None:
                    frame_kwargs[attr] = frame_val
        else:
            raise ValueError("Transform `frame` must be a frame name, class, or instance")

        # Hacky solution here -- Frames other than LunarTopo cannot accept a MoonLocation object
        # and can get confused by it. Do not pass along `location` unless certain it will work.
        moonloc_incompatible = not isinstance(frame, LunarTopo)
        graph_attrs = frame_transform_graph.frame_attributes
        if hasattr(self, "location"):
            loc = getattr(self, "location")
            if isinstance(loc, MoonLocation):
                if moonloc_incompatible:
                    frame_kwargs.pop("location")
            elif "location" in graph_attrs.keys() and isinstance(
                graph_attrs["location"], MoonLocationAttribute
            ):
                frame_transform_graph.frame_attributes[
                    "location"
                ] = EarthLocationAttribute(default=None)

        # Get the composite transform to the new frame
        trans = frame_transform_graph.get_transform(self.frame.__class__, new_frame_cls)
        if trans is None:
            raise ConvertError(
                "Cannot transform from {} to {}".format(
                    self.frame.__class__, new_frame_cls
                )
            )

        # Make a generic frame which will accept all the frame kwargs that
        # are provided and allow for transforming through intermediate frames
        # which may require one or more of those kwargs.
        generic_frame = GenericFrame(frame_kwargs)

        # Do the transformation, returning a coordinate frame of the desired
        # final type (not generic).
        new_coord = trans(self.frame, generic_frame)

        # Finally make the new SkyCoord object from the `new_coord` and
        # remaining frame_kwargs that are not frame_attributes in `new_coord`.
        for attr in set(new_coord.get_frame_attr_names()) & set(frame_kwargs.keys()):
            frame_kwargs.pop(attr)

        return self.__class__(new_coord, **frame_kwargs)

    def radial_velocity_correction(self, kind="barycentric", obstime=None, location=None):

        # Need to copy location parsing syntax first.
        # location validation
        timeloc = getattr(obstime, "location", None)
        if location is None:
            if self.location is not None:
                location = self.location
                if timeloc is not None:
                    raise ValueError(
                        "`location` cannot be in both the "
                        "passed-in `obstime` and this `SkyCoord` "
                        "because it is ambiguous which is meant "
                        "for the radial_velocity_correction."
                    )
            elif timeloc is not None:
                location = timeloc
            else:
                raise TypeError(
                    "Must provide a `location` to "
                    "radial_velocity_correction, either as a "
                    "SkyCoord frame attribute, as an attribute on "
                    "the passed in `obstime`, or in the method "
                    "call."
                )

        elif self.location is not None or timeloc is not None:
            raise ValueError(
                "Cannot compute radial velocity correction if "
                "`location` argument is passed in and there is "
                "also a  `location` attribute on this SkyCoord or "
                "the passed-in `obstime`."
            )

        # Use parent method if given an EarthLocation
        if isinstance(location, EarthLocation):
            return super().radial_velocity_correction(
                kind=kind, obstime=obstime, location=location
            )

        from astropy.coordinates.solar_system import get_body_barycentric_posvel
        from astropy.coordinates import solar_system_ephemeris

        solar_system_ephemeris.set("jpl")

        # With a MoonLocation:
        # Get ICRS (barycentric) cartesian position and velocity of the Earth
        pos_moon, v_moon = get_body_barycentric_posvel("moon", obstime)
        if kind == "barycentric":
            v_origin_to_moon = v_moon
        elif kind == "heliocentric":
            v_sun = get_body_barycentric_posvel("sun", obstime)[1]
            v_origin_to_moon = v_moon - v_sun
        else:
            raise ValueError(
                "`kind` argument to radial_velocity_correction must "
                "be 'barycentric' or 'heliocentric', but got "
                "'{}'".format(kind)
            )

        mcmf_p, mcmf_v = location.get_mcmf_posvel(obstime)
        # transforming to GCRS is not the correct thing to do here, since we don't want to
        # include aberration (or light deflection)? Instead, only apply parallax if necessary
        if self.data.__class__ is UnitSphericalRepresentation:
            targcart = self.mcmf.cartesian
        else:
            # skycoord has distances so apply parallax
            obs_mcmf_cart = pos_moon + mcmf_p
            mcmf_cart = self.mcmf.cartesian
            targcart = mcmf_cart - obs_mcmf_cart
            targcart /= targcart.norm()

        if kind == "barycentric":
            beta_obs = (v_origin_to_moon + mcmf_v) / speed_of_light
            gamma_obs = 1 / sqrt(1 - beta_obs.norm() ** 2)
            gr = location.gravitational_redshift(obstime)
            # barycentric redshift according to eq 28 in Wright & Eastmann (2014),
            # neglecting Shapiro delay and effects of the star's own motion
            zb = gamma_obs * (1 + targcart.dot(beta_obs)) / (1 + gr / speed_of_light) - 1
            return zb * speed_of_light
        else:
            # do a simpler correction ignoring time dilation and gravitational redshift
            # this is adequate since Heliocentric corrections shouldn't be used if
            # cm/s precision is required.
            return targcart.dot(v_origin_to_moon + mcmf_v)

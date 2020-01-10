
import numpy as np
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates import SkyCoord, AltAz, ICRS, EarthLocation, Angle
from astropy.utils.data import download_files_in_parallel
import lunarsky
import lunarsky.tests as ltests
import spiceypy as spice


def test_spice_earth():
    # Replace the ICRS->AltAz transform in astropy with one using SPICE.
    # Confirm that star positions are the same as with the original transform
    # to within the error due to relativistic aberration (~ 21 arcsec)

    stars = ltests.get_catalog()

    lat, lon = 30, 25

    loc = EarthLocation.from_geodetic(lon, lat)

    altaz = stars.transform_to(AltAz(location=loc))

    trans_path, steps = frame_transform_graph.find_shortest_path(ICRS, AltAz)
    assert steps == 2

    # Make the Earth topo frame in spice.
    framename, idnum, frame_dict = lunarsky.kernel_manager.topo_frame_def(lat, lon, moon=False)

    # One more kernel is needed for the ITRF93 frame.
    kname = 'pck/earth_latest_high_prec.bpc'
    kurl = [lunarsky.kernel_manager._naif_kernel_url + '/' + kname]
    kernpath = download_files_in_parallel(kurl, cache=True, show_progress=False, pkgname='lunarsky')
    spice.furnsh(kernpath)

    frame_strs = ["{}={}".format(k, v) for (k, v) in frame_dict.items()]
    spice.lmpool(frame_strs)

    @frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, AltAz)
    def icrs_to_mcmf(icrs_coo, mcmf_frame):

        mat = spice.pxform('J2000', framename, 0)
        newrepr = icrs_coo.cartesian.transform(mat)

        return mcmf_frame.realize_frame(newrepr)

    trans_path2, steps2 = frame_transform_graph.find_shortest_path(ICRS, AltAz)
    assert steps2 == 1

    altaz2 = stars.transform_to(AltAz(location=loc))

    # Having done the transform, remove the spice transform from the graph
    frame_transform_graph.remove_transform(ICRS, AltAz, None)
    trans_path2, steps2 = frame_transform_graph.find_shortest_path(ICRS, AltAz)
    assert steps2 == 2

    astro_enu_vecs = altaz.cartesian.xyz.value
    spice_enu_vecs = altaz2.cartesian.xyz.value
    dots = np.array([np.dot(astro_enu_vecs[:, mi], spice_enu_vecs[:, mi]) for mi in range(stars.size)])
    dev_angs_arcsec = 3600 * np.degrees(np.arccos(dots))

    assert ltests.positions_close(altaz, altaz2, Angle(25, 'arcsec'))

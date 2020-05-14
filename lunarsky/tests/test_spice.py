

from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates import AltAz, ICRS, EarthLocation, Angle
from astropy.utils.data import download_files_in_parallel
import lunarsky
import lunarsky.tests as ltests
import lunarsky.spice_utils as spice_utils


import spiceypy as spice


def test_topo_frame_setup():
    # Check that the frame setup puts all the correct values in the kernel pool.

    latitude, longitude = 30, 25
    name, idnum, frame_dict, latlon = spice_utils.topo_frame_def(latitude, longitude)
    frame_strs = ["{}={}".format(k, v) for (k, v) in frame_dict.items()]
    spice.lmpool(frame_strs)

    for k, v in frame_dict.items():
        N, typecode = spice.dtpool(k)
        if typecode == 'N':
            res = spice.gdpool(k, 0, 100)
            if len(res) == 1:
                res = [int(res[0])]
            if N > 1:
                v = [float(it) for it in v.split(' ')]
                res = res.tolist()
            else:
                res = res[0]
                v = int(v)
            assert v == res
        else:
            res = spice.gcpool(k, 0, 100)[0]
            v = v.replace('\'', '')
            assert v == res


def test_kernel_paths():
    # Check that the correct kernel files are downloaded
    # Need to unhash the file names

    assert len(lunarsky.spice_utils.KERNEL_PATHS) == 3


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
    framename, idnum, frame_dict, latlon = lunarsky.spice_utils.topo_frame_def(lat, lon, moon=False)

    # One more kernel is needed for the ITRF93 frame.
    kname = 'pck/earth_latest_high_prec.bpc'
    kurl = [lunarsky.spice_utils._naif_kernel_url + '/' + kname]
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

    assert ltests.positions_close(altaz, altaz2, Angle(25, 'arcsec'))


def test_topo_kernel_setup():
    # Tests a single function in topo.py
    # Need to clear the kernel pool first

    spice.clpool()
    # Confirm no variables are loaded.
    assert not lunarsky.spice_utils.check_is_loaded("*")

    for filepath in lunarsky.spice_utils.KERNEL_PATHS:
        spice.furnsh(filepath)
    lat, lon = 30, 20
    lunarsky.topo._spice_setup(lat, lon)
    station_name, idnum, frame_specs, latlon =\
        lunarsky.spice_utils.topo_frame_def(lat, lon, moon=True)
    assert lunarsky.spice_utils.check_is_loaded('*{}*'.format(idnum))

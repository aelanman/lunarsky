
import numpy as np
from astropy.utils.data import download_files_in_parallel
from astropy.coordinates.matrix_utilities import rotation_matrix

import spiceypy as spice

_naif_kernel_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels'


def check_is_loaded(search):
    """
    Search the kernel pool variable names for a given string.
    """

    try:
        spice.gnpool(search, 0, 100)
    except(spice.support_types.SpiceyError):
        return False

    return True


def download_kernels(furnish=True):
    kernel_names = ['pck/moon_pa_de421_1900-2050.bpc',
                    'fk/satellites/moon_080317.tf',
                    'fk/satellites/moon_assoc_me.tf']

    kernel_urls = [_naif_kernel_url + '/' + kn for kn in kernel_names]
    kernel_paths = download_files_in_parallel(kernel_urls, cache=True, pkgname='lunarsky', show_progress=False)

    if furnish:
        for kern in kernel_paths:
            spice.furnsh(kern)

    return kernel_paths


def topo_frame_def(latitude, longitude, moon=False):
    """
    Make a list of strings defining a topocentric frame. This can then be loaded
    with spiceypy.lmpool.
    """

    if moon:
        idnum = 1301000
        station_name = 'LUNAR-TOPO'
        relative = 'MOON_ME'
    else:
        idnum = 1399000
        station_name = 'EARTH-TOPO'
        relative = 'ITRF93'

    # The DSS stations are built into SPICE, and they number up to 66.
    # We will call this station number 98.
    station_num = 98
    idnum += station_num
    fm_center_id = idnum - 1000000

#    ecef_to_enu = Rotation.from_euler('zyx', [-longitude, latitude, 0], degrees=True).as_matrix()
    ecef_to_enu = np.matmul(rotation_matrix(-longitude, 'z', unit='deg'), rotation_matrix(latitude, 'y', unit='deg')).T

#    # This reorders the axes to match the XYZ axes of the enu frame.
    ecef_to_enu = ecef_to_enu[[2, 1, 0]]


    mat = " ".join(map("{:.7f}".format, ecef_to_enu.flatten()))

    fmt_strs = [
            "FRAME_{1}                     = {0}",
            "FRAME_{0}_NAME                = '{1}'",
            "FRAME_{0}_CLASS                   = 4",
            "FRAME_{0}_CLASS_ID                = {0}",
            "FRAME_{0}_CENTER                  = {2}",
            "OBJECT_{2}_FRAME                   = '{1}'",
            "TKFRAME_{0}_RELATIVE          = '{3}'",
            "TKFRAME_{0}_SPEC              = 'MATRIX'",
            "TKFRAME_{0}_MATRIX            = {4}"
        ]
    frame_specs = [s.format(idnum, station_name, fm_center_id, relative, mat) for s in fmt_strs]

    return station_name, idnum, frame_specs


def cleanup():
    # TODO Clear the kernel pool
    return 0


KERNEL_PATHS = download_kernels()


import numpy as np
from astropy.utils.data import download_files_in_parallel, get_cached_urls, download_file
from astropy.coordinates.solar_system import solar_system_ephemeris
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time
import astropy.units as unit

import spiceypy as spice

from .mcmf import MCMF

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
    kernel_paths = download_files_in_parallel(
        kernel_urls, cache=True, pkgname='lunarsky', show_progress=False
    )

    if furnish:
        for kern in kernel_paths:
            spice.furnsh(kern)

    return kernel_paths


def topo_frame_def(latitude, longitude, moon=True):
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

    ecef_to_enu = np.matmul(
        rotation_matrix(-longitude, 'z', unit='deg'), rotation_matrix(latitude, 'y', unit='deg')
    ).T
    # Reorder the axes so that X,Y,Z = E,N,U
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

    frame_dict = {}

    for spec in frame_specs:
        k, v = map(str.strip, spec.split('='))
        frame_dict[k] = v

    latlon = ["{:.4f}".format(l) for l in [latitude, longitude]]

    return station_name, idnum, frame_dict, latlon


def earth_pos_mcmf(obstime):
    """
    Get the position of the Earth in the MCMF frame.

    Used for tests.
    """
    solar_system_ephemeris.set('jpl')
    spkurls = [url for url in get_cached_urls() if 'spk' in url]
    for url in spkurls:
        # Roundabout way to get the path of the cached spk file.
        fpath = download_file(url, cache=True, show_progress=False)
        spice.furnsh(fpath)
        print(url, fpath)
    et = (obstime - Time("J2000")).sec
    earthpos, ltt = spice.spkpos('earth', et, 'MOON_ME', 'None', 'moon')
    earthpos = unit.Quantity(earthpos, 'km')
    return MCMF(*earthpos, obstime=obstime)


def cleanup():
    # TODO Clear the kernel pool
    return 0


KERNEL_PATHS = download_kernels()

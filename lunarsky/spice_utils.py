import numpy as np
import os
import tempfile
from astropy.utils.data import download_files_in_parallel
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time
import astropy.units as unit

import spiceypy as spice

from .mcmf import MCMF
from .data import DATA_PATH

_J2000 = Time("J2000")

TEMPORARY_KERNEL_DIR = tempfile.TemporaryDirectory()


def check_is_loaded(search):
    """
    Search the kernel pool variable names for a given string.
    """
    try:
        spice.gnpool(search, 0, 100)
    except (spice.support_types.SpiceyError):
        return False
    return True


def list_kernels():
    """
    List loaded kernels.

    Returns
    -------
    list of str
        Kernel names (file paths)
    list of str
        Corresponding kernel types
    """
    knames, ktypes = [], []
    for typ in ["spk", "fk", "tk", "pck", "lsk"]:
        for ii in range(spice.ktotal(typ)):
            dat = spice.kdata(ii, typ)
            knames.append(dat[0])
            ktypes.append(dat[1])
    return knames, ktypes


def furnish_kernels():
    kernel_names = [
        "pck/moon_pa_de421_1900-2050.bpc",
        "fk/satellites/moon_080317.tf",
        "fk/satellites/moon_assoc_me.tf",
    ]

    kernel_paths = [os.path.join(DATA_PATH, kn) for kn in kernel_names]
    for kp in kernel_paths:
        spice.furnsh(kp)

    # LSK and DE430 Kernels
    knames = ["lsk/naif0012.tls", "spk/planets/de430.bsp"]
    _naif_kernel_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
    kurls = [_naif_kernel_url + kname for kname in knames]
    paths = download_files_in_parallel(kurls, cache=True, show_progress=False)
    for kp in paths:
        kernel_paths.append(kp)
        spice.furnsh(kp)

    return kernel_paths


def lunar_surface_ephem(latitude, longitude, station_num=98):
    """
    Make an SPK for the point on the lunar surface

    Creates a temporary file and furnishes from that.

    Parameters
    ----------
    latitude: float
        Mean-Earth frame selenodetic latitude in degrees.
    longitude: float
        Mean-Earth frame selenodetic longitude in degrees.
    station_num: int
        Station number

    Returns
    -------
    int:
        Ephemeris ID number
    """
    point_id = 301000 + station_num

    lat = np.radians(latitude)
    lon = np.radians(longitude)
    lunar_radius = 1737.1  # km
    ets = np.array([spice.str2et("1950-01-01"), spice.str2et("2150-01-01")])
    pos_mcmf = spice.latrec(lunar_radius, lon, lat)  # TODO Use MoonLocation instead?

    states = np.zeros((len(ets), 6))
    states[:, :3] = np.repeat(pos_mcmf[None, :], len(ets), axis=0)

    center = 301
    frame = "MOON_ME"
    degree = 1

    fname = os.path.join(TEMPORARY_KERNEL_DIR.name, f"lunar_points.bsp")
    if os.path.exists(fname):
        spice.unload(fname)
        handle = spice.spkopa(fname)
    else:
        handle = spice.spkopn(fname, "SPK_FILE", 0)
    spice.spkw09(
        handle,
        point_id,
        center,
        frame,
        ets[0],
        ets[-1],
        "0",
        degree,
        len(ets),
        states.tolist(),
        ets.tolist(),
    )
    spice.spkcls(handle)
    spice.furnsh(fname)

    return point_id


def topo_frame_def(latitude, longitude, station_num=98, moon=True):
    """
    Make a list of strings defining a topocentric frame. This can then be loaded
    with spiceypy.lmpool.
    """
    if moon:
        idnum = 1301000
        station_name = f"LUNAR-TOPO-{station_num}"
        relative = "MOON_ME"
    else:
        # Used in tests only
        idnum = 1399000
        station_name = f"EARTH-TOPO-{station_num}"
        relative = "ITRF93"

    # The DSS stations are built into SPICE, and they number up to 66.
    # We will call this station number 98.
    idnum += station_num
    fm_center_id = idnum - 1000000

    ecef_to_enu = np.matmul(
        rotation_matrix(-longitude, "z", unit="deg"),
        rotation_matrix(latitude, "y", unit="deg"),
    ).T
    # Reorder the axes so that X,Y,Z = E,N,U
    ecef_to_enu = ecef_to_enu[[2, 1, 0]]

    mat = " ".join(map("{:.7f}".format, ecef_to_enu.flatten()))

    fmt_strs = [
        "FRAME_{1}                     = {0}",
        "FRAME_{0}_NAME                = '{1}'",
        "FRAME_{0}_CLASS               = 4",
        "FRAME_{0}_CLASS_ID            = {0}",
        "FRAME_{0}_CENTER              = {2}",
        "OBJECT_{2}_FRAME              = '{1}'",
        "TKFRAME_{0}_RELATIVE          = '{3}'",
        "TKFRAME_{0}_SPEC              = 'MATRIX'",
        "TKFRAME_{0}_MATRIX            = {4}",
    ]

    frame_specs = [
        s.format(idnum, station_name, fm_center_id, relative, mat) for s in fmt_strs
    ]

    frame_dict = {}

    for spec in frame_specs:
        k, v = map(str.strip, spec.split("="))
        frame_dict[k] = v

    latlon = ["{:.4f}".format(l) for l in [latitude, longitude]]

    return station_name, idnum, frame_dict, latlon


def remove_topo(station_num):
    """Remove a lunar station, by number, from variable pool."""

    idnum = 1301000 + station_num
    fm_center_id = idnum - 1000000
    station_name = f"LUNAR-TOPO-{station_num}"

    fmt_vars = [
        "FRAME_{1}",
        "FRAME_{0}_NAME",
        "FRAME_{0}_CLASS",
        "FRAME_{0}_CLASS_ID",
        "FRAME_{0}_CENTER",
        "OBJECT_{2}_FRAME",
        "TKFRAME_{0}_RELATIVE",
        "TKFRAME_{0}_SPEC",
        "TKFRAME_{0}_MATRIX",
    ]

    frame_vars = [s.format(idnum, station_name, fm_center_id) for s in fmt_vars]

    # Handle a glitch in spiceypy for older versions of numpy
    if np.str_ is None:
        return

    for var in frame_vars:
        spice.dvpool(var)

    # Ideally, one would also remove the ephemeris data for this station from
    # the lunar_points.bsp file. This doesn't seem to be possible. However,
    # the ephemeris _should_ be overwritten if the station_id is reused.


def earth_pos_mcmf(obstimes):
    """
    Get the position of the Earth in the MCMF frame.

    Using SPICE.

    Used for tests.
    """
    ets = (obstimes - Time("J2000")).sec
    earthpos = np.stack(
        [spice.spkpos("399", et, "MOON_ME", "None", "301")[0] for et in ets]
    )
    earthpos = unit.Quantity(earthpos.T, "km")
    return MCMF(*earthpos, obstime=obstimes)


KERNEL_PATHS = furnish_kernels()

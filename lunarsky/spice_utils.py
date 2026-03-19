import numpy as np
import os
from astropy.utils.data import download_files_in_parallel
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time
import astropy.units as unit

from .mcmf import MCMF
from .data import DATA_PATH

_J2000 = Time("J2000")

# ---------------------
# Binary PCK reader
# ---------------------

_PCK_DATA = None


def _read_bpc(filepath):
    """
    Read a DAF-format binary PCK file (Type 2 Chebyshev).

    Returns a dict with segment metadata and coefficient arrays.
    """
    from jplephem.daf import DAF

    daf = DAF(open(filepath, "rb"))

    segments = list(daf.summaries())
    if len(segments) != 1:
        raise ValueError(f"Expected 1 segment in binary PCK, got {len(segments)}")

    name, values = segments[0]
    data_type = int(values[4])
    start_addr = int(values[5])
    end_addr = int(values[6])

    if data_type != 2:
        raise ValueError(f"Unsupported binary PCK data type {data_type}")

    arr = daf.read_array(start_addr, end_addr)

    init_epoch = arr[-4]
    interval = arr[-3]
    rsize = int(arr[-2])
    n_records = int(arr[-1])
    n_coeffs = (rsize - 2) // 3

    data = arr[: n_records * rsize].reshape(n_records, rsize)

    return {
        "init_epoch": init_epoch,
        "interval": interval,
        "n_coeffs": n_coeffs,
        "n_records": n_records,
        "data": data,
    }


def _ensure_pck():
    global _PCK_DATA
    if _PCK_DATA is None:
        _PCK_DATA = _read_bpc(
            os.path.join(DATA_PATH, "pck", "moon_pa_de421_1900-2050.bpc")
        )
    return _PCK_DATA


def _cheby_eval(coeffs, tau):
    """Evaluate Chebyshev polynomial at tau in [-1, 1]. coeffs shape: (..., n_coeffs)."""
    n = coeffs.shape[-1]
    if n == 0:
        return np.zeros(coeffs.shape[:-1])
    T_prev = np.ones(coeffs.shape[:-1])
    if n == 1:
        return coeffs[..., 0] * T_prev
    T_curr = tau
    result = coeffs[..., 0] * T_prev + coeffs[..., 1] * T_curr
    for i in range(2, n):
        T_prev, T_curr = T_curr, 2 * tau * T_curr - T_prev
        result = result + coeffs[..., i] * T_curr
    return result


def _R1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _R2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# PA to ME constant rotation from moon_080317.tf (FRAME_31007 = MOON_ME_DE421).
# These angles are specific to DE421; they will differ for other ephemerides.
# TKFRAME_31007_ANGLES = (67.92, 78.56, 0.30) arcseconds, AXES = (3, 2, 1)
# SPICE convention: R_PA_to_ME = R1(-a3) @ R2(-a2) @ R3(-a1)
_a1 = np.radians(67.92 / 3600)
_a2 = np.radians(78.56 / 3600)
_a3 = np.radians(0.30 / 3600)
_PA_TO_ME = _R1(-_a3) @ _R2(-_a2) @ _R3(-_a1)


def j2000_to_moon_me(ets):
    """
    Compute J2000 -> MOON_ME rotation matrices.

    Parameters
    ----------
    ets : float or array_like
        TDB seconds past J2000.

    Returns
    -------
    mats : ndarray, shape (..., 3, 3)
    """
    pck = _ensure_pck()
    ets = np.atleast_1d(np.asarray(ets, dtype=float))

    record_idx = ((ets - pck["init_epoch"]) / pck["interval"]).astype(int)
    record_idx = np.clip(record_idx, 0, pck["n_records"] - 1)

    records = pck["data"][record_idx]
    mid = records[:, 0]
    half = records[:, 1]
    nc = pck["n_coeffs"]

    tau = (ets - mid) / half

    ra_coeffs = records[:, 2 : 2 + nc]
    dec_coeffs = records[:, 2 + nc : 2 + 2 * nc]
    w_coeffs = records[:, 2 + 2 * nc : 2 + 3 * nc]

    ra = _cheby_eval(ra_coeffs, tau)
    dec = _cheby_eval(dec_coeffs, tau)
    w = _cheby_eval(w_coeffs, tau)

    # J2000 -> MOON_PA: R3(W) @ R1(Dec) @ R3(RA)
    # Then apply PA -> ME constant rotation
    mats = np.zeros((len(ets), 3, 3))
    for i in range(len(ets)):
        j2000_to_pa = _R3(w[i]) @ _R1(dec[i]) @ _R3(ra[i])
        mats[i] = _PA_TO_ME @ j2000_to_pa

    return mats


def moon_me_to_j2000(ets):
    """Transpose of j2000_to_moon_me."""
    return np.swapaxes(j2000_to_moon_me(ets), -2, -1)


# ---------------------
# Ephemeris (jplephem)
# ---------------------

_SPK = None


def _ensure_spk():
    global _SPK
    if _SPK is None:
        from jplephem.spk import SPK

        spk_name = "spk/planets/de430.bsp"
        _naif_kernel_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        kurl = [_naif_kernel_url + spk_name]
        paths = download_files_in_parallel(kurl, cache=True, show_progress=False)
        _SPK = SPK.open(paths[0])
    return _SPK


def _et_to_jd(ets):
    """Convert ET (TDB seconds past J2000) to Julian date."""
    return np.asarray(ets) / 86400.0 + 2451545.0


def _pos_ssb_j2000(body_id, ets):
    """
    Get position of a solar system body relative to SSB in J2000.

    Parameters
    ----------
    body_id : int
        0=SSB, 301=Moon, 399=Earth
    ets : ndarray
        ET values

    Returns
    -------
    pos : ndarray, shape (N, 3)
        Position in km.
    """
    if body_id == 0:
        return np.zeros((len(ets), 3))

    spk = _ensure_spk()
    jd = _et_to_jd(ets)

    if body_id == 301:
        emb = np.asarray(spk[0, 3].compute(jd)).T           # spk keys are tuples (center, target)
        moon_emb = np.asarray(spk[3, 301].compute(jd)).T
        return emb + moon_emb
    elif body_id == 399:
        emb = np.asarray(spk[0, 3].compute(jd)).T
        earth_emb = np.asarray(spk[3, 399].compute(jd)).T
        return emb + earth_emb
    else:
        raise ValueError(f"Unsupported body ID {body_id}")


def body_position(target_id, ets, frame, observer_id):
    """
    Get position of target relative to observer in given frame.

    Parameters
    ----------
    target_id : int
        NAIF body ID of target (0=SSB, 301=Moon, 399=Earth)
    ets : array_like
        ET values (TDB seconds past J2000)
    frame : str
        "J2000" or "MOON_ME"
    observer_id : int
        NAIF body ID of observer

    Returns
    -------
    pos : ndarray, shape (N, 3)
        Position in km.
    """
    ets = np.atleast_1d(np.asarray(ets, dtype=float))

    pos_j2000 = _pos_ssb_j2000(target_id, ets) - _pos_ssb_j2000(observer_id, ets)

    if frame == "J2000":
        return pos_j2000
    elif frame == "MOON_ME":
        mats = j2000_to_moon_me(ets)
        return np.einsum("nij,nj->ni", mats, pos_j2000)
    else:
        raise ValueError(f"Unsupported frame {frame}")


def station_pos_ssb_j2000(pos_me_km, ets):
    """
    Get position of a lunar surface station relative to SSB in J2000.

    Parameters
    ----------
    pos_me_km : ndarray, shape (3,)
        Station position in MOON_ME frame, in km.
    ets : ndarray
        ET values.

    Returns
    -------
    pos : ndarray, shape (N, 3)
        Position in km.
    """
    ets = np.atleast_1d(np.asarray(ets, dtype=float))
    moon_ssb = _pos_ssb_j2000(301, ets)
    me_to_j2000 = moon_me_to_j2000(ets)
    station_j2000 = np.einsum("nij,j->ni", me_to_j2000, pos_me_km)
    return moon_ssb + station_j2000


def topo_rotation_matrix(lat_deg, lon_deg):
    """
    Compute the MOON_ME -> topocentric (E/N/U) rotation matrix for a surface location.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Selenodetic latitude and longitude in degrees.

    Returns
    -------
    matrix : ndarray, shape (3, 3)
    """
    ecef_to_enu = np.matmul(
        rotation_matrix(-lon_deg, "z", unit="deg"),
        rotation_matrix(lat_deg, "y", unit="deg"),
    ).T
    ecef_to_enu = ecef_to_enu[[2, 1, 0]]
    return np.asarray(ecef_to_enu)


def earth_pos_mcmf(obstimes):
    """
    Get the position of the Earth in the MCMF frame.

    Used for tests.
    """
    ets = (obstimes.tdb - Time("J2000")).sec
    earthpos = body_position(399, ets, "MOON_ME", 301)
    earthpos = unit.Quantity(earthpos.T, "km")
    return MCMF(*earthpos, obstime=obstimes)

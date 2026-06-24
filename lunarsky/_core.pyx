# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compiled module for lunarsky's coordinate transformations.

Replaces the per-call Python overhead in `j2000_to_moon_me(scalar_et)` and
`topo_rotation_matrix(lat, lon)` — the two functions that pyuvsim/pyradiosky
hit thousands of times per simulation timestep. The vectorized Python path
in spice_utils.py stays the entry point for array inputs; this module is
the scalar-fast-path implementation.

Initialize once via `init(pck_init_epoch, pck_interval, pck_n_coeffs,
pck_n_records, pck_data, pa_to_me)` before calling the scalar routines.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin


# ----- module-level state, populated by init() -----
cdef bint _initialized = False
cdef double _pck_init_epoch = 0.0
cdef double _pck_interval = 0.0
cdef int _pck_n_coeffs = 0
cdef int _pck_n_records = 0
cdef double[:, ::1] _pck_data            # (n_records, rsize) C-contiguous
cdef double _pa_to_me[3][3]


def init(double init_epoch, double interval, int n_coeffs, int n_records,
         cnp.ndarray[double, ndim=2, mode="c"] data,
         cnp.ndarray[double, ndim=2, mode="c"] pa_to_me):
    """Set the PCK Chebyshev parameters and the constant PA->ME rotation.

    Must be called once before invoking the scalar routines. Callers in
    spice_utils.py do this lazily on first use.
    """
    global _initialized, _pck_init_epoch, _pck_interval
    global _pck_n_coeffs, _pck_n_records, _pck_data
    cdef int i, j
    _pck_init_epoch = init_epoch
    _pck_interval = interval
    _pck_n_coeffs = n_coeffs
    _pck_n_records = n_records
    _pck_data = data
    for i in range(3):
        for j in range(3):
            _pa_to_me[i][j] = pa_to_me[i, j]
    _initialized = True


cdef inline double _cheby_eval_scalar(double[::1] coeffs, double tau) noexcept nogil:
    """Evaluate Chebyshev polynomial of the first kind at scalar tau."""
    cdef int n = coeffs.shape[0]
    cdef double T_prev, T_curr, T_next, result
    cdef int i

    if n == 0:
        return 0.0
    if n == 1:
        return coeffs[0]

    T_prev = 1.0
    T_curr = tau
    result = coeffs[0] + coeffs[1] * tau
    for i in range(2, n):
        T_next = 2.0 * tau * T_curr - T_prev
        result += coeffs[i] * T_next
        T_prev = T_curr
        T_curr = T_next
    return result


def j2000_to_moon_me_scalar(double et):
    """Compute the 3x3 J2000 -> MOON_ME rotation matrix for a single ET.

    Parameters
    ----------
    et : float
        TDB seconds past J2000.

    Returns
    -------
    matrix : ndarray, shape (3, 3)
    """
    if not _initialized:
        raise RuntimeError("lunarsky._core: init() must be called first")

    cdef int record_idx = <int>((et - _pck_init_epoch) / _pck_interval)
    if record_idx < 0:
        record_idx = 0
    elif record_idx >= _pck_n_records:
        record_idx = _pck_n_records - 1

    cdef double mid = _pck_data[record_idx, 0]
    cdef double half = _pck_data[record_idx, 1]
    cdef double tau = (et - mid) / half
    cdef int nc = _pck_n_coeffs

    cdef double ra = _cheby_eval_scalar(_pck_data[record_idx, 2:2 + nc], tau)
    cdef double dec = _cheby_eval_scalar(_pck_data[record_idx, 2 + nc:2 + 2 * nc], tau)
    cdef double w = _cheby_eval_scalar(_pck_data[record_idx, 2 + 2 * nc:2 + 3 * nc], tau)

    cdef double cw = cos(w), sw = sin(w)
    cdef double cd = cos(dec), sd = sin(dec)
    cdef double cr = cos(ra), sr = sin(ra)

    # j = R3(w) @ R1(dec) @ R3(ra), expanded symbolically.
    cdef double j[3][3]
    j[0][0] = cw * cr - sw * cd * sr
    j[0][1] = cw * sr + sw * cd * cr
    j[0][2] = sw * sd
    j[1][0] = -sw * cr - cw * cd * sr
    j[1][1] = -sw * sr + cw * cd * cr
    j[1][2] = cw * sd
    j[2][0] = sd * sr
    j[2][1] = -sd * cr
    j[2][2] = cd

    # out = _pa_to_me @ j
    cdef cnp.ndarray[double, ndim=2] out = np.empty((3, 3), dtype=np.float64)
    cdef int i, k
    for i in range(3):
        for k in range(3):
            out[i, k] = (_pa_to_me[i][0] * j[0][k]
                         + _pa_to_me[i][1] * j[1][k]
                         + _pa_to_me[i][2] * j[2][k])
    return out


def topo_rotation_matrix(double lat_deg, double lon_deg):
    """Compute the MOON_ME -> topocentric (E/N/U) rotation matrix.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Selenodetic latitude and longitude in degrees.

    Returns
    -------
    matrix : ndarray, shape (3, 3)
    """
    cdef double DEG2RAD = 0.017453292519943295  # math.pi / 180
    cdef double lat = lat_deg * DEG2RAD
    cdef double lon = lon_deg * DEG2RAD
    cdef double sl = sin(lat), cl = cos(lat)
    cdef double slo = sin(lon), clo = cos(lon)

    # Derived to match the existing astropy rotation_matrix chain:
    #   ((R_z(-lon) @ R_y(lat))^T)[[2, 1, 0]]
    cdef cnp.ndarray[double, ndim=2] out = np.empty((3, 3), dtype=np.float64)
    out[0, 0] = -clo * sl
    out[0, 1] = -slo * sl
    out[0, 2] = cl
    out[1, 0] = -slo
    out[1, 1] = clo
    out[1, 2] = 0.0
    out[2, 0] = clo * cl
    out[2, 1] = slo * cl
    out[2, 2] = sl
    return out

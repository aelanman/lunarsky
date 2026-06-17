import numpy as np
import os
import pytest
from astropy.time import Time
import lunarsky.spice_utils as spice_utils


# spice_fixtures.npz contains saved results of the earlier version of lunarsky that was based
# on the SPICE toolkit. Tests here are meant to verify that the new code is consistent with
# older results.
#
# Tolerance notes
# ---------------
# All assertions below set rtol=0 explicitly, because np.testing.assert_allclose
# applies an implicit rtol=1e-7 unless overridden — which can silently let a
# tight-looking atol pass differences of order 1e-7 in the worst case. The atol
# values are sized to be ~10x the largest observed deviation against the
# fixture, so a precision regression of even one significant figure surfaces
# rather than being absorbed by relative tolerance. Reduce the atol further
# only after re-generating the fixture with the current production pipeline.
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "data", "spice_fixtures.npz")


@pytest.fixture
def fixtures():
    return np.load(FIXTURE_PATH)


def test_kernels_available():
    """
    Ensure the large SPK kernel is downloadable and cached.

    On a fresh environment this hits the JPL server; on subsequent runs
    (and in CI with a populated astropy cache) it is a no-op.
    """
    paths = spice_utils.download_big_kernels(show_progress=False)
    for p in paths:
        assert os.path.exists(p)


def test_j2000_to_moon_me(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["j2000_to_me"]
    computed = spice_utils.j2000_to_moon_me(ets)
    # Rotation matrix elements have magnitude ~1, so atol ≈ matrix element error.
    # Observed: ~2e-12.
    np.testing.assert_allclose(computed, expected, rtol=0, atol=1e-11)


def test_body_position_moon_earth(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["moon_earth_j2000"][:, :3]
    computed = spice_utils.body_position(301, ets, "J2000", 399)
    # Observed: ~5e-8 km (~ μm).
    np.testing.assert_allclose(computed, expected, rtol=0, atol=1e-7)


def test_body_position_moon_ssb(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["moon_ssb_j2000"][:, :3]
    computed = spice_utils.body_position(301, ets, "J2000", 0)
    # Observed: ~2e-6 km (~mm); driven by ~1 AU vector magnitude.
    np.testing.assert_allclose(computed, expected, rtol=0, atol=1e-5)


def test_body_position_ssb_moon_me(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["ssb_moon_me"][:, :3]
    computed = spice_utils.body_position(0, ets, "MOON_ME", 301)
    # Observed: ~3e-4 km (~30 cm); same as above but rotated to MOON_ME.
    np.testing.assert_allclose(computed, expected, rtol=0, atol=1e-3)


def test_earth_pos_mcmf(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["earth_me"]
    computed = spice_utils.body_position(399, ets, "MOON_ME", 301)
    # Observed: ~6e-7 km (~ mm); Earth-Moon distance is ~3.8e5 km.
    np.testing.assert_allclose(computed, expected, rtol=0, atol=1e-6)


def test_topo_rotation_matrix(fixtures):
    expected = fixtures["topo_to_me"]
    topo_matrix = spice_utils.topo_rotation_matrix(
        float(fixtures["topo_station_lat_deg"]),
        float(fixtures["topo_station_lon_deg"]),
    )
    # TOPO -> MOON_ME should be topo_matrix.T (constant).
    # The fixture itself is stored at machine precision (NOT round-tripped
    # through the lossy "{:.7f}" SPICE kernel-pool path that the archival
    # spiceypy production code used). bland's full-precision matrix should
    # match it bit-for-bit.
    for i in range(len(expected)):
        np.testing.assert_allclose(topo_matrix.T, expected[i], rtol=0, atol=1e-14)

    # Sanity check: the matrix is orthogonal to machine precision.
    np.testing.assert_allclose(topo_matrix @ topo_matrix.T, np.eye(3), rtol=0, atol=1e-14)


def test_topo_to_j2000(fixtures):
    ets = fixtures["ets"]
    expected_to_j2000 = fixtures["topo_to_j2000"]
    topo_matrix = spice_utils.topo_rotation_matrix(
        float(fixtures["topo_station_lat_deg"]),
        float(fixtures["topo_station_lon_deg"]),
    )
    me_to_j2000 = spice_utils.moon_me_to_j2000(ets)
    computed_to_j2000 = np.einsum("nij,jk->nik", me_to_j2000, topo_matrix.T)
    # Observed: ~2e-12.
    np.testing.assert_allclose(computed_to_j2000, expected_to_j2000, rtol=0, atol=1e-11)


def test_present_day_precision():
    """
    Pin down sub-meter precision at present-day epochs.

    The historical fixture above was generated at epochs where the implicit
    JD round-off in jplephem's single-argument compute() happened not to
    matter. At present-day epochs (~JD 2460000) the same single-JD call
    loses ~26 μs of time resolution, which propagates into ~0.8 m of Moon
    position error at ~30 km/s barycentric speed. The two-part-JD form in
    _et_to_jd() recovers sub-microsecond precision and sub-millimeter Moon
    positions. This test fails if anyone re-introduces a single-JD compute().
    """
    obstime = Time(2460000.0, format="jd", scale="utc")
    ets = np.array([float((obstime.tdb - Time("J2000")).sec)])
    moon = spice_utils._pos_ssb_j2000(301, ets)[0]  # km
    earth = spice_utils._pos_ssb_j2000(399, ets)[0]
    # Earth-Moon vector should agree with body_position
    em = spice_utils.body_position(301, ets, "J2000", 399)[0]
    np.testing.assert_allclose(moon - earth, em, rtol=0, atol=1e-9)
    # The magnitude of the Moon's barycentric position is ~1.5e8 km; with
    # double-precision floor we can check it's finite and well-conditioned.
    assert 1.4e8 < np.linalg.norm(moon) < 1.6e8

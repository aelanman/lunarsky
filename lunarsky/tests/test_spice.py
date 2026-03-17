import numpy as np
import os
import pytest
import lunarsky.spice_utils as spice_utils


# spice_fixtures.npz contains saved results of the earlier version of lunarsky that was based on the SPICE toolkit
# Tests here are meant to verify that the new code is consistent with older results.
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "data", "spice_fixtures.npz")


@pytest.fixture
def fixtures():
    return np.load(FIXTURE_PATH)


def test_j2000_to_moon_me(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["j2000_to_me"]
    computed = spice_utils.j2000_to_moon_me(ets)
    np.testing.assert_allclose(computed, expected, atol=1e-10)


def test_body_position_moon_earth(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["moon_earth_j2000"][:, :3]
    computed = spice_utils.body_position(301, ets, "J2000", 399)
    np.testing.assert_allclose(computed, expected, atol=1e-6)


def test_body_position_moon_ssb(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["moon_ssb_j2000"][:, :3]
    computed = spice_utils.body_position(301, ets, "J2000", 0)
    np.testing.assert_allclose(computed, expected, atol=1e-6)


def test_body_position_ssb_moon_me(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["ssb_moon_me"][:, :3]
    computed = spice_utils.body_position(0, ets, "MOON_ME", 301)
    np.testing.assert_allclose(computed, expected, atol=1e-4)


def test_earth_pos_mcmf(fixtures):
    ets = fixtures["ets"]
    expected = fixtures["earth_me"]
    computed = spice_utils.body_position(399, ets, "MOON_ME", 301)
    np.testing.assert_allclose(computed, expected, atol=1e-4)


def test_topo_rotation_matrix(fixtures):
    expected = fixtures["topo_to_me"]
    topo_matrix = spice_utils.topo_rotation_matrix(
        float(fixtures["topo_station_lat_deg"]),
        float(fixtures["topo_station_lon_deg"]),
    )
    # TOPO -> MOON_ME should be topo_matrix.T (constant)
    for i in range(len(expected)):
        np.testing.assert_allclose(topo_matrix.T, expected[i], atol=1e-7)

    # Verify full precision matrix is orthogonal
    np.testing.assert_allclose(topo_matrix @ topo_matrix.T, np.eye(3), atol=1e-14)


def test_topo_to_j2000(fixtures):
    ets = fixtures["ets"]
    expected_to_j2000 = fixtures["topo_to_j2000"]
    topo_matrix = spice_utils.topo_rotation_matrix(
        float(fixtures["topo_station_lat_deg"]),
        float(fixtures["topo_station_lon_deg"]),
    )
    me_to_j2000 = spice_utils.moon_me_to_j2000(ets)
    computed_to_j2000 = np.einsum("nij,jk->nik", me_to_j2000, topo_matrix.T)
    np.testing.assert_allclose(computed_to_j2000, expected_to_j2000, atol=1e-10)

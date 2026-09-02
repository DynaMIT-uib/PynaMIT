"""Reference checks against Kaiju's Fortran Geopack implementation."""

import datetime as dt

import numpy as np
import pytest

from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm

# These matrices were generated from Kaiju's RECALC and GEO2SM
# routines at JHUAPL/kaiju commit 9e19bfc by transforming the three
# geographic Cartesian basis vectors. Stored values let the test run
# without a Fortran compiler or a separate Kaiju checkout.
KAIJU_GEO_TO_SM_REFERENCES = (
    (
        dt.datetime(2001, 5, 12, 21, 45),
        np.array(
            [
                [-0.8736076578306136, -0.4854391124177372, -0.03403716079212693],
                [0.4832943625663665, -0.8573163025553616, -0.1773000746880087],
                [0.05688777804835682, -0.1713406709099841, 0.9835680734961040],
            ]
        ),
    ),
    (
        dt.datetime(2011, 10, 24, 18, 0, 10),
        np.array(
            [
                [-0.06578210795995332, -0.9849321525364415, -0.1599417680666216],
                [0.9964782817230269, -0.05649060395080135, -0.06196648868216956],
                [0.05199758000781188, -0.1634547844671596, 0.9851795699810900],
            ]
        ),
    ),
    (
        dt.datetime(2020, 3, 20, 12),
        np.array(
            [
                [0.9983060496712131, 0.03917558219661672, -0.04301517115406623],
                [-0.03194277709618358, 0.9869930655116385, 0.1575574422974741],
                [0.04862808017227304, -0.1559165237923630, 0.9865725251735245],
            ]
        ),
    ),
)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    KAIJU_GEO_TO_SM_REFERENCES,
    ids=("2001-05-12", "2011-10-24", "2020-03-20"),
)
def test_kaiju_geopack_sm_matches_fortran_reference(epoch, expected):
    """Match Kaiju at distinct seasons and years."""
    observed = kaiju_geopack_sm(epoch).geo_to_sm_matrix

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5e-12)

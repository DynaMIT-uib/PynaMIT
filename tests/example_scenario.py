"""The empirical event shared by PynaMIT's regression tests.

These are inputs to Hardy, AMPS, and HWM, not defaults of the PynaMIT
model. Keeping them here makes the physical test case explicit without
repeating it in every numerical regression test.
"""

import datetime

from pynamit.workflows.example import run_example as _run_example
from pynamit.workflows.example_inputs import prepare_example_inputs as _prepare_example_inputs

EVENT_TIME = datetime.datetime(2001, 5, 12, 21, 45)
KP = 5
STARLIGHT_CONDUCTANCE_S = 1.0
SOLAR_WIND_SPEED_KM_S = 300.0
IMF_BY_NT = 0.0
IMF_BZ_NT = -4.0
DIPOLE_TILT_DEG = 20.0
F107_SFU = 100.0
AMPS_MIN_LATITUDE_DEG = 50.0
HWM_AP = (-1, 35)

EMPIRICAL_INPUTS = {
    "event_time": EVENT_TIME,
    "kp": KP,
    "starlight_conductance_S": STARLIGHT_CONDUCTANCE_S,
    "solar_wind_speed_km_s": SOLAR_WIND_SPEED_KM_S,
    "imf_By_nT": IMF_BY_NT,
    "imf_Bz_nT": IMF_BZ_NT,
    "dipole_tilt_deg": DIPOLE_TILT_DEG,
    "f107_sfu": F107_SFU,
    "amps_min_latitude_deg": AMPS_MIN_LATITUDE_DEG,
    "hwm_ap": HWM_AP,
}


def prepare_example_inputs(*args, **kwargs):
    """Prepare the shared 12 May 2001 test inputs."""
    return _prepare_example_inputs(*args, **EMPIRICAL_INPUTS, **kwargs)


def run_example(*args, **kwargs):
    """Run a simulation using the shared 12 May 2001 test inputs."""
    return _run_example(*args, **EMPIRICAL_INPUTS, **kwargs)

"""Main entry point for running PynaMIT's bundled demonstration.

The empirical-model inputs are stated here as ordinary script settings.
"""

import datetime

from .workflows.example import run_example

if __name__ == "__main__":
    run_example(
        event_time=datetime.datetime(2001, 5, 12, 21, 45),
        kp=5,
        starlight_conductance_S=1.0,
        solar_wind_speed_km_s=300.0,
        imf_By_nT=0.0,
        imf_Bz_nT=-4.0,
        dipole_tilt_deg=20.0,
        f107_sfu=100.0,
        amps_min_latitude_deg=50.0,
        hwm_ap=(-1, 35),
    )

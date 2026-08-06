import datetime as dt
import hashlib
import importlib.metadata
import tempfile
from pathlib import Path

import apexpy
import numpy as np
import pyamps
from dipole import Dipole
from kompe.constants import EARTH_RADIUS_M
from lompe import conductance

from pynamit.geomagnetism import decimal_year
from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.prepared_inputs import _DEFAULT_INPUT_TIME


def digest(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:20]


def show(name, values):
    array = np.asarray(values)
    indices = [1, 23, 2231, 2555, 2652, 2712]
    indices = [i for i in indices if i < array.size]
    flat = array.reshape(-1)

    print(
        f"{name:16} "
        f"sha={digest(array)} "
        f"dtype={array.dtype} "
        f"values={[float(flat[i]) for i in indices]}"
    )


for package in ("numpy", "scipy", "apexpy", "dipole", "lompe", "pyamps"):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, "not installed")

event_time = _DEFAULT_INPUT_TIME
if event_time.tzinfo is not None:
    event_time = event_time.astimezone(
        dt.timezone.utc
    ).replace(tzinfo=None)

with tempfile.TemporaryDirectory() as directory:
    simulation = Simulation(
        run_directory=Path(directory),
        Nmax=12,
        Mmax=12,
        Ncs=22,
        RI=EARTH_RADIUS_M + 110e3,
        main_field_kind="dipole",
        main_field_epoch=2020.0,
        t0=event_time.isoformat(sep=" "),
        enable_pfac_coupling=False,
        backend="numpy",
    )

    model_lat = np.asarray(simulation.geometry.model_grid.lat)
    model_lon = np.asarray(simulation.geometry.model_grid.lon)

    geo_lat, geo_lon = (
        simulation.geometry.main_field.model_to_geo_coordinates(
            model_lat,
            model_lon,
            event_time=event_time,
        )
    )

apex = apexpy.Apex(date=event_time, refh=110.0)
apex_lat, apex_lon = apex.geo2apex(
    geo_lat,
    geo_lon,
    110.0,
)

dipole_mlt = Dipole(event_time.year).mlon2mlt(
    apex_lon,
    event_time,
)

amps_mlt = pyamps.mlon_to_mlt(
    apex_lon,
    event_time,
    decimal_year(event_time),
)

hall, pedersen = conductance.hardy_EUV(
    geo_lon,
    geo_lat,
    5,
    event_time,
    starlight=1.0,
    dipole=False,
)

amps = pyamps.AMPS(
    300.0,
    0.0,
    -4.0,
    20.0,
    100.0,
    minlat=50.0,
)

jr = np.asarray(
    amps.get_upward_current(
        mlat=apex_lat,
        mlt=amps_mlt,
    )
) * 1e-6

jr = np.array(jr, copy=True)
jr[np.abs(apex_lat) < 50.0] = 0.0

print("event_time", event_time.isoformat())
print("apex_datafile", getattr(apex, "datafile", None))
print()

show("model_lat", model_lat)
show("model_lon", model_lon)
show("geo_lat", geo_lat)
show("geo_lon", geo_lon)
show("apex_lat", apex_lat)
show("apex_lon", apex_lon)
show("dipole_mlt", dipole_mlt)
show("amps_mlt", amps_mlt)
show("hall", hall)
show("pedersen", pedersen)
show("jr", jr)




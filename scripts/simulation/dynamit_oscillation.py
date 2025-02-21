"""Oscillating input simulation."""

import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014  # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy
import matplotlib.pyplot as plt

RE = 6371.2e3
RI = RE + 110e3
LATITUDE_BOUNDARY = 40

STEADY_STATE_INITIALIZATION = True
RELAXATION_TIME = 0.0
TAPERING_TIME = 200.0
FINAL_TIME = 300.0
JR_SAMPLING_DT = 0.5
JR_PERIOD = 20.0

PLOT_SCALING = False

WIND_FACTOR = 1  # scale wind by this factor
FLOAT_ERROR_MARGIN = 1e-6

Nmax, Mmax, Ncs = 50, 50, 50
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2  # int(80 / Nmax)])) ** 2
# print(len(rk))

for JR_PERIOD in [50, 25, 10, 5, 1]:
    JR_SAMPLING_DT = JR_PERIOD / 50

    dataset_filename_prefix = "oscillations/" + str(int(JR_PERIOD)).zfill(2) + "s"

    date = datetime.datetime(2001, 5, 12, 17, 0)
    d = dipole.Dipole(date.year)
    noon_longitude = d.mlt2mlon(12, date)  # noon longitude
    noon_mlon = d.mlt2mlon(12, date)  # noon longitude

    ## SET UP SIMULATION OBJECT
    dynamics = pynamit.Dynamics(
        dataset_filename_prefix=dataset_filename_prefix,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=RI,
        mainfield_kind="igrf",
        FAC_integration_steps=rk,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=LATITUDE_BOUNDARY,
        ih_constraint_scaling=1e-5,
        t0=str(date),
    )

    print("Setting wind", flush=True)
    ## WIND INPUT
    hwm14Obj = pyhwm2014.HWM142D(
        alt=110.0,
        ap=[35, 35],
        glatlim=[-88.5, 88.5],
        glatstp=1.5,
        glonlim=[-180.0, 180.0],
        glonstp=3.0,
        option=6,
        verbose=False,
        ut=date.hour + date.minute / 60,
        day=date.timetuple().tm_yday,
    )

    u_theta, u_phi = (
        -hwm14Obj.Vwind.flatten() * WIND_FACTOR,
        hwm14Obj.Uwind.flatten() * WIND_FACTOR,
    )
    u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")
    # u_lat, u_lon, u_phi, u_theta = (
    #     np.load("ulat.npy"),
    #     np.load("ulon.npy"),
    #     np.load("uphi.npy"),
    #     np.load("utheta.npy"),
    # )
    # u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing="ij")
    # dynamics.set_u(
    #     u_theta=u_theta,
    #     u_phi=u_phi,
    #     lat=u_lat,
    #     lon=u_lon,
    #     weights=np.tile(
    #         np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)
    #     ),
    # )

    print("Setting conductance", flush=True)
    ## CONDUCTANCE INPUT
    conductance_lat = dynamics.state_grid.lat
    conductance_lon = dynamics.state_grid.lon

    sza = conductance.sunlight.sza(conductance_lat, conductance_lon, date, degrees=True)
    hall_EUV, pedersen_EUV = conductance.EUV_conductance(sza)
    hall_EUV, pedersen_EUV = (
        np.sqrt(hall_EUV**2 + 1),
        np.sqrt(pedersen_EUV**2 + 1),
    )  # add starlight
    dynamics.set_conductance(
        hall_EUV, pedersen_EUV, lat=conductance_lat, lon=conductance_lon
    )

    ## jr STATIC INPUT
    jr_lat = dynamics.state_grid.lat
    jr_lon = dynamics.state_grid.lon
    apx = apexpy.Apex(refh=(RI - RE) * 1e-3, date=2020)
    mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
    mlt = d.mlon2mlt(mlon, date)
    _, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI - RE) * 1e-3)  # fix this
    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
    jr = a.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
    jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr

    if STEADY_STATE_INITIALIZATION:
        dynamics.set_jr(jr=jr, lat=jr_lat, lon=jr_lon)

        dynamics.impose_steady_state()

    # Create array that will store all jr values
    time_values = np.arange(
        0,
        FINAL_TIME + JR_SAMPLING_DT - FLOAT_ERROR_MARGIN,
        JR_SAMPLING_DT,
        dtype=np.float64,
    )
    scaled_jr_values = np.zeros((time_values.size, jr.size), dtype=np.float64)

    envelope_factors = np.empty(time_values.size, dtype=np.float64)
    sine_wave_factors = np.empty(time_values.size, dtype=np.float64)

    print("Interpolating jr", flush=True)
    for time_index in range(time_values.size):
        if time_values[time_index] < RELAXATION_TIME - FLOAT_ERROR_MARGIN:
            envelope_factor = 0.0
        elif (
            time_values[time_index]
            < RELAXATION_TIME + TAPERING_TIME - FLOAT_ERROR_MARGIN
        ):
            envelope_factor = (
                np.sin(
                    0.5
                    * np.pi
                    * (time_values[time_index] - RELAXATION_TIME)
                    / TAPERING_TIME
                )
                ** 2
            )
        else:
            envelope_factor = 1.0

        sine_wave_factor = np.sin(
            2.0 * np.pi * (time_values[time_index] - RELAXATION_TIME) / JR_PERIOD
        )

        scaled_jr = jr * (1.0 + envelope_factor * 0.5 * sine_wave_factor)

        scaled_jr_values[time_index] = scaled_jr

        envelope_factors[time_index] = envelope_factor
        sine_wave_factors[time_index] = sine_wave_factor

    if PLOT_SCALING:
        fig, axs = plt.subplots(3, 1, tight_layout=True)

        axs[0].plot(time_values, envelope_factors)
        axs[1].plot(time_values, sine_wave_factors)
        axs[2].plot(time_values, envelope_factors * sine_wave_factors)

        axs[0].set_title("Envelope")
        axs[1].set_title("Oscillation")
        axs[2].set_title("Product")

        plt.savefig(dataset_filename_prefix + ".png")
        plt.close()

    print("Setting jr", flush=True)
    dynamics.set_jr(jr=scaled_jr_values, lat=jr_lat, lon=jr_lon, time=time_values)

    print("Starting simulation", flush=True)
    dynamics.evolve_to_time(FINAL_TIME, interpolation=True)

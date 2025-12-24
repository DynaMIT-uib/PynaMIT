"""Script to check if the interhemispheric connection works."""

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance
import pyhwm2014  # https://github.com/rilma/pyHWM14
import cartopy.crs as ccrs
import os

PLOT_WIND = False  # True to make a plot of the wind field
SIMULATE = True

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

filename_prefix = "ihc_test"

Nmax, Mmax, Ncs = 25, 15, 50  # Model resolution
print(
    "we need a check that the poloidal field calculation is high enough resoultion compared to SH "
)


rk = RI / np.cos(np.deg2rad(np.linspace(0, 70, int(360 / (Nmax + 0.5)) + 1))) ** 2
# rk = np.hstack(
#     rk,
#     np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)[5:] / RE
# )


# Define parameters for empirical models.
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date)  # noon longitude


# Define plotting parameters.
fig_directory = "figs/"
Blevels = np.linspace(-5, 5, 22) * 1e-9  # color levels for Br
levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100)  # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

# Set up simulation object.
dynamics = pynamit.Dynamics(
    filename_prefix=filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    mainfield_kind="dipole",
    FAC_integration_steps=rk,
    ignore_PFAC=False,
    connect_hemispheres=True,
    latitude_boundary=latitude_boundary,
)

# Get and set conductance input.
conductance_lat = dynamics.state.geometry.grid.lat
conductance_lon = dynamics.state.geometry.grid.lon
hall, pedersen = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=True
)
dynamics.set_conductance(hall, pedersen, lat=conductance_lat, lon=conductance_lon)

# Get and set jr input.
jr_lat = dynamics.state.geometry.grid.lat
jr_lon = dynamics.state.geometry.grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=jr_lat, mlt=d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)

# Get and set wind input.
hwm14Obj = pyhwm2014.HWM142D(
    alt=110.0,
    ap=[35, 35],
    glatlim=[-89.0, 88.0],
    glatstp=3.0,
    glonlim=[-180.0, 180.0],
    glonstp=8.0,
    option=6,
    verbose=False,
    ut=date.hour,
    day=date.timetuple().tm_yday,
)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")
dynamics.set_u(
    u_theta=u_theta,
    u_phi=u_phi,
    lat=u_lat,
    lon=u_lon,
    sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
)

dynamics.update_conductance()
dynamics.update_u()
dynamics.update_jr()
dynamics.state.update_m_imp()
dynamics.state.update_E()

# Set up plotting grid and evaluators.
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat=lat, lon=lon)
# Removed BasisEvaluator. Logic will use basis and grid directly.
plt_state_basis = dynamics.state_basis

G_Br = plt_state_basis.get_scaled_matrix(plt_grid, dynamics.state.m_ind_to_Br)
Br = G_Br.dot(dynamics.state.geometry.T_to_Ve.dot(dynamics.state.m_imp.coeffs))


if PLOT_WIND:
    u_grid = pynamit.Grid(lat=u_lat, lon=u_lon)
    u_theta_field = pynamit.Field.from_grid_values_expansion(
        dynamics.state_basis,
        grid_values=u_theta,
        grid=u_grid,
        vector_type="scalar",
    )
    u_phi_field = pynamit.Field.from_grid_values_expansion(
        dynamics.state_basis,
        grid_values=u_phi,
        grid=u_grid,
        vector_type="scalar",
    )

    u_theta_int = u_theta_field.to_grid_values(dynamics.state.geometry.grid)
    u_phi_int = u_phi_field.to_grid_values(dynamics.state.geometry.grid)

    fig, ax = plt.subplots(
        figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=lon0)}
    )
    ax.coastlines()
    Q = ax.quiver(
        u_lon.flatten(),
        u_lat.flatten(),
        u_phi.flatten(),
        -u_theta.flatten(),
        color="blue",
        transform=ccrs.PlateCarree(),
    )
    ax.quiver(
        dynamics.state.geometry.grid.lon,
        dynamics.state.geometry.grid.lat,
        u_phi_int,
        -u_theta_int,
        color="red",
        scale=Q.scale,
        transform=ccrs.PlateCarree(),
    )


if SIMULATE:
    dt = 5e-4
    totalsteps = 200001
    # Define plotting parameters.
    plotsteps = 500
    fig_directory = "figs/"
    Blevels = np.linspace(-50, 50, 22) * 1e-9  # color levels for Br
    levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
    c_levels = np.linspace(0, 20, 100)  # color levels for conductance
    Wlevels = np.r_[-512.5:512.5:5]
    Philevels = np.r_[-212.5:212.5:2.5]

    # Run the simulation.
    coeffs = []
    count = 0
    filecount = 1
    time = 0
    while True:
        dynamics.state.evolve_Br(dt)
        time = time + dt
        coeffs.append(dynamics.state.m_ind.coeffs)
        count += 1
        # print(
        #     count,
        #     time,
        #     (
        #         dynamics.state.m_ind.coeffs
        #         * dynamics.state.m_ind_to_Br)[:3]
        #     ),
        # )

        if count % plotsteps == 0:
            print(count, time, (dynamics.state.m_ind.coeffs * dynamics.state.m_ind_to_Br)[:3])
            fn = os.path.join(fig_directory, "new_" + str(filecount).zfill(3) + ".png")
            filecount += 1
            title = "t = {:.3} s".format(time)
            Br = dynamics.state.get_Br(plt_grid)
            fig, paxn, paxs, axg = pynamit.globalplot(
                plt_grid.lon,
                plt_grid.lat,
                Br.reshape(plt_grid.lat.shape),
                title=title,
                returnplot=True,
                levels=Blevels,
                cmap="bwr",
                noon_longitude=lon0,
                extend="both",
            )

            W = dynamics.state.get_W(plt_grid) * 1e-3

            dynamics.state.update_E()
            Phi = dynamics.state.get_Phi(plt_grid) * 1e-3

            plt.savefig(fn)
            plt.close()

        if count > totalsteps:
            break

else:
    fig, paxn, paxs, axg = pynamit.globalplot(
        plt_grid.lon,
        plt_grid.lat,
        Br.reshape(plt_grid.lat.shape),
        returnplot=True,
        levels=Blevels,
        cmap="bwr",
        noon_longitude=lon0,
        extend="both",
    )

    plt.show()
    plt.close()

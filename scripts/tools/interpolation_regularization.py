"""Investigate intepolation regularization and resolution."""

import numpy as np
import pynamit
import dipole
import pyhwm2014  # https://github.com/rilma/pyHWM14
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.rcParams["figure.constrained_layout.use"] = True

PLOT = True
SH_COMPARISON = True
GRID_COMPARISON = True
L_CURVE = True

WIND = False
CONDUCTANCE = False
CURRENT = True

MIN_NMAX = 30
MAX_NMAX = 30
MMAX_EQUALS_NMAX = False
MMAX = 3
NMAX_STEP = 10
MIN_REG_LAMBDA_LOG = -6
MAX_REG_LAMBDA_LOG = -1
REG_LAMBDA_LOG_STEPS = 21

rtol = 1e-15
Ncs = 70

date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_lon = d.mlt2mlon(12, date)  # noon longitude

# Define cubed sphere basis and grid
cs_basis = pynamit.CSBasis(Ncs)
output_grid = pynamit.Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
output_weights = None

# Use regular grid from PyHWM14.
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

u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")
input_grid = pynamit.Grid(lat=u_lat.flatten(), lon=u_lon.flatten())

if CONDUCTANCE:
    # Get and set conductance input.
    from lompe import conductance

    Kp = 5
    hall, pedersen = conductance.hardy_EUV(
        input_grid.lon, input_grid.lat, Kp, date, starlight=1, dipole=True
    )

    input_grid_values = hall
    input_weights = None
    field_type = "scalar"
    nmin = 0

    interpolated_data = cs_basis.interpolate_scalar(
        hall, input_grid.theta, input_grid.phi, output_grid.theta, output_grid.phi
    )

if WIND:
    # Get and set wind input.
    # hwm14Obj = pyhwm2014.HWM142D(
    #    alt=110.0,
    #    ap=[35, 35],
    #    glatlim=[-89.0, 88.0],
    #    glatstp=3.0,
    #    glonlim=[-180.0, 180.0],
    #    glonstp=8.0,
    #    option=6,
    #    verbose=False,
    #    ut=date.hour + date.minute / 60,
    #    day=date.timetuple().tm_yday,
    # )

    # hwm14Obj = pyhwm2014.HWM142D(
    #    alt=110.0,
    #    ap=[35, 35],
    #    glatlim=[-88.5, 88.5],
    #    glatstp=6.0,
    #    glonlim=[-180.0, 180.0],
    #    glonstp=12.0,
    #    option=6,
    #    verbose=False,
    #    ut=date.hour + date.minute / 60,
    #    day=date.timetuple().tm_yday,
    # )

    input_grid_values = np.array([u_theta, u_phi])
    # input_weights = np.tile(
    #    np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)
    # )
    input_weights = None
    field_type = "tangential"
    nmin = 1

    interpolated_east, interpolated_north, _ = cs_basis.interpolate_vector_components(
        u_phi,
        -u_theta,
        np.zeros_like(u_phi),
        input_grid.theta,
        input_grid.phi,
        output_grid.theta,
        output_grid.phi,
    )
    interpolated_data = np.array([-interpolated_north, interpolated_east])  # Convert to theta, phi

if CURRENT:
    import pyamps

    a = pyamps.AMPS(300, 0, -3, 20, 100)
    je, jn = a.get_total_current(mlat=input_grid.lat, mlt=input_grid.lon / 15)

    input_grid_values = np.array([-jn, je])
    input_weights = None
    field_type = "tangential"
    nmin = 1

    interpolated_east, interpolated_north, _ = cs_basis.interpolate_vector_components(
        je,
        jn,
        np.zeros_like(je),
        input_grid.theta,
        input_grid.phi,
        output_grid.theta,
        output_grid.phi,
    )
    interpolated_data = np.array([-interpolated_north, interpolated_east])  # convert to theta, phi


lon = output_grid.lon.flatten()
lat = output_grid.lat.flatten()

if GRID_COMPARISON:
    relative_grid_errors = []
if SH_COMPARISON:
    relative_coeff_errors = []
if L_CURVE:
    sh_norms = []
    sh_resiudal_norms = []
    reg_lambda_values = []

Nmax_values = []

for reg_lambda in np.logspace(MIN_REG_LAMBDA_LOG, MAX_REG_LAMBDA_LOG, REG_LAMBDA_LOG_STEPS):
    for Nmax in range(MIN_NMAX, MAX_NMAX + 1, NMAX_STEP):
        if MMAX_EQUALS_NMAX:
            Mmax = Nmax
        else:
            Mmax = MMAX

        Nmax_values.append(Nmax)

        sh_basis = pynamit.SHBasis(Nmax, Mmax, nmin)
        input_basis_evaluator = pynamit.BasisEvaluator(
            sh_basis, input_grid, weights=input_weights, reg_lambda=reg_lambda, pinv_rtol=rtol
        )
        output_basis_evaluator = pynamit.BasisEvaluator(
            sh_basis, output_grid, weights=output_weights, reg_lambda=reg_lambda, pinv_rtol=rtol
        )

        input_sh = pynamit.FieldExpansion(
            sh_basis,
            basis_evaluator=input_basis_evaluator,
            grid_values=input_grid_values,
            field_type=field_type,
        )

        print(
            "Interpolation with Nmax = %d, Mmax = %d:, reg lambda: %e" % (Nmax, Mmax, reg_lambda)
        )

        if L_CURVE:
            reg_lambda_values.append(reg_lambda)
            # sh_norms.append(np.linalg.norm(input_sh.coeffs))
            sh_norms.append(np.linalg.norm(input_sh.regularization_term(input_basis_evaluator)))
            input_sh_on_input_grid = input_sh.to_grid(input_basis_evaluator)
            sh_resiudal_norms.append(
                np.linalg.norm(input_sh_on_input_grid - input_grid_values)
                / np.linalg.norm(input_grid_values)
            )

        if GRID_COMPARISON:
            cs_interpolated_output = interpolated_data
            sh_interpolated_output = input_sh.to_grid(output_basis_evaluator)
            relative_grid_errors.append(
                np.linalg.norm(cs_interpolated_output - sh_interpolated_output)
                / np.linalg.norm(cs_interpolated_output)
            )
            print("   Relative grid error = %e" % (relative_grid_errors[-1]))

        if SH_COMPARISON:
            cs_interpolated_output_sh = pynamit.FieldExpansion(
                sh_basis,
                basis_evaluator=output_basis_evaluator,
                grid_values=interpolated_data,
                field_type=field_type,
            )
            relative_coeff_errors.append(
                np.linalg.norm(cs_interpolated_output_sh.coeffs - input_sh.coeffs)
                / np.linalg.norm(cs_interpolated_output_sh.coeffs)
            )
            print("   Relative coefficient error = %e" % (relative_coeff_errors[-1]))

        if PLOT:
            if GRID_COMPARISON:
                grid_fig, (grid_cs_ax, grid_sh_ax) = plt.subplots(
                    1,
                    2,
                    figsize=(20, 5),
                    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=noon_lon)},
                )
                grid_cs_ax.coastlines()
                grid_sh_ax.coastlines()

                grid_cs_ax.title.set_text("Cubed sphere")
                grid_sh_ax.set_title("Spherical harmonics")

                if field_type == "scalar":
                    # Scatter plot scalar field
                    grid_cs_ax.scatter(
                        lon,
                        lat,
                        c=cs_interpolated_output,
                        cmap="viridis",
                        transform=ccrs.PlateCarree(),
                    )
                    grid_sh_ax.scatter(
                        lon,
                        lat,
                        c=sh_interpolated_output,
                        cmap="viridis",
                        transform=ccrs.PlateCarree(),
                    )
                elif field_type == "tangential":
                    # Quiver plot tangential vector field
                    cs_quiver = grid_cs_ax.quiver(
                        lon,
                        lat,
                        np.split(cs_interpolated_output, 2)[1].flatten(),
                        -np.split(cs_interpolated_output, 2)[0].flatten(),
                        color="blue",
                        transform=ccrs.PlateCarree(),
                    )
                    grid_sh_ax.quiver(
                        lon,
                        lat,
                        np.split(sh_interpolated_output, 2)[1].flatten(),
                        -np.split(sh_interpolated_output, 2)[0].flatten(),
                        color="red",
                        scale=cs_quiver.scale,
                        transform=ccrs.PlateCarree(),
                    )

                plt.show()

            if SH_COMPARISON:
                coeff_fig, (coeff_cs_ax, coeff_sh_ax) = plt.subplots(1, 2, figsize=(20, 5))
                abs_coeff_cs = np.abs(cs_interpolated_output_sh.coeffs)
                abs_coeff_sh = np.abs(input_sh.coeffs)

                coeff_cs_ax.set_title("Cubed sphere coefficient magnitudes")
                coeff_sh_ax.set_title("Spherical harmonics coefficient magnitudes")

                if field_type == "scalar":
                    coeff_cs_ax.plot(abs_coeff_cs, label="CS")
                    coeff_sh_ax.plot(abs_coeff_sh, label="SH")
                elif field_type == "tangential":
                    # Plot curl free and divergence free coefficients
                    coeff_cs_ax.plot(abs_coeff_cs[0], label="CF")
                    coeff_cs_ax.plot(abs_coeff_cs[1], label="DF")
                    coeff_sh_ax.plot(abs_coeff_sh[0], label="CF")
                    coeff_sh_ax.plot(abs_coeff_sh[1], label="DF")

                min_coeff = min(np.min(abs_coeff_cs), np.min(abs_coeff_sh))
                max_coeff = max(np.max(abs_coeff_cs), np.max(abs_coeff_sh))
                coeff_cs_ax.set_ylim(min_coeff * 0.75, max_coeff * 1.25)
                coeff_sh_ax.set_ylim(min_coeff * 0.75, max_coeff * 1.25)

                coeff_cs_ax.legend()
                coeff_sh_ax.legend()

                coeff_cs_ax.set_yscale("log")
                coeff_sh_ax.set_yscale("log")

                plt.show()

# Plot errors.
if GRID_COMPARISON or SH_COMPARISON:
    if GRID_COMPARISON:
        plt.plot(Nmax_values, relative_grid_errors, label="Grid values")
        plt.yscale("log")
        plt.xlabel("Nmax = Mmax")
        plt.ylabel("Error (relative to CS interpolation)")

    if SH_COMPARISON:
        plt.plot(Nmax_values, relative_coeff_errors, label="Coefficients")
        plt.yscale("log")
        plt.xlabel("Nmax = Mmax")
        plt.ylabel("Error (relative to CS interpolation)")

    plt.legend()
    plt.show()

if L_CURVE:
    scatter = plt.plot(sh_resiudal_norms, sh_norms)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Residual norms")
    plt.ylabel("Regularization term norms")

    for i, reg_lambda_val in enumerate(reg_lambda_values):
        plt.annotate(
            f"{reg_lambda_val:.1e}",
            (sh_resiudal_norms[i], sh_norms[i]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
        )

    plt.show()

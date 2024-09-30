import numpy as np
import pynamit
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

PLOT = True
SH_COMPARISON = True
GRID_COMPARISON = True
MIN_NMAX_MMAX = 10
MAX_NMAX_MMAX = 60
NMAX_MMAX_STEP = 10
REG_LAMBDA = 0.0

rtol = 1e-15
Ncs = 70

date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
nooNmax_Mmaxlon = d.mlt2mlon(12, date) # noon longitude

## WIND INPUT
#hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3.,
#                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-88.5, 88.5], glatstp = 1.5,
                             glonlim=[-180., 180.], glonstp = 3., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
u_grid = pynamit.Grid(lat = u_lat.flatten(), lon = u_lon.flatten())

## CS PROJECTION
csp = pynamit.CSProjection(Ncs)
csp_grid = pynamit.Grid(theta = csp.arr_theta, phi = csp.arr_phi)

interpolated_east, interpolated_north, _ = csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), u_grid.theta, u_grid.phi, csp_grid.theta, csp_grid.phi)
interpolated_data = np.hstack((-interpolated_north, interpolated_east)) # convert to theta, phi

lon = csp_grid.lon.flatten()
lat = csp_grid.lat.flatten()

if GRID_COMPARISON:
    relative_grid_errors = []
if SH_COMPARISON:
    relative_coeff_errors = []


for Nmax_Mmax in range(MIN_NMAX_MMAX, MAX_NMAX_MMAX + 1, NMAX_MMAX_STEP):
    sh_basis = pynamit.SHBasis(Nmax_Mmax, Nmax_Mmax)
    input_basis_evaluator = pynamit.BasisEvaluator(sh_basis, u_grid, pinv_rtol = rtol, weights = np.sin(np.deg2rad(90 - u_lat.flatten())), reg_lambda = REG_LAMBDA)
    state_basis_evaluator = pynamit.BasisEvaluator(sh_basis, csp_grid, pinv_rtol = rtol, reg_lambda = REG_LAMBDA)

    sh_interpolated_u = pynamit.Vector(sh_basis, basis_evaluator = input_basis_evaluator, grid_values = np.hstack((u_theta, u_phi)), type = 'tangential')

    print("Interpolation with Nmax = %d, Mmax = %d:" % (Nmax_Mmax, Nmax_Mmax))

    if GRID_COMPARISON:
        cs_interpolated_u_on_grid = interpolated_data
        sh_interpolated_u_on_grid = sh_interpolated_u.to_grid(state_basis_evaluator).flatten()
        relative_grid_errors.append(np.linalg.norm(cs_interpolated_u_on_grid - sh_interpolated_u_on_grid)/np.linalg.norm(cs_interpolated_u_on_grid))
        print("   Relative grid error = %e" % (relative_grid_errors[-1]))

    if SH_COMPARISON:
        cs_interpolated_u = pynamit.Vector(sh_basis, basis_evaluator = state_basis_evaluator, grid_values = interpolated_data, type = 'tangential')
        relative_coeff_errors.append(np.linalg.norm(cs_interpolated_u.coeffs - sh_interpolated_u.coeffs)/np.linalg.norm(cs_interpolated_u.coeffs))
        print("   Relative coefficient error = %e" % (relative_coeff_errors[-1]))

    if PLOT:
        if GRID_COMPARISON:
            grid_fig, (grid_cs_ax, grid_sh_ax) = plt.subplots(1, 2, figsize=(20, 5), layout = 'constrained', subplot_kw={'projection': ccrs.PlateCarree(central_longitude = nooNmax_Mmaxlon)})
            grid_cs_ax.coastlines()
            grid_sh_ax.coastlines()

            grid_cs_ax.title.set_text("Cubed sphere")
            grid_sh_ax.set_title("Spherical harmonics")

            # Plot grid wind field
            cs_quiver = grid_cs_ax.quiver(lon, lat, np.split(cs_interpolated_u_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_on_grid, 2)[0].flatten(), color='blue', transform=ccrs.PlateCarree())
            grid_sh_ax.quiver(lon, lat, np.split(sh_interpolated_u_on_grid, 2)[1].flatten(), -np.split(sh_interpolated_u_on_grid, 2)[0].flatten(), color='red', scale = cs_quiver.scale, transform=ccrs.PlateCarree())

            plt.show()

        if SH_COMPARISON:
            coeff_fig, (coeff_cs_ax, coeff_sh_ax) = plt.subplots(1, 2, figsize=(20, 5), layout = 'constrained')
            abs_coeff_cs = np.abs(cs_interpolated_u.coeffs)
            abs_coeff_sh = np.abs(sh_interpolated_u.coeffs)

            coeff_cs_ax.set_title("Cubed sphere coefficient magnitudes")
            coeff_sh_ax.set_title("Spherical harmonics coefficient magnitudes")

            # Plot curl free and divergence free coefficients
            coeff_cs_ax.plot(abs_coeff_cs[0], label = "CF")
            coeff_cs_ax.plot(abs_coeff_cs[1], label = "DF")
            coeff_sh_ax.plot(abs_coeff_sh[0], label = "CF")
            coeff_sh_ax.plot(abs_coeff_sh[1], label = "DF")

            min_coeff = min(np.min(abs_coeff_cs), np.min(abs_coeff_sh))
            max_coeff = max(np.max(abs_coeff_cs), np.max(abs_coeff_sh))
            coeff_cs_ax.set_ylim(min_coeff*0.75, max_coeff*1.25)
            coeff_sh_ax.set_ylim(min_coeff*0.75, max_coeff*1.25)

            coeff_cs_ax.legend()
            coeff_sh_ax.legend()

            coeff_cs_ax.set_yscale("log")
            coeff_sh_ax.set_yscale("log")

            plt.show()

# Plot errors
if GRID_COMPARISON:
    plt.plot(relative_grid_errors, label = "Grid values")
    plt.yscale("log")
    plt.xlabel("Nmax = Mmax")
    plt.ylabel("Error (relative to CS interpolation)")

if SH_COMPARISON:
    plt.plot(relative_coeff_errors, label = "Coefficients")
    plt.yscale("log")
    plt.xlabel("Nmax = Mmax")
    plt.ylabel("Error (relative to CS interpolation)")

plt.legend()

plt.show()

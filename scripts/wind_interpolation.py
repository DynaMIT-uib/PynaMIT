import numpy as np
import pynamit
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
from pynamit.cubed_sphere.cubed_sphere import csp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

PLOT = True
PLOT_CF_DF_DIFF = False

rtol = 1e-15
Ncs = 70

date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_mlon = d.mlt2mlon(12, date) # noon longitude

## WIND INPUT
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

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

relative_errors = []

for Nmax_Mmax in range(1, 30):
    sh_basis = pynamit.SHBasis(Nmax_Mmax, Nmax_Mmax)
    input_basis_evaluator = pynamit.BasisEvaluator(sh_basis, u_grid, pinv_rtol = rtol, weights = np.sin(np.deg2rad(90 - u_lat.flatten())))
    state_basis_evaluator = pynamit.BasisEvaluator(sh_basis, csp_grid, pinv_rtol = rtol)

    cs_interpolated_u = pynamit.Vector(sh_basis, basis_evaluator = state_basis_evaluator, grid_values = interpolated_data, type = 'tangential')
    sh_interpolated_u = pynamit.Vector(sh_basis, basis_evaluator = input_basis_evaluator, grid_values = np.hstack((u_theta, u_phi)), type = 'tangential')

    if PLOT:
        ## Evaluate full wind field on cubed sphere grid
        cs_interpolated_u_on_grid = cs_interpolated_u.to_grid(state_basis_evaluator)
        sh_interpolated_u_on_grid = sh_interpolated_u.to_grid(state_basis_evaluator)

        ## Plot full wind field
        full_fig, (full_cs_ax, full_sh_ax) = plt.subplots(1, 2, figsize=(20, 5), layout = 'constrained', subplot_kw={'projection': ccrs.PlateCarree(central_longitude = noon_mlon)})
        full_cs_ax.coastlines()
        full_sh_ax.coastlines()

        cs_quiver = full_cs_ax.quiver(lon, lat, np.split(cs_interpolated_u_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_on_grid, 2)[0].flatten(), color='blue', transform=ccrs.PlateCarree())
        full_sh_ax.quiver(lon, lat, np.split(sh_interpolated_u_on_grid, 2)[1].flatten(), -np.split(sh_interpolated_u_on_grid, 2)[0].flatten(), color='red', scale = cs_quiver.scale, transform=ccrs.PlateCarree())

        full_cs_ax.title.set_text("Cubed Sphere")
        full_sh_ax.set_title("Spherical Harmonics")
        plt.show()

        if PLOT_CF_DF_DIFF:
            cf_df_diff_fig, (cf_diff_ax, df_diff_ax) = plt.subplots(1, 2, figsize=(20, 5), layout = 'constrained', subplot_kw={'projection': ccrs.PlateCarree(central_longitude = noon_mlon)})
            cf_diff_ax.coastlines()
            df_diff_ax.coastlines()

            cs_interpolated_u_min_grad_on_grid = -state_basis_evaluator.G_grad.dot(np.split(cs_interpolated_u.coeffs, 2)[0])
            sh_interpolated_u_min_grad_on_grid = -state_basis_evaluator.G_grad.dot(np.split(sh_interpolated_u.coeffs, 2)[0])

            cs_interpolated_u_rxgrad_on_grid   = state_basis_evaluator.G_rxgrad.dot(np.split(cs_interpolated_u.coeffs, 2)[1])
            sh_interpolated_u_rxgrad_on_grid   = state_basis_evaluator.G_rxgrad.dot(np.split(sh_interpolated_u.coeffs, 2)[1])

            u_min_grad_on_grid_diff = cs_interpolated_u_min_grad_on_grid - sh_interpolated_u_min_grad_on_grid
            u_rxgrad_on_grid_diff = cs_interpolated_u_rxgrad_on_grid - sh_interpolated_u_rxgrad_on_grid

            cf_diff_ax.quiver(lon, lat, np.split(u_min_grad_on_grid_diff, 2)[1].flatten(), -np.split(u_min_grad_on_grid_diff, 2)[0].flatten(), color='blue', scale = cs_quiver.scale, transform=ccrs.PlateCarree())
            df_diff_ax.quiver(lon, lat, np.split(u_rxgrad_on_grid_diff, 2)[1].flatten(), -np.split(u_rxgrad_on_grid_diff, 2)[0].flatten(), color='red', scale = cs_quiver.scale, transform=ccrs.PlateCarree())

            cf_diff_ax.title.set_text("Curl Free Difference")
            df_diff_ax.set_title("Divergence Free Difference")

            plt.show()

    relative_errors.append(np.linalg.norm(cs_interpolated_u.coeffs - sh_interpolated_u.coeffs)/np.linalg.norm(cs_interpolated_u.coeffs))
    print("Finished interpolation with Nmax = %d, Mmax = %d, relative error = %e" % (Nmax_Mmax, Nmax_Mmax, relative_errors[-1]))

# Plot errors
plt.plot(relative_errors)
plt.xlabel("Nmax = Mmax")
plt.ylabel("Error (relative to CS interpolation)")
plt.show()
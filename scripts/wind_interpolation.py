import numpy as np
import pynamit
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
from pynamit.cubed_sphere.cubed_sphere import csp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

WIND_FACTOR = 1 # scale wind by this factor
FLOAT_ERROR_MARGIN = 1e-6

dataset_filename_prefix = 'aurora2'
Nmax, Mmax, Ncs = 10, 10, 70
rk = RI / np.cos(np.deg2rad(np.r_[0: 70: 2]))**2 #int(80 / Nmax)])) ** 2
print(len(rk))


date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude
noon_mlon = d.mlt2mlon(12, date) # noon longitude
lon0 = d.mlt2mlon(12, date) # noon longitude

## SET UP SIMULATION OBJECT
dynamics = pynamit.Dynamics(dataset_filename_prefix = dataset_filename_prefix,
                            Nmax = Nmax,
                            Mmax = Mmax,
                            Ncs = Ncs,
                            RI = RI,
                            mainfield_kind = 'igrf',
                            FAC_integration_steps = rk,
                            ignore_PFAC = True,
                            connect_hemispheres = False,
                            latitude_boundary = latitude_boundary,
                            ih_constraint_scaling = 1e-5,
                            t0 = str(date))

## WIND INPUT
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten() * WIND_FACTOR, hwm14Obj.Uwind.flatten() * WIND_FACTOR)
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
u_grid = pynamit.Grid(lat = u_lat.flatten(), lon = u_lon.flatten())

input_basis_evaluator = pynamit.BasisEvaluator(dynamics.bases['u'], u_grid, dynamics.pinv_rtols['u'])
state_basis_evaluator = pynamit.BasisEvaluator(dynamics.bases['u'], dynamics.state_grid, dynamics.pinv_rtols['u'])

interpolated_east, interpolated_north, _ = csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), u_grid.theta, u_grid.phi, dynamics.state_grid.theta, dynamics.state_grid.phi)
interpolated_data = np.hstack((-interpolated_north, interpolated_east)) # convert to theta, phi

cs_interpolated_u = pynamit.Vector(dynamics.bases['u'], basis_evaluator = state_basis_evaluator, grid_values = interpolated_data, type = 'tangential')
sh_interpolated_u = pynamit.Vector(dynamics.bases['u'], basis_evaluator = input_basis_evaluator, grid_values = np.hstack((u_theta, u_phi)), type = 'tangential')

cs_interpolated_u_on_grid = cs_interpolated_u.to_grid(state_basis_evaluator)
sh_interpolated_u_on_grid = sh_interpolated_u.to_grid(state_basis_evaluator)


## Curl free components
cs_interpolated_u_min_grad_on_grid = -state_basis_evaluator.G_grad.dot(np.split(cs_interpolated_u.coeffs, 2)[0])
sh_interpolated_u_min_grad_on_grid = -state_basis_evaluator.G_grad.dot(np.split(sh_interpolated_u.coeffs, 2)[0])

fig3, (ax13, ax23) = plt.subplots(1, 2, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude = lon0)})
ax13.coastlines()
ax23.coastlines()

Q = ax13.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(cs_interpolated_u_min_grad_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_min_grad_on_grid, 2)[0].flatten(), color='blue', transform=ccrs.PlateCarree())
ax23.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(sh_interpolated_u_min_grad_on_grid, 2)[1].flatten(), -np.split(sh_interpolated_u_min_grad_on_grid, 2)[0].flatten(), color='red', scale = Q.scale, transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()

## Divergence free components
cs_interpolated_u_rxgrad_on_grid   = state_basis_evaluator.G_rxgrad.dot(np.split(cs_interpolated_u.coeffs, 2)[1])
sh_interpolated_u_rxgrad_on_grid   = state_basis_evaluator.G_rxgrad.dot(np.split(sh_interpolated_u.coeffs, 2)[1])

fig4, (ax14, ax24) = plt.subplots(1, 2, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude = lon0)})
ax14.coastlines()
ax24.coastlines()

Q = ax14.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(cs_interpolated_u_rxgrad_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_rxgrad_on_grid, 2)[0].flatten(), color='blue', transform=ccrs.PlateCarree())
ax24.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(sh_interpolated_u_rxgrad_on_grid, 2)[1].flatten(), -np.split(sh_interpolated_u_rxgrad_on_grid, 2)[0].flatten(), scale = Q.scale, color='red', transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()


## Full wind field
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude = lon0)})
ax1.coastlines()
ax2.coastlines()
Q = ax1.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(cs_interpolated_u_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_on_grid, 2)[0].flatten(), color='blue', transform=ccrs.PlateCarree())
ax2.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(sh_interpolated_u_on_grid, 2)[1].flatten(), -np.split(sh_interpolated_u_on_grid, 2)[0].flatten(), color='red', scale = Q.scale, transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()

## Difference
fig2, (ax12, ax22) = plt.subplots(1, 2, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude = lon0)})
ax12.coastlines()
ax22.coastlines()

ax12.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(cs_interpolated_u_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_on_grid, 2)[0].flatten(), scale = Q.scale, color='blue', transform=ccrs.PlateCarree())
ax22.quiver(dynamics.state_grid.lon.flatten(), dynamics.state_grid.lat.flatten(), np.split(cs_interpolated_u_on_grid - sh_interpolated_u_on_grid, 2)[1].flatten(), -np.split(cs_interpolated_u_on_grid - sh_interpolated_u_on_grid, 2)[0].flatten(), scale = Q.scale, color='red', transform=ccrs.PlateCarree())

plt.tight_layout()

plt.show()

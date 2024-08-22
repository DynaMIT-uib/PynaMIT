import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
from pynamit.cubed_sphere.cubed_sphere import csp

RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

WIND_FACTOR = 1 # scale wind by this factor
FLOAT_ERROR_MARGIN = 1e-6

dataset_filename_prefix = 'aurora2'
Nmax, Mmax, Ncs = 50, 50, 50
rk = RI / np.cos(np.deg2rad(np.r_[0: 70: 2]))**2 #int(80 / Nmax)])) ** 2
print(len(rk))



date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude
noon_mlon = d.mlt2mlon(12, date) # noon longitude

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

# Scatter plot of the interpolated wind
fig1 = pynamit.globalplot(lon = dynamics.state_grid.lon, lat = dynamics.state_grid.lat, data = np.split(cs_interpolated_u_on_grid.grid_values, 2)[0], title = 'CS interpolated u_theta')
fig2 = pynamit.globalplot(lon = dynamics.state_grid.lon, lat = dynamics.state_grid.lat, data = np.split(cs_interpolated_u_on_grid.grid_values, 2)[1], title = 'CS interpolated u_phi')
fig3 = pynamit.globalplot(lon = dynamics.state_grid.lon, lat = dynamics.state_grid.lat, data = np.split(sh_interpolated_u_on_grid.grid_values, 2)[0], title = 'SH interpolated u_theta')
fig4 = pynamit.globalplot(lon = dynamics.state_grid.lon, lat = dynamics.state_grid.lat, data = np.split(sh_interpolated_u_on_grid.grid_values, 2)[1], title = 'SH interpolated u_phi')



dynamics.set_u(u_theta = u_theta, u_phi = u_phi, lat = u_lat, lon = u_lon)



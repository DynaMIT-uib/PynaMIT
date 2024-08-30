import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy

RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

STEADY_STATE_INITIALIZATION = True
STEADY_STATE_ITERATIONS = 500
UNDER_RELAXATION_FACTOR = 0.5

WIND_FACTOR = 1 # scale wind by this factor
FLOAT_ERROR_MARGIN = 1e-6

dataset_filename_prefix = 'aurora2'
Nmax, Mmax, Ncs = 10, 10, 10
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
                            ignore_PFAC = False,
                            connect_hemispheres = True,
                            latitude_boundary = latitude_boundary,
                            ih_constraint_scaling = 1e-5,
                            t0 = str(date))

print('Setting wind', flush = True)
## WIND INPUT
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-88.5, 88.5], glatstp = 1.5,
                             glonlim=[-180., 180.], glonstp = 3., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten() * WIND_FACTOR, hwm14Obj.Uwind.flatten() * WIND_FACTOR)
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
#u_lat, u_lon, u_phi, u_theta = np.load('ulat.npy'), np.load('ulon.npy'), np.load('uphi.npy'), np.load('utheta.npy')
#u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')
dynamics.set_u(u_theta = u_theta, u_phi = u_phi, lat = u_lat, lon = u_lon, weights = np.sin(np.deg2rad(90 - u_lat.flatten())))

print('Setting conductance', flush = True)
## CONDUCTANCE INPUT
conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon

sza = conductance.sunlight.sza(conductance_lat, conductance_lon, date, degrees=True)
hall_EUV, pedersen_EUV = conductance.EUV_conductance(sza)
hall_EUV, pedersen_EUV = np.sqrt(hall_EUV**2 + 1), np.sqrt(pedersen_EUV**2 + 1) # add starlight
dynamics.set_conductance(hall_EUV, pedersen_EUV, lat = conductance_lat, lon = conductance_lon)

## jr STATIC INPUT
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
apx = apexpy.Apex(refh = (RI - RE) * 1e-3, date = 2020)
mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)
_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI-RE)*1e-3) # fix this
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jr = a.get_upward_current(mlat = mlat, mlt = mlt) * 1e-6
jr[np.abs(jr_lat) < 50] = 0 # filter low latitude jr

relaxation_time = 100.
final_time = 600.
jr_sampling_dt = 0.5
jr_period = 20.

if STEADY_STATE_INITIALIZATION:
    dynamics.set_jr(jr = jr, lat = jr_lat, lon = jr_lon)

    timeseries_keys = list(dynamics.timeseries.keys())
    if 'state' in timeseries_keys:
        timeseries_keys.remove('state')
    if timeseries_keys is not None:
        for key in timeseries_keys:
            dynamics.select_timeseries_data(key, interpolation = False)

    dynamics.state.impose_constraints()

    for iteration in range(STEADY_STATE_ITERATIONS):
        print('Calculating steady state', flush = True)
        mv = dynamics.steady_state_m_ind(m_imp = dynamics.state.m_imp.coeffs)
    
        print('Difference between iteration %d and iteration %d:' % (iteration, iteration + 1), np.linalg.norm(mv - dynamics.state.m_ind.coeffs), flush = True)
        dynamics.state.set_coeffs(m_ind = dynamics.state.m_ind.coeffs + UNDER_RELAXATION_FACTOR * (mv - dynamics.state.m_ind.coeffs))

        dynamics.state.impose_constraints()

# Create array that will store all jr values
time_values = np.arange(0, final_time + jr_sampling_dt - FLOAT_ERROR_MARGIN, jr_sampling_dt, dtype = np.float64)
print('size of time_values:', time_values.size)
scaled_jr_values = np.zeros((time_values.size, jr.size), dtype = np.float64)

print('Interpolating jr', flush = True)
for time_index in range(time_values.size):
    if time_values[time_index] < relaxation_time - FLOAT_ERROR_MARGIN:
        scaled_jr = jr
    else:
        # sinusoidal variation of jr
        scaled_jr = jr * (1 + 0.5 * np.sin(2 * np.pi * (time_values[time_index] - relaxation_time) / jr_period))
    scaled_jr_values[time_index] = scaled_jr
    #print('Interpolated jr at t =', time_values[time_index], flush = True)

print('Setting jr', flush = True)
dynamics.set_jr(jr = scaled_jr_values, lat = jr_lat, lon = jr_lon, time = time_values)

print('Starting simulation', flush = True)
dynamics.evolve_to_time(final_time, interpolation = True)

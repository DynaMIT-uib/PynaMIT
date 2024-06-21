import numpy as np
import xarray as xr

class SaveLoad(object):
    """ Save and load simulation results """

    def initialize_save_file(self, model_settings, state, result_filename_prefix):
        """ Initialize save file """

        self.dataset = xr.Dataset()
        self.dataset.attrs.update(model_settings)

        # resolution parameters:
        resolution_params = {}
        resolution_params['Ncs'] = int(np.sqrt(state.num_grid.size / 6))
        resolution_params['N']   = state.basis.Nmax
        resolution_params['M']   = state.basis.Mmax
        resolution_params['FAC_integration_steps'] = state.FAC_integration_steps

        self.dataset.attrs.update(resolution_params)

        PFAC_matrix = state.m_imp_to_B_pol

        self.dataset.attrs.update({'PFAC_matrix':PFAC_matrix.flatten()})

        self.dataset['n'] = xr.DataArray(state.basis.n, coords = {'i': range(state.basis.num_coeffs)}, dims = ['i'], name = 'n')
        self.dataset['m'] = xr.DataArray(state.basis.m, coords = {'i': range(state.basis.num_coeffs)}, dims = ['i'], name = 'm')

        self.dataset.to_netcdf(result_filename_prefix + '.ncdf')
        print('Created {}'.format(result_filename_prefix))
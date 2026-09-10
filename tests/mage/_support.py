"""Small fake datasets shared by the MAGE workflow tests."""

from pathlib import Path

import h5py
import numpy as np

from pynamit.workflows.mage.prepared_forcing import (
    CONDUCTANCE_FLOOR_MODEL,
    HALL_CONDUCTANCE_FLOOR_S,
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    PEDERSEN_CONDUCTANCE_FLOOR_S,
)


class _FakeVariable:
    def __init__(self, values, **attrs):
        self.values = np.asanyarray(values)
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getitem__(self, item):
        return self.values[item]

    @property
    def shape(self):
        return self.values.shape


class _FakeDataset:
    def __init__(self, **variables):
        self.variables = {
            name: values if isinstance(values, _FakeVariable) else _FakeVariable(values)
            for name, values in variables.items()
        }


class _FakeH5(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})


def _write_projection_forcing(path: Path, *, hall_conductance=5.0) -> None:
    """Write a complete, tiny prepared-forcing contract."""
    latitude, longitude = np.meshgrid(
        np.array([-60.0, -20.0, 20.0, 60.0]), np.array([-135.0, -45.0, 45.0, 135.0]), indexing="ij"
    )
    step_shape = (2, *latitude.shape)
    hall = np.broadcast_to(np.asarray(hall_conductance, dtype=float), step_shape)

    with h5py.File(path, "w") as output:
        string_dtype = h5py.string_dtype("utf-8")
        times = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:10"], dtype=object)
        for name in ("time", "gamera_source_time", "remix_source_time"):
            output.create_dataset(name, data=times, dtype=string_dtype)
        for name in ("gamera_time_offset_seconds", "remix_time_offset_seconds"):
            output.create_dataset(name, data=np.zeros(2))
            output[name].attrs["units"] = "s"
        for name, values in {
            "boundary_radius": np.full(latitude.shape, 7.0e6),
            "boundary_solid_angle": np.full(latitude.shape, 4.0 * np.pi / latitude.size),
            "ionosphere_lat": latitude,
            "ionosphere_lon": longitude,
            "boundary_lat": latitude,
            "boundary_lon": longitude,
            "delta_Br": np.full(step_shape, 10.0),
            "jr": np.full(step_shape, 0.1),
            "SH": hall,
            "SP": np.full(step_shape, 10.0),
            "u_p_theta": np.full(step_shape, 20.0),
            "u_p_phi": np.full(step_shape, 50.0),
            "u_h_theta": np.full(step_shape, -10.0),
            "u_h_phi": np.full(step_shape, 30.0),
        }.items():
            output.create_dataset(name, data=values)
        for name, units in {
            "boundary_radius": "m",
            "boundary_solid_angle": "sr",
            "ionosphere_lat": "degree",
            "ionosphere_lon": "degree",
            "boundary_lat": "degree",
            "boundary_lon": "degree",
            "delta_Br": "nT",
            "jr": "uA m-2",
            "SH": "S",
            "SP": "S",
            "u_p_theta": "m s-1",
            "u_p_phi": "m s-1",
            "u_h_theta": "m s-1",
            "u_h_phi": "m s-1",
        }.items():
            output[name].attrs["units"] = units
        output.attrs["gamera_mag_m0_nT"] = -30_000.0
        output.attrs["main_field_B0_T"] = 3.0e-5
        output.attrs["main_field_B0_reference_radius_m"] = 6_371_200.0
        output.attrs["gamera_internal_dipole_moment_axis"] = [0.0, 0.0, -1.0]
        output.attrs["gamera_internal_magnetic_north_axis"] = [0.0, 0.0, 1.0]
        output.attrs["gamera_source_coordinate_system"] = "SM"
        output.attrs["gamera_sm_transform_time_convention"] = "kaiju_mjdrecalc_nearest_second"
        output.attrs["coordinate_system"] = "GEO"
        output.attrs["longitude_convention"] = "east_positive_degrees"
        output.attrs["fac_convention"] = "upward"
        output.attrs["radial_current_convention"] = "outward"
        output.attrs["remix_fac_interpolation"] = "kaiju_native_periodic"
        output.attrs["gamera_boundary_interpolation"] = (
            "gamera_native_periodic_bilinear_with_polar_mean"
        )
        output.attrs["gamera_background_reference"] = "cell_volume_average_split_B0"
        output.attrs["tiegcm_source_coordinate_system"] = "geographic"
        output.attrs["tiegcm_conductance_integration"] = (
            "radial_geometric_height_with_lower_dynamo_extension"
        )
        output.attrs["tiegcm_dynamo_bottom_ilev"] = -8.5
        output.attrs["tiegcm_dynamo_reference_height_m"] = 90_000.0
        output.attrs["tiegcm_pedersen_lower_scale_m"] = 5_000.0
        output.attrs["tiegcm_hall_lower_scale_m"] = 3_000.0
        output.attrs["conductance_floor_model"] = CONDUCTANCE_FLOOR_MODEL
        output.attrs["pedersen_conductance_floor_S"] = PEDERSEN_CONDUCTANCE_FLOOR_S
        output.attrs["hall_conductance_floor_S"] = HALL_CONDUCTANCE_FLOOR_S
        output.attrs["remix_grid_equatorward_sm_latitude_deg"] = 35.0
        output.attrs["ionosphere_radius_m"] = 6.5e6
        output.attrs["time_axis"] = "tiegcm_mtime_nominal"
        output.attrs["source_time_tolerance_seconds"] = 0.1
        output.attrs["kind"] = MAGE_FORCING_KIND
        output.attrs["version"] = MAGE_FORCING_VERSION
        output.attrs["complete"] = True

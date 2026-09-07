"""Scientific ground-field evaluation, independent of plotting."""

import numpy as np
import pytest
from kompe import SphericalGrid

from pynamit import Simulation
from pynamit.results import SimulationResults, evaluate_ground_magnetic_field
from pynamit.results.output_fields import build_ground_magnetic_field_operators


@pytest.mark.parametrize("main_field_kind", ["dipole", "radial"])
def test_ground_field_geographic_components_and_time_selection(tmp_path, main_field_kind):
    """Geographic sites and vector components use consistent frames."""
    simulation = Simulation(
        tmp_path,
        Nmax=2,
        Mmax=2,
        Ncs=4,
        enable_pfac_coupling=False,
        main_field_kind=main_field_kind,
        artifact_storage="netcdf",
    )
    basis = simulation.geometry.solid_harmonics.basis
    coefficient = np.asarray((basis.n == 1) & (basis.m == 0), dtype=float) * 1e-9
    spaces = simulation.data.schema.output_field_spaces["dynamic"]
    for time, scale in ((0.0, 1.0), (10.0, 2.0)):
        values = {name: np.zeros(space.shape) for name, space in spaces.items()}
        values["induced_Br"] = scale * coefficient
        simulation.data.output_series.add_entry("dynamic", values, time)
    simulation.data.output_series.save("dynamic", simulation.data.artifact_store)

    grid = SphericalGrid(lat=[[10.0, 45.0, 70.0]], lon=[[0.0, 90.0, -60.0]])
    main_field = simulation.main_field
    lat, lon = main_field.geo_to_model_coordinates(grid.lat, grid.lon)
    operators = build_ground_magnetic_field_operators(
        simulation.geometry, SphericalGrid(lat=lat, lon=lon)
    )
    radial = np.asarray(operators["radial"].matvec(coefficient))
    theta, phi = np.asarray(operators["tangential"].matvec(coefficient)).reshape(2, -1)
    _, _, east, north = main_field.model_to_geo_coordinates(lat, lon, east=phi, north=-theta)
    results = SimulationResults.from_directory(tmp_path)
    for source in (simulation, results):
        fields = evaluate_ground_magnetic_field(source, [10.0, 0.0], grid=grid)
        assert fields["radial"].shape == (1, 3, 2)
        assert fields["tangential"].shape == (2, 1, 3, 2)
        np.testing.assert_allclose(fields["radial"][0], radial[:, None] * [2.0, 1.0])
        np.testing.assert_allclose(
            fields["tangential"][:, 0],
            np.stack((-north, east))[..., None] * [2.0, 1.0],
            atol=1e-22,
        )
        all_times = evaluate_ground_magnetic_field(source, grid=grid)
        np.testing.assert_allclose(all_times["radial"], fields["radial"][..., ::-1])

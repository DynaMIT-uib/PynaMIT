import numpy as np
import pytest
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.gauss_legendre.gl_basis import GLBasis
from pynamit.primitives.field_spec import FieldSpec
from pynamit.primitives.grid import Grid
from pynamit.primitives.input_manager import InputManager
from pynamit.primitives.timeseries import Timeseries

# Shared Constants
N_sh = 10
N_lat = 45
N_lon = 90
lat = np.linspace(-90, 90, N_lat)
lon = np.linspace(0, 360, N_lon, endpoint=False)
theta = np.deg2rad(90 - lat)
phi = np.deg2rad(lon)
tt, pp = np.meshgrid(theta, phi, indexing="ij")
lat_mesh = 90 - np.rad2deg(tt)
lon_mesh = np.rad2deg(pp)


def _flat_grid(theta_vals=tt, phi_vals=pp):
    return Grid(theta=np.rad2deg(theta_vals).flatten(), phi=np.rad2deg(phi_vals).flatten())


def _evaluate_vector_coeffs(basis, coeffs, grid=None):
    target_grid = _flat_grid() if grid is None else grid
    coeffs = np.asarray(coeffs, dtype=float).reshape(2, basis.index_length)
    return basis.evaluate(coeffs, target_grid, vector_type="tangential")


@pytest.fixture
def setup_manager():
    def _create(var_type="scalar", var_name="hall", key="conductance"):
        basis = SHBasis(Nmax=N_sh, Mmax=N_sh, mean_free=False)
        sim_basis = GLBasis(Nmax=N_sh)
        variables_dict = {key: {var_name: var_type}}
        storage_specs = {key: FieldSpec(basis=basis, field_type=var_type, mean_free=False)}
        ts = Timeseries(storage_specs=storage_specs, variables=variables_dict)
        manager = InputManager(ts, sim_basis, variables_dict)
        return manager, basis, ts

    return _create


def test_fast_input_scalar(setup_manager):
    """Verify Fast SHT integration for Scalar fields."""
    manager, basis, ts = setup_manager()

    # Field: P_10 = cos(theta)
    field_values = np.cos(tt)
    input_data = {"hall": np.array([field_values.flatten()])}

    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs = ts.get_entry("conductance", 0.0)["hall"]

    # Check Coeff (1,0) should be 1.0 (Schmidt norm S_10=1)
    c_10 = coeffs[1]
    print(f"Scalar P_10 Coeff: {c_10}")
    assert np.abs(c_10 - 1.0) < 1e-10


def test_fast_input_scalar_batched(setup_manager):
    """Verify fast scalar path projects multiple time slices in one call."""
    manager, basis, ts = setup_manager()

    field_1 = np.cos(tt)
    field_2 = 0.5 * (3 * np.cos(tt) ** 2 - 1)
    input_data = {"hall": np.stack([field_1.flatten(), field_2.flatten()], axis=0)}

    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([0.0, 1.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs_0 = ts.get_entry("conductance", 0.0)["hall"]
    coeffs_1 = ts.get_entry("conductance", 1.0)["hall"]
    assert np.abs(coeffs_0[1] - 1.0) < 1e-10
    assert np.max(np.abs(coeffs_1)) > 0.9


def test_fast_input_vector(setup_manager):
    """Verify Fast SHT integration for Vector fields."""
    # Setup 'tangential' variable
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    coeffs_true = np.zeros((2, basis.index_length))
    coeffs_true[0, 1] = 1.0
    u_eval = _evaluate_vector_coeffs(basis, coeffs_true)
    val_tuple = (u_eval[0].reshape(-1), u_eval[1].reshape(-1))
    input_data = {"u": [val_tuple]}

    manager.interpolate_and_add_entry(
        key="wind",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs = ts.get_entry("wind", 0.0)["u"]

    # Vector Coeffs Structure: [Pol (0..L), Tor (0..L)] (flattened)
    # Expected: c_pol(1,0) = 1.0. All others 0.

    # Index 1 is Pol(1,0).
    c_pol_10 = coeffs[1]

    # Index L + ... is Tor. Tor starts at index_length.
    c_tor_start = basis.index_length

    print(f"Vector Pol_10 Coeff: {c_pol_10}")

    assert np.abs(c_pol_10 - coeffs_true[0, 1]) < 1e-10

    # Verify Toroidal is zero
    norm_tor = np.linalg.norm(coeffs[c_tor_start:])
    assert norm_tor < 1e-10


def test_fast_input_vector_batched(setup_manager):
    """Verify fast vector path projects multiple time slices in one call."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    coeffs_true_1 = np.zeros((2, basis.index_length))
    coeffs_true_2 = np.zeros((2, basis.index_length))
    coeffs_true_1[0, 1] = 1.0
    coeffs_true_2[0, 1] = 2.0
    u_eval_1 = _evaluate_vector_coeffs(basis, coeffs_true_1)
    u_eval_2 = _evaluate_vector_coeffs(basis, coeffs_true_2)
    packed = np.zeros((2, 2, u_eval_1.shape[1]))
    packed[0, 0] = u_eval_1[0].reshape(-1)
    packed[0, 1] = u_eval_1[1].reshape(-1)
    packed[1, 0] = u_eval_2[0].reshape(-1)
    packed[1, 1] = u_eval_2[1].reshape(-1)
    input_data = {"u": packed}

    manager.interpolate_and_add_entry(
        key="wind",
        input_data=input_data,
        time=np.array([0.0, 1.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs_0 = ts.get_entry("wind", 0.0)["u"]
    coeffs_1 = ts.get_entry("wind", 1.0)["u"]
    assert np.abs(coeffs_0[1] - coeffs_true_1[0, 1]) < 1e-10
    assert np.abs(coeffs_1[1] - coeffs_true_2[0, 1]) < 1e-10
    assert np.linalg.norm(coeffs_0[basis.index_length :]) < 1e-10
    assert np.linalg.norm(coeffs_1[basis.index_length :]) < 1e-10


def test_fast_input_vector_ndarray_packed_time_slice(setup_manager):
    """Verify fast vector path accepts ndarray packing from Dynamics.set_u (shape (T,2,N))."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    coeffs_true = np.zeros((2, basis.index_length))
    coeffs_true[0, 1] = 1.0
    u_eval = _evaluate_vector_coeffs(basis, coeffs_true)
    npts = u_eval.shape[1]

    # Mimic Dynamics.set_u storage after np.moveaxis(...): (time, 2, N)
    packed = np.zeros((1, 2, npts))
    packed[0, 0] = u_eval[0].reshape(-1)
    packed[0, 1] = u_eval[1].reshape(-1)
    input_data = {"u": packed}

    manager.interpolate_and_add_entry(
        key="wind",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs = ts.get_entry("wind", 0.0)["u"]
    c_pol_10 = coeffs[1]
    c_tor_start = basis.index_length
    norm_tor = np.linalg.norm(coeffs[c_tor_start:])

    assert np.abs(c_pol_10 - coeffs_true[0, 1]) < 1e-10
    assert norm_tor < 1e-10


def test_fast_input_vector_ndarray_packed_time_slice_with_stacked_weights(setup_manager):
    """Fast vector path should also accept duplicated component weights (2, N)."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    coeffs_true = np.zeros((2, basis.index_length))
    coeffs_true[0, 1] = 1.0
    u_eval = _evaluate_vector_coeffs(basis, coeffs_true)
    npts = u_eval.shape[1]
    packed = np.zeros((1, 2, npts))
    packed[0, 0] = u_eval[0].reshape(-1)
    packed[0, 1] = u_eval[1].reshape(-1)
    input_data = {"u": packed}

    w_theta = np.sin(theta)
    w_points = np.repeat(w_theta[:, None], N_lon, axis=1).reshape(-1)
    w_stacked = np.vstack([w_points, w_points])  # Matches mage_forcing_3 pattern (2, N)

    manager.interpolate_and_add_entry(
        key="wind",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
        sqrt_weights=w_stacked,
    )

    coeffs = ts.get_entry("wind", 0.0)["u"]
    assert np.abs(coeffs[1] - coeffs_true[0, 1]) < 1e-10
    assert np.linalg.norm(coeffs[basis.index_length :]) < 1e-10


def test_batched_slow_input_scalar_multi_rhs(setup_manager):
    """Verify non-fast scalar path still batches multiple RHS columns."""
    manager, basis, ts = setup_manager()
    manager.enable_fast_path = False

    field_1 = np.cos(tt)
    field_2 = 0.5 * (3 * np.cos(tt) ** 2 - 1)
    input_data = {"hall": np.stack([field_1.flatten(), field_2.flatten()], axis=0)}

    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([0.0, 1.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs_0 = ts.get_entry("conductance", 0.0)["hall"]
    coeffs_1 = ts.get_entry("conductance", 1.0)["hall"]
    assert np.abs(coeffs_0[1] - 1.0) < 1e-8
    assert np.max(np.abs(coeffs_1)) > 0.9


def test_fast_input_regularization(setup_manager):
    """Verify Regularization Damping."""
    manager, basis, ts = setup_manager()

    # Field: P_20 = 0.5 * (3 cos^2 - 1).
    # Unnorm P_20 coeff is 1.0.
    # Schmidt S_20 = 1.0.
    field_values = 0.5 * (3 * np.cos(tt) ** 2 - 1)
    input_data = {"hall": np.array([field_values.flatten()])}

    # Solve 1: No Reg (lambda=0)
    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
        reg_lambda=0.0,
    )
    c_base = ts.get_entry("conductance", 0.0)["hall"][
        2
    ]  # (l=2,m=0) is index 2 usually (00, 10, 20...)
    # Wait, indices: 0=(0,0), 1=(1,0), 2=(1,1-is-sin), ...
    # SHBasis index order: (0,0), (1,0), (1,1)...
    # Let's check index 2. Actually let's just find max.

    # Solve 2: High Reg (lambda=1.0)
    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([1.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
        reg_lambda=1.0,
    )
    c_reg = ts.get_entry("conductance", 1.0)["hall"]

    # Identify dominant index
    idx_dom = np.argmax(np.abs(ts.get_entry("conductance", 0.0)["hall"]))
    val_base = ts.get_entry("conductance", 0.0)["hall"][idx_dom]
    val_reg = c_reg[idx_dom]

    print(f"Reg Test: Base={val_base:.5f}, Reg={val_reg:.5f}")

    # Should be dampened
    assert val_base > 0.99  # Should be ~1
    # Should be dampened
    assert val_base > 0.99
    assert val_reg < val_base
    assert val_reg > 0


def test_fast_input_vector_coupled(setup_manager):
    """Verify Vector SHT for Coupled (m>0) modes with Poloidal/Toroidal mixing."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    idx_21 = basis.index_pairs.index((2, 1))
    coeffs_true = np.zeros((2, basis.index_length))
    coeffs_true[0, idx_21] = 1.0
    coeffs_true[1, idx_21] = 0.5
    u_eval = _evaluate_vector_coeffs(basis, coeffs_true)

    input_data = {"u": [(u_eval[0].reshape(-1), u_eval[1].reshape(-1))]}

    manager.interpolate_and_add_entry(
        key="wind",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )

    coeffs = ts.get_entry("wind", 0.0)["u"]

    # Verify Poloidal (2,1)
    c_pol_rec = coeffs[idx_21]
    print(f"Coupled Pol(2,1): {c_pol_rec} (Expect 1.0)")
    assert np.abs(c_pol_rec - coeffs_true[0, idx_21]) < 1e-10

    # Verify Toroidal (2,1)
    # Toroidal part is second half of coeffs
    c_tor_rec = coeffs[basis.index_length + idx_21]
    print(f"Coupled Tor(2,1): {c_tor_rec} (Expect 0.5)")
    assert np.abs(c_tor_rec - coeffs_true[1, idx_21]) < 1e-10


def test_fast_input_vector_coupled_with_wrapped_longitude_endpoint(setup_manager):
    """Fast vector path should handle a duplicated periodic longitude endpoint."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    lon_wrapped = np.linspace(-180, 180, N_lon + 1)
    phi_wrapped = np.deg2rad(lon_wrapped)
    tt_wrapped, pp_wrapped = np.meshgrid(theta, phi_wrapped, indexing="ij")
    lat_mesh_wrapped = 90 - np.rad2deg(tt_wrapped)
    lon_mesh_wrapped = np.rad2deg(pp_wrapped)

    idx_21 = basis.index_pairs.index((2, 1))
    grid_flat = _flat_grid(tt_wrapped, pp_wrapped)
    coeffs_true = np.zeros((2, basis.index_length))
    coeffs_true[0, idx_21] = 1.0
    coeffs_true[1, idx_21] = 0.5
    u_eval = _evaluate_vector_coeffs(basis, coeffs_true, grid=grid_flat)

    input_data = {"u": [(u_eval[0].reshape(-1), u_eval[1].reshape(-1))]}

    manager.interpolate_and_add_entry(
        key="wind",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh_wrapped.flatten(),
        lon=lon_mesh_wrapped.flatten(),
    )

    coeffs = ts.get_entry("wind", 0.0)["u"]
    c_pol_rec = coeffs[idx_21]
    c_tor_rec = coeffs[basis.index_length + idx_21]

    assert np.abs(c_pol_rec - coeffs_true[0, idx_21]) < 1e-10
    assert np.abs(c_tor_rec - coeffs_true[1, idx_21]) < 1e-10


def test_fast_input_weights(setup_manager):
    """Verify Weights impact the solution."""
    manager, basis, ts = setup_manager()

    # Field: 1.0 everywhere. P_00=1. C_00=1.
    val = np.ones_like(tt)
    # Add a huge outlier at the equator
    mid = N_lat // 2
    val[mid, :] = 1000.0  # Anomaly

    input_data = {"hall": np.array([val.flatten()])}

    # Case 1: Unweighted (Outlier affects result)
    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([0.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
    )
    c_unweighted = ts.get_entry("conductance", 0.0)["hall"][0]

    # Case 2: Weighted (Zero weight on outlier)
    # sqrt_weights. Same shape as lat/lon.
    w = np.ones_like(tt)
    w[mid, :] = 0.0  # Ignore outlier row

    manager.interpolate_and_add_entry(
        key="conductance",
        input_data=input_data,
        time=np.array([1.0]),
        projection_basis=basis,
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
        sqrt_weights=w.flatten(),
    )
    c_weighted = ts.get_entry("conductance", 1.0)["hall"][0]

    print(f"Unweighted Mean: {c_unweighted}")
    print(f"Weighted Mean: {c_weighted} (Expect 1.0)")

    assert c_unweighted > 1.1  # Skewed by outlier
    assert np.abs(c_weighted - 1.0) < 1e-10  # Perfectly ignored

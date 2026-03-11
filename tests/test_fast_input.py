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

    # Vector Field: Pure Poloidal Dipole (l=1, m=0)
    # V = -Grad(Y_10). Y_10 = cos(theta).
    # u_theta = -d/dth(cos th) = sin(theta)
    # u_phi = -1/sin d/dph = 0

    u_th = np.sin(tt)
    u_ph = np.zeros_like(tt)

    # Input data must be tuple for tangential
    # Note: Timeseries structure for 'u': input_data['u'] is the value.
    # We pass array of objects? Or list of tuples?
    # InputManager logic: raw_values = input_data[var][time_index]
    # So input_data['u'] should be a list [ (u_th, u_ph) ] (length=1 for time)

    # Flatten components
    val_tuple = (u_th.flatten(), u_ph.flatten())
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

    assert np.abs(c_pol_10 - 1.0) < 1e-10

    # Verify Toroidal is zero
    norm_tor = np.linalg.norm(coeffs[c_tor_start:])
    assert norm_tor < 1e-10


def test_fast_input_vector_batched(setup_manager):
    """Verify fast vector path projects multiple time slices in one call."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    u_th_1 = np.sin(tt)
    u_ph_1 = np.zeros_like(tt)
    u_th_2 = 2.0 * np.sin(tt)
    u_ph_2 = np.zeros_like(tt)
    packed = np.zeros((2, 2, u_th_1.size))
    packed[0, 0] = u_th_1.flatten()
    packed[0, 1] = u_ph_1.flatten()
    packed[1, 0] = u_th_2.flatten()
    packed[1, 1] = u_ph_2.flatten()
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
    assert np.abs(coeffs_0[1] - 1.0) < 1e-10
    assert np.abs(coeffs_1[1] - 2.0) < 1e-10
    assert np.linalg.norm(coeffs_0[basis.index_length :]) < 1e-10
    assert np.linalg.norm(coeffs_1[basis.index_length :]) < 1e-10


def test_fast_input_vector_ndarray_packed_time_slice(setup_manager):
    """Verify fast vector path accepts ndarray packing from Dynamics.set_u (shape (T,2,N))."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    u_th = np.sin(tt)
    u_ph = np.zeros_like(tt)
    npts = u_th.size

    # Mimic Dynamics.set_u storage after np.moveaxis(...): (time, 2, N)
    packed = np.zeros((1, 2, npts))
    packed[0, 0] = u_th.flatten()
    packed[0, 1] = u_ph.flatten()
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

    assert np.abs(c_pol_10 - 1.0) < 1e-10
    assert norm_tor < 1e-10


def test_fast_input_vector_ndarray_packed_time_slice_with_stacked_weights(setup_manager):
    """Fast vector path should also accept duplicated component weights (2, N)."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    u_th = np.sin(tt)
    u_ph = np.zeros_like(tt)
    npts = u_th.size
    packed = np.zeros((1, 2, npts))
    packed[0, 0] = u_th.flatten()
    packed[0, 1] = u_ph.flatten()
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
    assert np.abs(coeffs[1] - 1.0) < 1e-10
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

    # Target: l=2, m=1
    # Poloidal Field: u_pol = -Grad Y_21
    # Toroidal Field: u_tor = Curl rY_21

    # We construct the field explicitly using basis functions
    # Y_21 exists in basis.
    # idx_21 = basis.get_index(2, 1) -> Replace with manual lookup
    idx_21 = basis.index_pairs.index((2, 1))

    # 1. Generate grid values for Y_21 derivatives
    # We need G_theta (dP/dth) and G_phi (1/sin dY/dphi)
    # SHBasis.get_evaluation_matrix can gives us these.
    # Note: Flattened grid required for get_evaluation_matrix
    grid_flat = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())

    G_th = basis.get_evaluation_matrix(grid_flat, derivative="theta")
    G_ph = basis.get_evaluation_matrix(grid_flat, derivative="phi")

    # Extract columns for (2,1) Cosine and Sine
    # idx_21 is the index of the coefficient.
    # For Cosine coeff: c_21=1. This corresponds to index `idx_21`.
    # For Sine coeff: s_21=1. This corresponds to `basis.index_length + idx_21 - ...`
    # Actually easier: Create a full coeff vector for desired Pol/Tor fields and project.

    c_pol_true = np.zeros(basis.index_length)
    c_tor_true = np.zeros(basis.index_length)

    # Set Poloidal (2,1) Cosine = 1.0
    c_pol_true[idx_21] = 1.0

    # Set Toroidal (2,1) Sine = 0.5 (Mix types)
    # Sine index logic:
    # SHBasis stores Cosine coeffs (0..N) then Sine coeffs?
    # No, SHBasis c_nm/s_nm arrays are filters.
    # coeffs array is ordered by index_pairs.
    # Wait, implementation of `grid_to_basis_fast` returns flattened coeffs.
    # But SHBasis usually returns separate Cnm, Snm arrays?
    # No, SHBasis returns a single 1D array.
    # Order: [ (l,m) pairs ].
    # Wait, SHBasis.index_pairs enumerates all (l,m).
    # For each (l,m), we have a value?
    # Only if it's Real SH.
    # Actually SHBasis stores Real SH coeffs.
    # Usually: [C_00, C_10, C_11c, C_11s, ...]
    # Let's verify SHBasis indexing.
    # basis.cnm_filter and basis.snm_filter map to the 1D array.

    # Let's rely on constructing the field via get_evaluation_matrix
    # u_th = -dY/dth * c_pol + 1/sin dY/dphi * c_tor
    # u_ph = -1/sin dY/dphi * c_pol - dY/dth * c_tor

    # Pol contribution (C_21^c = 1.0)
    # Y_21^c -> dY/dth is G_th[:, idx_21].
    #           1/sin dY/dphi is G_ph[:, idx_21].

    u_th_flat = -G_th[:, idx_21] * 1.0
    u_ph_flat = -G_ph[:, idx_21] * 1.0

    # 2. Add Toroidal Contribution (C_21^c = 0.5) to mix things up
    # Toroidal field from same harmonic (Cosine 2,1)
    # u_phi   += +G_th * 0.5

    u_th_flat += G_ph[:, idx_21] * 0.5
    u_ph_flat -= G_th[:, idx_21] * 0.5

    input_data = {"u": [(u_th_flat, u_ph_flat)]}

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
    assert np.abs(c_pol_rec - 1.0) < 1e-10

    # Verify Toroidal (2,1)
    # Toroidal part is second half of coeffs
    c_tor_rec = coeffs[basis.index_length + idx_21]
    print(f"Coupled Tor(2,1): {c_tor_rec} (Expect 0.5)")
    assert np.abs(c_tor_rec - 0.5) < 1e-10


def test_fast_input_vector_coupled_with_wrapped_longitude_endpoint(setup_manager):
    """Fast vector path should handle a duplicated periodic longitude endpoint."""
    manager, basis, ts = setup_manager(var_type="tangential", var_name="u", key="wind")

    lon_wrapped = np.linspace(-180, 180, N_lon + 1)
    phi_wrapped = np.deg2rad(lon_wrapped)
    tt_wrapped, pp_wrapped = np.meshgrid(theta, phi_wrapped, indexing="ij")
    lat_mesh_wrapped = 90 - np.rad2deg(tt_wrapped)
    lon_mesh_wrapped = np.rad2deg(pp_wrapped)

    idx_21 = basis.index_pairs.index((2, 1))
    grid_flat = Grid(theta=np.rad2deg(tt_wrapped).flatten(), phi=np.rad2deg(pp_wrapped).flatten())

    G_th = basis.get_evaluation_matrix(grid_flat, derivative="theta")
    G_ph = basis.get_evaluation_matrix(grid_flat, derivative="phi")

    u_th_flat = -G_th[:, idx_21] * 1.0
    u_ph_flat = -G_ph[:, idx_21] * 1.0
    u_th_flat += G_ph[:, idx_21] * 0.5
    u_ph_flat -= G_th[:, idx_21] * 0.5

    input_data = {"u": [(u_th_flat, u_ph_flat)]}

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

    assert np.abs(c_pol_rec - 1.0) < 1e-10
    assert np.abs(c_tor_rec - 0.5) < 1e-10


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

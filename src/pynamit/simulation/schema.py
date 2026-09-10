"""Simulation storage schema construction.

This module describes existing numerical bases with ``CoefficientSpace``
and physical metadata for persisted input and output time series.
"""

from dataclasses import dataclass

from kompe.coefficients import CoefficientSpace

from pynamit.storage import FieldTimeSeries

INPUT_VARIABLES = {
    "boundary_jr": ("boundary_jr",),
    "boundary_Br": ("boundary_Br",),
    "conductance": ("log_conductance_magnitude", "log_hall_to_pedersen_ratio"),
    "u": ("u",),
    "Q_eff": ("Q_eff",),
    "E_neutral_wind": ("E_neutral_wind",),
}

WIND_FORCING_INPUTS = frozenset({"u", "Q_eff", "E_neutral_wind"})

INPUT_VARIABLE_ATTRS = {
    "boundary_jr": {
        "boundary_jr": {
            "units": "A m-2",
            "long_name": "upper-boundary outward radial current density",
        }
    },
    "boundary_Br": {
        "boundary_Br": {"units": "T", "long_name": "outer-boundary outward radial magnetic field"}
    },
    "conductance": {
        "log_conductance_magnitude": {
            "units": "1",
            "long_name": "log conductance magnitude relative to 1 S",
        },
        "log_hall_to_pedersen_ratio": {
            "units": "1",
            "long_name": "log Hall-to-Pedersen conductance ratio",
        },
    },
    "u": {"u": {"units": "m s-1", "long_name": "neutral wind velocity"}},
    "Q_eff": {
        "Q_eff": {"units": "A m-1", "long_name": "effective neutral-wind sheet-current forcing"}
    },
    "E_neutral_wind": {
        "E_neutral_wind": {"units": "V m-1", "long_name": "equivalent neutral-wind electric field"}
    },
}

OUTPUT_VARIABLES = {
    "dynamic": ("induced_Br", "boundary_jr", "Phi", "W"),
    "equilibrium": ("induced_Br", "boundary_jr", "Phi", "W"),
}

_OUTPUT_FIELD_ATTRS = {
    "induced_Br": {
        "units": "T",
        "long_name": "outward radial magnetic perturbation at the ionosphere",
    },
    "boundary_jr": {
        "units": "A m-2",
        "long_name": "upper-boundary outward radial current density",
    },
    "Phi": {
        "units": "V m-1",
        "long_name": "curl-free electric potential divided by ionospheric radius",
    },
    "W": {
        "units": "V m-1",
        "long_name": "divergence-free electric potential divided by ionospheric radius",
    },
}
OUTPUT_VARIABLE_ATTRS = {
    key: {name: dict(attrs) for name, attrs in _OUTPUT_FIELD_ATTRS.items()}
    for key in OUTPUT_VARIABLES
}

INPUT_DATASET_KEYS = tuple(INPUT_VARIABLES)
OUTPUT_DATASET_KEYS = tuple(OUTPUT_VARIABLES)
SIMULATION_ARTIFACT_NAMES = frozenset(
    {"settings", "gap_Br_response", *INPUT_DATASET_KEYS, *OUTPUT_DATASET_KEYS}
)


__all__ = [
    "INPUT_DATASET_KEYS",
    "INPUT_VARIABLE_ATTRS",
    "OUTPUT_DATASET_KEYS",
    "OUTPUT_VARIABLE_ATTRS",
    "SIMULATION_ARTIFACT_NAMES",
    "WIND_FORCING_INPUTS",
    "SimulationSchema",
    "build_simulation_schema",
]


@dataclass
class SimulationSchema:
    """Basis choices and physical storage metadata for one simulation.

    The mappings are ordinary dictionaries, making the complete schema
    easy to inspect interactively. The builder creates them once;
    simulation code treats them as configuration, not mutable state.
    Series created from this schema own their datasets independently.
    """

    input_variables: dict[str, tuple[str, ...]]
    output_variables: dict[str, tuple[str, ...]]
    input_field_spaces: dict[str, CoefficientSpace]
    output_field_spaces: dict[str, dict[str, CoefficientSpace]]

    def create_input_series(self, *, time_origin) -> FieldTimeSeries:
        """Create empty input streams with units and a UTC origin."""
        return FieldTimeSeries(
            self.input_field_spaces,
            self.input_variables,
            variable_attrs=INPUT_VARIABLE_ATTRS,
            time_origin=time_origin,
        )

    def create_output_series(self, *, time_origin) -> FieldTimeSeries:
        """Create empty output streams with units and a UTC origin."""
        return FieldTimeSeries(
            self.output_field_spaces,
            self.output_variables,
            variable_attrs=OUTPUT_VARIABLE_ATTRS,
            time_origin=time_origin,
        )


def build_simulation_schema(geometry, config) -> SimulationSchema:
    """Describe existing numerical objects with storage metadata."""
    sh_basis = geometry.sh_basis
    mean_free_sh_basis = geometry.poloidal_basis
    cs_basis = geometry.cs_basis
    horizontal_basis = geometry.horizontal_basis

    poloidal_space = CoefficientSpace(mean_free_sh_basis, representation="scalar", mean_free=True)
    surface_space = CoefficientSpace(horizontal_basis, representation="scalar", mean_free=True)
    tangential_space = CoefficientSpace(
        horizontal_basis, representation="helmholtz", mean_free=True
    )
    input_field_spaces = {
        "boundary_jr": surface_space,
        # Boundary Br participates in radial continuation, so it is
        # always stored in poloidal SH space.
        "boundary_Br": poloidal_space,
        # Conductance has a nonzero mean, so SH needs the full basis.
        "conductance": CoefficientSpace(
            sh_basis if config.conductance_basis == "SH" else cs_basis,
            representation="scalar",
            mean_free=False,
        ),
        "u": tangential_space,
        "Q_eff": tangential_space,
        "E_neutral_wind": tangential_space,
    }
    boundary_current_output_space = CoefficientSpace(
        horizontal_basis,
        representation="scalar",
        # In CS space the discrete Laplacian's exact range is not
        # identical to the area-mean projector. Preserve the current
        # produced by the private toroidal potential exactly so it can
        # be inverted without changing the derived sheet current.
        mean_free=False,
    )
    output_field_spaces = {
        key: {
            "induced_Br": poloidal_space,
            "boundary_jr": boundary_current_output_space,
            "Phi": surface_space,
            "W": surface_space,
        }
        for key in OUTPUT_VARIABLES
    }

    return SimulationSchema(
        input_variables=dict(INPUT_VARIABLES),
        output_variables=dict(OUTPUT_VARIABLES),
        input_field_spaces=input_field_spaces,
        output_field_spaces=output_field_spaces,
    )

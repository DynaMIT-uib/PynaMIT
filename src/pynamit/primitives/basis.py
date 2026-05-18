"""Basis interface utilities."""

from abc import ABC, abstractmethod


class Basis(ABC):
    """Abstract metadata interface for basis representations.

    Concrete basis classes may expose these fields as regular instance
    attributes. Class-level placeholders satisfy the abstract contract,
    while ``validate_metadata`` checks initialized instances.
    """

    required_attributes = (
        "kind",
        "index_names",
        "index_length",
        "index_arrays",
        "minimum_phi_sampling",
        "caching",
    )

    @property
    @abstractmethod
    def kind(self):
        """Short identifier for the basis."""
        pass

    @property
    @abstractmethod
    def index_names(self):
        """Names of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def index_length(self):
        """Total number of basis functions."""
        pass

    @property
    @abstractmethod
    def index_arrays(self):
        """Arrays of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        pass

    @property
    @abstractmethod
    def caching(self):
        """Whether basis evaluations can be cached."""
        pass

    def validate_metadata(self) -> None:
        """Validate that required basis metadata is initialized."""
        missing = [name for name in self.required_attributes if getattr(self, name, None) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{type(self).__name__} is missing basis metadata: {joined}.")


class EvaluableBasis(Basis):
    """Basis whose functions can be evaluated on a grid."""

    @abstractmethod
    def get_G(self, grid, derivative=None, cache_in=None, cache_out=False):
        """Evaluate basis functions or derivatives on ``grid``."""
        pass

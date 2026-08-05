"""The narrow historical PynaMIT evaluator compatibility surface."""

from kompe import SphericalTransform


class BasisEvaluator(SphericalTransform):
    """Compatibility spelling for :class:`kompe.SphericalTransform`.

    New code should use Kompe's descriptive synthesis properties.  The two
    historical matrix attributes below remain supported because they are used
    by existing PynaMIT collaborators.
    """

    @property
    def G(self):
        """Return the scalar synthesis matrix."""
        return self.scalar_synthesis_matrix

    @property
    def G_helmholtz(self):
        """Return the Helmholtz synthesis matrix."""
        return self.helmholtz_synthesis_matrix

__all__ = ["BasisEvaluator"]

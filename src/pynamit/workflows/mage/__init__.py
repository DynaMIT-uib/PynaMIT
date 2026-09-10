"""Prepare and project MAGE forcing for PynaMIT."""

_LAZY_EXPORTS = {
    "MAGE_MAIN_FIELD_KIND": ("pynamit.workflows.mage.projection", "MAGE_MAIN_FIELD_KIND"),
    "ForcingSettings": ("pynamit.workflows.mage.preparation", "ForcingSettings"),
    "plot_input_projection_comparison": (
        "pynamit.workflows.mage.diagnostics",
        "plot_input_projection_comparison",
    ),
    "prepare_forcing": ("pynamit.workflows.mage.preparation", "prepare_forcing"),
    "prepare_inputs": ("pynamit.workflows.mage.projection", "prepare_inputs"),
    "write_input_projection_diagnostics": (
        "pynamit.workflows.mage.diagnostics",
        "write_input_projection_diagnostics",
    ),
}


def __getattr__(name):
    """Load the requested MAGE workflow component."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return public MAGE workflow attributes including lazy exports."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["ForcingSettings", "prepare_forcing", "prepare_inputs"]

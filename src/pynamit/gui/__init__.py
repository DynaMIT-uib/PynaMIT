"""Interactive GUI for preparing, running, and exploring PynaMIT."""

_LAZY_EXPORTS = {
    "PynamitGUI": ("pynamit.gui.panel_app", "PynamitGUI"),
    "build_gui": ("pynamit.gui.panel_app", "build_gui"),
    "main": ("pynamit.gui.cli", "main"),
}


def __getattr__(name):
    """Load Panel and plotting dependencies only when requested."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return public GUI attributes including lazy exports."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["PynamitGUI", "build_gui"]

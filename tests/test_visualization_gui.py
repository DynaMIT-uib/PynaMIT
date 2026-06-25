"""Tests for the installable Panel frontend entry point."""

from pynamit.visualization.gui import build_arg_parser, default_websocket_origins


def test_pynamit_gui_parser_defaults_to_auto_detection():
    """The GUI auto-detects a run when no directory is given."""
    args = build_arg_parser().parse_args([])

    assert args.run_directory is None
    assert args.port == 5006
    assert args.route == "/pynamit"
    assert args.show is True


def test_pynamit_gui_parser_accepts_remote_no_show_options():
    """The GUI command should support remote/headless serving."""
    args = build_arg_parser().parse_args(
        ["run-a", "--address", "0.0.0.0", "--port", "6006", "--no-show"]
    )

    assert args.run_directory == "run-a"
    assert args.address == "0.0.0.0"
    assert args.port == 6006
    assert args.show is False


def test_default_websocket_origins_allow_localhost_and_loopback():
    """Local serving should work for localhost and 127.0.0.1."""
    origins = default_websocket_origins("127.0.0.1", 5006)

    assert origins == ["localhost:5006", "127.0.0.1:5006"]


def test_default_websocket_origins_keep_explicit_origins():
    """Explicit websocket origins should be kept."""
    origins = default_websocket_origins(
        "0.0.0.0", 6006, ["myhost.example:6006", "localhost:6006"]
    )

    assert origins == [
        "localhost:6006",
        "127.0.0.1:6006",
        "0.0.0.0:6006",
        "myhost.example:6006",
    ]

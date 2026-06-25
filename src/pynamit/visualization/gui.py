"""Command-line entry point for the PynaMIT Panel frontend."""

from __future__ import annotations

import argparse
from pathlib import Path


def default_websocket_origins(
    address: str, port: int, extra_origins: list[str] | None = None
) -> list[str]:
    """Return websocket origins for local browser access."""
    hosts = ["localhost", "127.0.0.1", str(address)]
    origins = []
    for host in hosts:
        origin = f"{host}:{int(port)}"
        if origin not in origins:
            origins.append(origin)
    for origin in extra_origins or []:
        if origin not in origins:
            origins.append(origin)
    return origins


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the ``pynamit-gui`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="pynamit-gui",
        description="Serve the PynaMIT Panel visualization frontend.",
    )
    parser.add_argument(
        "run_directory",
        nargs="?",
        default=None,
        help=(
            "Run or projected-input directory to inspect. Defaults to the current "
            "directory when it contains PynaMIT artifacts, otherwise common local "
            "run directories are tried."
        ),
    )
    parser.add_argument("--port", type=int, default=5006, help="Port for the Panel server.")
    parser.add_argument(
        "--address", default="127.0.0.1", help="Address for the Panel server to bind."
    )
    parser.add_argument(
        "--route", default="/pynamit", help="URL route for the app, for example /pynamit."
    )
    parser.add_argument("--title", default="PynaMIT", help="Browser/page title.")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the app in a browser when the server starts.",
    )
    parser.add_argument(
        "--websocket-origin",
        action="append",
        default=None,
        help="Additional allowed websocket origin. May be supplied more than once.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Serve the Panel app."""
    args = build_arg_parser().parse_args(argv)
    try:
        import panel as pn
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "pynamit-gui requires Panel. Install with `pip install 'pynamit[gui]'` "
            "or install panel in the active environment."
        ) from exc

    from pynamit.visualization.panel_app import build_pynamit_panel_app

    route = "/" + str(args.route).strip("/")
    run_directory = Path(args.run_directory).expanduser() if args.run_directory else None
    app = build_pynamit_panel_app(run_directory=run_directory)
    serve_kwargs = {
        "address": args.address,
        "port": int(args.port),
        "show": bool(args.show),
        "title": str(args.title),
        "websocket_origin": default_websocket_origins(
            args.address, int(args.port), args.websocket_origin
        ),
    }
    label = run_directory if run_directory is not None else "auto-detected run directory"
    print(f"Serving PynaMIT GUI for {label} at http://{args.address}:{args.port}{route}")
    pn.serve({route: app}, **serve_kwargs)


if __name__ == "__main__":  # pragma: no cover
    main()

"""Lightweight HISCA entry point; model dependencies live in a private environment."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("hisca-web")
except PackageNotFoundError:
    __version__ = "0.3.0"

def launch(**kwargs):
    """Start the local web app and keep it running until interrupted."""
    from .cli import launch as _launch
    return _launch(**kwargs)

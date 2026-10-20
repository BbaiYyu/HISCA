
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("hisca-web")
except PackageNotFoundError:
    __version__ = "0.3.1"

def launch(**kwargs):

    from .cli import launch as _launch
    return _launch(**kwargs)

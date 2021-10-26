from typing import Optional, Union

from . import builtins, runtime, tracing


__all__ = ["builtins", "runtime", "tracing"]

from packaging.version import LegacyVersion, Version, parse
from pkg_resources import DistributionNotFound, get_distribution


try:
    __version__: str = get_distribution("gt4py").version
except DistributionNotFound:
    __version__ = "X.X.X.unknown"

__versioninfo__: Optional[Union[LegacyVersion, Version]] = parse(__version__)

del DistributionNotFound, LegacyVersion, Version, get_distribution, parse

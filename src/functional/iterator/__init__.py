from typing import Optional, Union

from packaging.version import LegacyVersion, Version, parse
from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

from . import tracing  # noqa: F401 # ignore unused


try:
    __version__: str = get_distribution("gt4py").version
except DistributionNotFound:
    __version__ = "X.X.X.unknown"

__versioninfo__: Optional[Union[LegacyVersion, Version]] = parse(__version__)

del DistributionNotFound, LegacyVersion, Version, get_distribution, parse

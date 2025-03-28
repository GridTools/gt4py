# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Package metadata: version, authors, license and copyright."""

import importlib.metadata as _imp_metadata
from typing import Final

from packaging import version as pkg_version


__all__ = ["__author__", "__copyright__", "__license__", "__version__", "__version_info__"]


__author__: Final = "ETH Zurich and individual contributors"
__copyright__: Final = "Copyright (c) 2014-2024 ETH Zurich"
__license__: Final = "BSD-3-Clause"

# This should be overwritten by the actual version at build time
# using the versioningit onbuild hook
__on_build_version: Final = "0.0.0+missing.version.info"

if dist := _imp_metadata.distribution("gt4py"):
    _version: str = dist.version

    if contents := dist.read_text("direct_url.json"):
        # This branch should only be executed in editable installs.
        # In this case, the version reported by `gt4py.__version__`
        # is directly computed from the status of the `git` repository
        # with the source code, if available, which might differ from the
        # version reported by `importlib.metadata.version("gt4py")`
        # (and `$ pip show gt4py`).
        try:
            import json as _json

            import versioningit as _versioningit

            _url_data = _json.loads(contents)
            assert _url_data["dir_info"]["editable"] is True
            assert _url_data["url"].startswith("file://")

            _src_path = _url_data["url"][7:]
            _version = _versioningit.get_version(_src_path)
        except Exception:
            pass

else:
    # Fallback to the static or default version
    _version = __on_build_version

__version__: Final[str] = _version
__version_info__: Final = pkg_version.parse(__version__)

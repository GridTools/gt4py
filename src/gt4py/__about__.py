# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Package metadata: version, authors, license and copyright."""

import pathlib
from typing import Any, Final

from packaging import version as pkg_version


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
]


__author__: Final = "ETH Zurich and individual contributors"
__copyright__: Final = "Copyright (c) 2014-2024 ETH Zurich"
__license__: Final = "BSD-3-Clause"

__version__: str
__version_info__: pkg_version.Version


#: This value should be overwritten with the actual version string at build
#: time by the `onbuild` hook of versioningit. If the hook is not run for
#: whatever reason, the current value defined here would be used as fallback.
#: Therefore, for consistency, the version hard-coded here should be kept in
#: sync with the `tool.versioningit.default-version` field in `pyproject.toml`.
on_build_version: Final = "1.1.7+unknown.version.details"

_cached_version_data: tuple[str, pkg_version.Version] | None = None
_dir: list[str] | None = None


def _get_version_from_versioningit(path: pathlib.Path | str) -> str:
    import os

    import versioningit

    devnull = open(os.devnull, "w")
    try:
        old_stdout_fd = os.dup(1)  # duplicate current stdout fd
        os.dup2(devnull.fileno(), 1)  # replace fd 1 with /dev/null
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 2)

        version = versioningit.get_version(path)

    finally:
        os.dup2(old_stdout_fd, 1)
        os.close(old_stdout_fd)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)

    return version


def _inspect_version() -> tuple[str, pkg_version.Version]:
    global _cached_version_data

    if _cached_version_data is not None:
        return _cached_version_data

    import importlib.metadata

    if dist := importlib.metadata.distribution("gt4py"):
        version: str = dist.version  # static installation version

        if contents := dist.read_text("direct_url.json"):
            # This branch should only be executed in editable installs.
            # In this case, the version reported by `gt4py.__version__`
            # is directly computed from the status of the `git` repository
            # with the source code, if available, which might differ from the
            # version reported by `importlib.metadata.version("gt4py")`
            # (and `$ pip show gt4py`).
            try:
                import json

                url_data = json.loads(contents)
                if url_data["dir_info"]["editable"] is not True:
                    raise ValueError("Not an editable install")
                if not url_data["url"].startswith("file://"):
                    raise ValueError("Not a local install")

                src_path = url_data["url"][7:]
                version = _get_version_from_versioningit(src_path)
            except Exception:
                # There is something wrong in the current editable installation.
                # Fallback to the static version, but don't cache the result as the
                # final version data, since the package is installed in editable mode.
                pass

            version_info = pkg_version.parse(version)

            return version, version_info

        else:
            # If the package is not installed in editable mode, the version
            # is always correctly reported by `importlib.metadata.version("gt4py")`.
            _cached_version_data = version, pkg_version.parse(version)

            return _cached_version_data

    else:
        # This branch is a weird case: since `importlib.metadata`
        # couldn't find the distribution info, it should only be executed
        # if the package is not properly installed but it runs anyway.
        # For example, when running directly from sources by adding the
        # project root to the `sys.path`.
        try:
            version = _get_version_from_versioningit(pathlib.Path(__file__).parent.resolve())
        except Exception:
            # Fallback to the on-build version, if everything else fails.
            version = on_build_version

        version_info = pkg_version.parse(version)

        return version, version_info


def __getattr__(name: str) -> Any:
    match name:
        case "__version__":
            return _inspect_version()[0]
        case "__version_info__":
            return _inspect_version()[1]
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    global _dir
    if _dir is None:
        import sys

        # Add virtual attributes (e.g. __version__, __version_info__) to the list
        annotations = globals().get("__annotations__", {})
        _dir = list(set(annotations.keys()) | vars(sys.modules[__name__]).keys())

    return _dir

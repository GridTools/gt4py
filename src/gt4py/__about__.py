# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Package metadata: version, authors, license and copyright."""

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


#: This should be overwritten by the `onbuild` hook of versioningit
# with the actual version string at build time. If the hook is not
# run for any reason, the fallback value defined here would be used,
# so, for consistency, it should be set to the same value as the one
# in `tool.versioningit.default-version` in pyproject.toml.
on_build_version: Final = "0.0.0+missing.version.info"

_static_version: tuple[str, pkg_version.Version] | None = None
_dir: list[str] | None = None


def _inspect_version() -> tuple[str, pkg_version.Version]:
    global _static_version

    if _static_version is not None:
        return _static_version

    import importlib.metadata

    import versioningit

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
                version = versioningit.get_version(src_path)

            except Exception:
                # There is something wrong in the current editable installation.
                # Fallback to the static version, but don't store the result as a
                # static version, since the package is installed in editable mode.
                pass

            version_info = pkg_version.parse(version)

            return version, version_info

        else:
            # If the package is not installed in editable mode, the version
            # is the one reported by `importlib.metadata.version("gt4py")`.
            _static_version = version, pkg_version.parse(version)

            return _static_version

    else:
        # This branch is a weird case: since `importlib.metadata`
        # couldn't find the distribution info, it should only be executed
        # if the package is not properly installed but it runs anyway.
        # For example, when running directly from sources by adding the
        # project root to the `sys.path`.
        try:
            import pathlib

            version = versioningit.get_version(pathlib.Path(__file__).parent.resolve())
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
        _dir = list(set(__annotations__.keys()) | vars(sys.modules[__name__]).keys())

    return _dir

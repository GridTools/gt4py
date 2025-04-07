# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re
import pytest
from unittest.mock import patch

import gt4py
from gt4py.__about__ import _inspect_version, on_build_version


def test_version():
    """Test the version string attribute."""
    assert isinstance(gt4py.__version__, str)
    assert len(gt4py.__version__) and all(len(p) for p in gt4py.__version__.split("."))
    assert gt4py.__about__.__version__ == gt4py.__version__


def test_version_info():
    """Test the parsed version info attribute."""
    from packaging.version import Version

    assert isinstance(gt4py.__version_info__, Version)
    assert gt4py.__version_info__.release == tuple(
        int(p) for p in re.split("[\.\+]", gt4py.__version__)[:3]
    )
    assert gt4py.__version__.startswith(gt4py.__version_info__.public)
    assert gt4py.__version__.endswith(gt4py.__version_info__.local)
    assert gt4py.__about__.__version_info__ == gt4py.__version_info__


def test_inspect_version_static():
    """Test version with a static installation."""
    with patch("importlib.metadata.distribution") as dist_mock:
        dist_mock.return_value.version = "1.2.3"

        version, version_info = _inspect_version()
        assert version == "1.2.3"
        assert str(version_info) == "1.2.3"


def test_inspect_version_editable():
    """Test version with an editable installation."""
    with (
        patch("importlib.metadata.distribution") as dist_mock,
        patch("versioningit.get_version") as get_version_mock,
    ):
        dist_mock.return_value.version = "1.2.3"
        dist_mock.return_value.read_text.return_value = (
            '{"dir_info": {"editable": true}, "url": "file:///some/path"}'
        )
        get_version_mock.return_value = "1.2.3.post42+e174a1f.dirty"

        version, version_info = _inspect_version()
        assert version == "1.2.3.post42+e174a1f.dirty"
        assert str(version_info) == "1.2.3.post42+e174a1f.dirty"


def test_inspect_version_fallback():
    """Test version fallback to `on_build_version`."""
    with (
        patch("importlib.metadata.distribution", return_value=None),
        patch("versioningit.get_version", side_effect=Exception),
    ):
        version, version_info = _inspect_version()
        assert version == on_build_version
        assert str(version_info) == on_build_version

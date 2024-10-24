# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Package metadata: version, authors, license and copyright."""

from typing import Final

from packaging import version as pkg_version


__all__ = ["__author__", "__copyright__", "__license__", "__version__", "__version_info__"]


__author__: Final = "ETH Zurich and individual contributors"
__copyright__: Final = "Copyright (c) 2014-2024 ETH Zurich"
__license__: Final = "BSD-3-Clause"


__version__: Final = "1.0.4"
__version_info__: Final = pkg_version.parse(__version__)

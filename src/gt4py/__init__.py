# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Python API to develop performance portable applications for weather and climate."""

import typing


__copyright__: typing.Final = "Copyright (c) 2014-2022 ETH Zurich"
__license__: typing.Final = "GPLv3+"

from .version import __version__, __version_info__  # isort:skip


from . import config, gtscript, storage
from .stencil_object import StencilObject

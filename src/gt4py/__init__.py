# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

__copyright__ = "Copyright (c) 2014-2020 ETH Zurich"
__license__ = "GPLv3+"


from pkg_resources import DistributionNotFound, get_distribution


# Disable isort to avoid circular imports
# isort: off
from . import config
from . import utils

#
from . import definitions
from . import gtscript

#
from . import ir
from . import analysis
from . import frontend
from . import backend
from . import stencil_object
from . import loader
from . import storage
from . import caching

from .definitions import AccessKind, Boundary, DomainInfo, FieldInfo, ParameterInfo, CartesianSpace
from .stencil_object import StencilObject

# isort: on


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

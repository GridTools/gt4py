# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

# Disable isort to avoid circular imports
# isort: off
from .base import *
from .module_generator import BaseModuleGenerator
from . import python_generator

# isort: on

from .debug_backend import *
from .gt_backends import *
from .gtc_backend import *
from .numpy_backend import *


try:
    import dawn4py

    from .dawn_backends import *
except ImportError:
    pass  # dawn4py not installed

from . import python_generator

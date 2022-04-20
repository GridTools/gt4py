# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Constant information about the current interpreter."""

from __future__ import annotations

import sys
from typing import Final


IS_PYTHON_AT_LEAST_3_8: Final = sys.version_info >= (3, 8)
IS_PYTHON_AT_LEAST_3_9: Final = sys.version_info >= (3, 9)
IS_PYTHON_AT_LEAST_3_10: Final = sys.version_info >= (3, 10)
IS_PYTHON_AT_LEAST_3_11: Final = sys.version_info >= (3, 11)
IS_PYTHON_3_8: Final = IS_PYTHON_AT_LEAST_3_8 and not IS_PYTHON_AT_LEAST_3_9
IS_PYTHON_3_9: Final = IS_PYTHON_AT_LEAST_3_9 and not IS_PYTHON_AT_LEAST_3_10
IS_PYTHON_3_10: Final = IS_PYTHON_AT_LEAST_3_10 and not IS_PYTHON_AT_LEAST_3_11
IS_PYTHON_3_11: Final = IS_PYTHON_AT_LEAST_3_11 and not sys.version_info > (3, 11)

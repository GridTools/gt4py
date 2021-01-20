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

"""GridTools storages classes."""

from .creation_routines import (
    as_storage,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    storage,
    zeros,
    zeros_like,
)
from .default_parameters import StorageDefaults, register_storage_defaults
from .definitions import Storage

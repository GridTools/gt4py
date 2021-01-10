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

"""GridTools storages classes."""

from .default_parameters import register_storage_defaults, StorageDefaults
from .definitions import Storage
from .creation_routines import (
    storage,
    as_storage,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
    Storage,
)

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

from __future__ import annotations

import attrs
from attrs.validators import *  # noqa  # unused star import for reexporting


@attrs.define(repr=False, frozen=True, slots=True)
class _NonEmptyValidator:
    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not value:
            raise ValueError(f"Empty '{attr.name}' value")

    def __repr__(self):
        return "<non_empty validator>"


def non_empty():
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is longer than *length*.

    :param int length: Maximum length of the string or iterable

    .. versionadded:: 21.3.0
    """
    return _NonEmptyValidator()

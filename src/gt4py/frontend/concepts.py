# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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

import abc

from gt4py import utils as gt_utils


REGISTRY = gt_utils.Registry()


def from_name(name: str):
    return REGISTRY.get(name, None)


def register(frontend_cls):
    assert issubclass(frontend_cls, Frontend) and frontend_cls.name is not None
    return REGISTRY.register(frontend_cls.name, frontend_cls)


class Frontend(abc.ABC):
    name = None

    @classmethod
    @abc.abstractmethod
    def get_stencil_id(cls, qualified_name, definition, externals, options_id):
        pass

    @classmethod
    @abc.abstractmethod
    def generate(cls, definition, options):
        pass

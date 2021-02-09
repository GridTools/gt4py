# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any

import eve
from gtc import oir
from gtc.cuir import cuir


class OIRToCUIR(eve.NodeTranslator):
    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> cuir.FieldDecl:
        return cuir.FieldDecl(name=node.name, dtype=node.dtype)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> cuir.Program:
        return cuir.Program(name=node.name, params=self.visit(node.params))

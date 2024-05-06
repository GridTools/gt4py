# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


from dataclasses import dataclass

import dace

from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtins.gtir_builtin_translator import (
    GTIRBuiltinTranslator,
)
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class GTIRBuiltinDomain(GTIRBuiltinTranslator):
    def build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        raise NotImplementedError

    def visit_domain(self, node: itir.Expr) -> list[tuple[Dimension, str, str]]:
        """
        Specialized visit method for domain expressions.

        Returns a list of dimensions and the corresponding range.
        """
        assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))

        domain = []
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            assert len(named_range.args) == 3
            axis = named_range.args[0]
            assert isinstance(axis, itir.AxisLiteral)
            dim = Dimension(axis.value)
            bounds = [self.visit(arg) for arg in named_range.args[1:3]]
            domain.append((dim, bounds[0], bounds[1]))

        return domain

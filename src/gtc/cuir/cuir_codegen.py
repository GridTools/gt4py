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

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import LeafNode
from gtc.cuir import cuir


class CUIRCodegen(codegen.TemplatedGenerator):
    FieldDecl = as_fmt("{name}")

    Program = as_mako(
        """#include <array>

        namespace ${name}_impl_{
            using Domain = std::array<unsigned, 3>;

            auto ${name}(Domain domain){
                return [domain](${','.join('auto&& ' + p for p in params)}){

                };
            };
        }

        using ${name}_impl_::${name};
        """
    )

    @classmethod
    def apply(cls, root: LeafNode, **kwargs: Any) -> str:
        if not isinstance(root, cuir.Program):
            raise ValueError("apply() requires gtcpp.Progam root node")
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.eve.codegen import FormatTemplate as as_fmt, TemplatedGenerator
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import apply_common_transforms
from gt4py.next.program_processors.processor_interface import program_formatter


class ToLispLike(TemplatedGenerator):
    Sym = as_fmt("{id}")
    FunCall = as_fmt("({fun} {' '.join(args)})")
    Literal = as_fmt("{value}")
    OffsetLiteral = as_fmt("{value}")
    SymRef = as_fmt("{id}")
    StencilClosure = as_fmt(
        """(
     :domain {domain}
     :stencil {stencil}
     :output {output}
     :inputs {' '.join(inputs)}
    )
    """
    )
    FencilDefinition = as_fmt(
        """
        ({' '.join(function_definitions)})
        (defen {id}({' '.join(params)})
        {''.join(closures)})
        """
    )
    FunctionDefinition = as_fmt(
        """(defun {id}({' '.join(params)})
        {expr}
        )

"""
    )
    Lambda = as_fmt(
        """(lambda ({' '.join(params)})
         {expr}
          )"""
    )

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        transformed = apply_common_transforms(
            root, lift_mode=kwargs.get("lift_mode"), offset_provider=kwargs["offset_provider"]
        )
        generated_code = super().apply(transformed, **kwargs)
        try:
            from yasi import indent_code  # type: ignore[import]

            indented = indent_code(generated_code, "--dialect lisp")
            return "".join(indented["indented_code"])
        except ImportError:
            return generated_code


@program_formatter
def format_lisp(program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> str:
    return ToLispLike.apply(program, **kwargs)

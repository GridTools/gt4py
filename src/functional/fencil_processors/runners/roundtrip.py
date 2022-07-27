# GT4Py Project - GridTools Framework
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
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import textwrap
from typing import TYPE_CHECKING

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from eve.concepts import Node
from functional.iterator.ir import AxisLiteral, FencilDefinition, OffsetLiteral
from functional.iterator.processor_interface import fencil_executor
from functional.iterator.transforms import apply_common_transforms


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Optional

    from functional.common import Dimension
    from functional.iterator.embedded import NeighborTableOffsetProvider


class EmbeddedDSL(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    Literal = as_fmt("{value}")
    NoneLiteral = as_fmt("None")
    OffsetLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako("(lambda ${','.join(params)}: ${expr})")
    StencilClosure = as_mako("closure(${domain}, ${stencil}, ${output}, [${','.join(inputs)}])")
    FencilDefinition = as_mako(
        """
${''.join(function_definitions)}
@fendef
def ${id}(${','.join(params)}):
    ${'\\n    '.join(closures)}
    """
    )
    FunctionDefinition = as_mako(
        """
@fundef
def ${id}(${','.join(params)}):
    return ${expr}
    """
    )


# TODO this wrapper should be replaced by an extension of the IR
class WrapperGenerator(EmbeddedDSL):
    def visit_FencilDefinition(self, node: FencilDefinition, *, tmps):
        params = self.visit(node.params)
        non_tmp_params = [param for param in params if param not in tmps]

        body = []
        for tmp, domain in tmps.items():
            axis_literals = [named_range.args[0].value for named_range in domain.args]
            origin = (
                "{"
                + ", ".join(
                    f"{named_range.args[0].value}: -{self.visit(named_range.args[1])}"
                    for named_range in domain.args
                )
                + "}"
            )
            shape = (
                "("
                + ", ".join(
                    f"{self.visit(named_range.args[2])}-{self.visit(named_range.args[1])}"
                    for named_range in domain.args
                )
                + ")"
            )
            body.append(
                f"{tmp} = np_as_located_field({','.join(axis_literals)}, origin={origin})(np.full({shape}, np.nan))"
            )

        body.append(f"{node.id}({','.join(params)}, **kwargs)")
        body_str = "\n    ".join(body)
        return f"\ndef {node.id}_wrapper({','.join(non_tmp_params)}, **kwargs):\n    {body_str}\n"


_BACKEND_NAME = "roundtrip"

_FENCIL_CACHE: dict[int, Callable] = {}


def fencil_generator(
    ir: Node,
    debug: bool,
    use_tmps: bool,
    use_embedded: bool,
    offset_provider: dict[str, NeighborTableOffsetProvider],
) -> Callable:
    # TODO(tehrengruber): just a temporary solution until we have a proper generic
    #  caching mechanism
    cache_key = hash((ir, use_tmps, debug, use_embedded, tuple(offset_provider.items())))
    if cache_key in _FENCIL_CACHE:
        return _FENCIL_CACHE[cache_key]

    tmps = {}

    def register_tmp(tmp, domain):
        tmps[tmp] = domain

    ir = apply_common_transforms(
        ir, use_tmps=use_tmps, offset_provider=offset_provider, register_tmp=register_tmp
    )

    program = EmbeddedDSL.apply(ir)
    wrapper = WrapperGenerator.apply(ir, tmps=tmps)
    offset_literals: Iterable[str] = (
        ir.pre_walk_values()
        .if_isinstance(OffsetLiteral)
        .getattr("value")
        .if_isinstance(str)
        .to_set()
    )
    axis_literals: Iterable[str] = (
        ir.pre_walk_values().if_isinstance(AxisLiteral).getattr("value").to_set()
    )

    if use_embedded:
        builtins_import = "from functional.iterator.embedded import *"
    else:
        builtins_import = "from functional.iterator.builtins import *"

    header = textwrap.dedent(
        f"""
        import numpy as np
        {builtins_import}
        from functional.iterator.runtime import *
        from functional.iterator.embedded import np_as_located_field
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as source_file:
        source_file_name = source_file.name
        if debug:
            print(source_file_name)
        offset_literals = [f'{o} = offset("{o}")' for o in offset_literals]
        axis_literals = [f'{o} = CartesianAxis("{o}")' for o in axis_literals]
        source_file.write(header)
        source_file.write("\n".join(offset_literals))
        source_file.write("\n")
        source_file.write("\n".join(axis_literals))
        source_file.write("\n")
        source_file.write(program)
        source_file.write(wrapper)

    try:
        spec = importlib.util.spec_from_file_location("module.name", source_file_name)
        foo = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(foo)  # type: ignore
    finally:
        if not debug:
            pathlib.Path(source_file_name).unlink(missing_ok=True)

    assert isinstance(ir, FencilDefinition)
    fencil_name = ir.id
    fencil = getattr(foo, fencil_name + "_wrapper")

    _FENCIL_CACHE[cache_key] = fencil

    return fencil


@fencil_executor
def executor(
    ir: Node,
    *args,
    column_axis: Optional[Dimension] = None,
    debug: bool = False,
    use_tmps: bool = False,
    dispatch_backend: Optional[str] = None,
    offset_provider: dict[str, NeighborTableOffsetProvider],
) -> None:
    fencil = fencil_generator(
        ir,
        offset_provider=offset_provider,
        debug=debug,
        use_tmps=use_tmps,
        use_embedded=dispatch_backend is None,
    )

    new_kwargs = {"offset_provider": offset_provider, "column_axis": column_axis}
    if dispatch_backend:
        new_kwargs["backend"] = dispatch_backend

    return fencil(*args, **new_kwargs)

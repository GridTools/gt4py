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

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import textwrap
from collections.abc import Callable, Iterable
from typing import Any, Optional

import gt4py.eve.codegen as codegen
import gt4py.next.allocators as next_allocators
import gt4py.next.common as common
import gt4py.next.iterator.embedded as embedded
import gt4py.next.iterator.ir as itir
import gt4py.next.iterator.transforms as itir_transforms
import gt4py.next.iterator.transforms.global_tmps as gtmps_transform
import gt4py.next.program_processors.otf_compile_executor as otf_compile_executor
import gt4py.next.program_processors.processor_interface as ppi
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako


def _create_tmp(axes, origin, shape, dtype):
    if isinstance(dtype, tuple):
        return f"({','.join(_create_tmp(axes, origin, shape, dt) for dt in dtype)},)"
    else:
        return (
            f"gtx.as_field([{axes}], np.empty({shape}, dtype=np.dtype('{dtype}')), origin={origin})"
        )


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

    # extension required by global_tmps
    def visit_FencilWithTemporaries(self, node, **kwargs):
        params = self.visit(node.params)

        tmps = "\n    ".join(self.visit(node.tmps))
        args = ", ".join(params + [tmp.id for tmp in node.tmps])
        params = ", ".join(params)
        fencil = self.visit(node.fencil)
        return (
            fencil
            + "\n"
            + f"def {node.fencil.id}_wrapper({params}, **kwargs):\n    "
            + tmps
            + f"\n    {node.fencil.id}({args}, **kwargs)\n"
        )

    def visit_Temporary(self, node, **kwargs):
        assert isinstance(node.domain, itir.FunCall) and node.domain.fun.id in (
            "cartesian_domain",
            "unstructured_domain",
        )
        assert all(
            isinstance(r, itir.FunCall) and r.fun == itir.SymRef(id="named_range")
            for r in node.domain.args
        )
        domain_ranges = [self.visit(r.args) for r in node.domain.args]
        axes = ", ".join(label for label, _, _ in domain_ranges)
        origin = "{" + ", ".join(f"{label}: -{start}" for label, start, _ in domain_ranges) + "}"
        shape = "(" + ", ".join(f"{stop}-{start}" for _, start, stop in domain_ranges) + ")"
        return f"{node.id} = {_create_tmp(axes, origin, shape, node.dtype)}"


_BACKEND_NAME = "roundtrip"

_FENCIL_CACHE: dict[int, Callable] = {}


def fencil_generator(
    ir: itir.Node,
    debug: bool,
    lift_mode: itir_transforms.LiftMode,
    use_embedded: bool,
    offset_provider: dict[str, embedded.NeighborTableOffsetProvider],
) -> Callable:
    """
    Generate a directly executable fencil from an ITIR node.

    Arguments:
        ir: The iterator IR (ITIR) node.
        debug: Keep module source containing fencil implementation.
        lift_mode: Change the way lifted function calls are evaluated.
        use_embedded: Directly use builtins from embedded backend instead of
                      generic dispatcher. Gives faster performance and is easier
                      to debug.
        offset_provider: A mapping from offset names to offset providers.
    """
    # TODO(tehrengruber): just a temporary solution until we have a proper generic
    #  caching mechanism
    cache_key = hash((ir, lift_mode, debug, use_embedded, tuple(offset_provider.items())))
    if cache_key in _FENCIL_CACHE:
        return _FENCIL_CACHE[cache_key]

    ir = itir_transforms.apply_common_transforms(
        ir, lift_mode=lift_mode, offset_provider=offset_provider
    )

    program = EmbeddedDSL.apply(ir)

    # format output in debug mode for better debuggability
    # (e.g. line numbers, overview in the debugger).
    if debug:
        program = codegen.format_python_source(program)

    offset_literals: Iterable[str] = (
        ir.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .getattr("value")
        .if_isinstance(str)
        .to_set()
    )
    axis_literals: Iterable[str] = (
        ir.pre_walk_values().if_isinstance(itir.AxisLiteral).getattr("value").to_set()
    )

    if use_embedded:
        builtins_import = "from gt4py.next.iterator.embedded import *"
    else:
        builtins_import = "from gt4py.next.iterator.builtins import *"

    header = textwrap.dedent(
        f"""
        import numpy as np
        import gt4py.next as gtx
        {builtins_import}
        from gt4py.next.iterator.runtime import *
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as source_file:
        source_file_name = source_file.name
        if debug:
            print(source_file_name)
        offset_literals = [f'{o} = offset("{o}")' for o in offset_literals]
        axis_literals = [f'{o} = gtx.Dimension("{o}")' for o in axis_literals]
        source_file.write(header)
        source_file.write("\n".join(offset_literals))
        source_file.write("\n")
        source_file.write("\n".join(axis_literals))
        source_file.write("\n")
        source_file.write(program)

    try:
        spec = importlib.util.spec_from_file_location("module.name", source_file_name)
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        if not debug:
            pathlib.Path(source_file_name).unlink(missing_ok=True)

    assert isinstance(ir, (itir.FencilDefinition, gtmps_transform.FencilWithTemporaries))
    fencil_name = (
        ir.fencil.id + "_wrapper"
        if isinstance(ir, gtmps_transform.FencilWithTemporaries)
        else ir.id
    )
    fencil = getattr(mod, fencil_name)

    _FENCIL_CACHE[cache_key] = fencil

    return fencil


def execute_roundtrip(
    ir: itir.Node,
    *args,
    column_axis: Optional[common.Dimension] = None,
    offset_provider: dict[str, embedded.NeighborTableOffsetProvider],
    debug: bool = False,
    lift_mode: itir_transforms.LiftMode = itir_transforms.LiftMode.FORCE_INLINE,
    dispatch_backend: Optional[ppi.ProgramExecutor] = None,
) -> None:
    fencil = fencil_generator(
        ir,
        offset_provider=offset_provider,
        debug=debug,
        lift_mode=lift_mode,
        use_embedded=dispatch_backend is None,
    )

    new_kwargs: dict[str, Any] = {
        "offset_provider": offset_provider,
        "column_axis": column_axis,
    }
    if dispatch_backend:
        new_kwargs["backend"] = dispatch_backend

    return fencil(*args, **new_kwargs)


executor = ppi.program_executor(execute_roundtrip)  # type: ignore[arg-type]

backend = otf_compile_executor.OTFBackend(
    executor=executor, allocator=next_allocators.StandardCPUFieldBufferAllocator()
)

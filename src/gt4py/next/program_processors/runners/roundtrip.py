# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import pathlib
import tempfile
import textwrap
import typing
from collections.abc import Callable, Iterable
from typing import Any, Optional

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next import allocators as next_allocators, backend as next_backend, common, config
from gt4py.next.ffront import foast_to_gtir, foast_to_past, past_to_itir
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import stages, workflow
from gt4py.next.type_system import type_specifications as ts


def _create_tmp(axes: str, origin: str, shape: str, dtype: ts.TypeSpec) -> str:
    if isinstance(dtype, ts.TupleType):
        return f"({','.join(_create_tmp(axes, origin, shape, dt) for dt in dtype.types)},)"
    else:
        assert isinstance(dtype, ts.ScalarType)
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
    FunctionDefinition = as_mako(
        """
@fundef
def ${id}(${','.join(params)}):
    return ${expr}
    """
    )
    Program = as_mako(
        """
${''.join(function_definitions)}
@fendef
def ${id}(${','.join(params)}):
    ${'\\n    '.join(declarations)}
    ${'\\n    '.join(body)}
    """
    )
    SetAt = as_mako("set_at(${expr}, ${domain}, ${target})")
    IfStmt = as_mako("""if_stmt(${cond}, 
        lambda: [${','.join(true_branch)}],
        lambda: [${','.join(false_branch)}]
    )""")

    def visit_Temporary(self, node: itir.Temporary, **kwargs: Any) -> str:
        assert (
            isinstance(node.domain, itir.FunCall)
            and isinstance(node.domain.fun, itir.SymRef)
            and node.domain.fun.id in ("cartesian_domain", "unstructured_domain")
        )
        assert all(
            isinstance(r, itir.FunCall) and r.fun == itir.SymRef(id="named_range")
            for r in node.domain.args
        )
        domain_ranges = [self.visit(r.args) for r in node.domain.args]  # type: ignore[attr-defined] # `node.domain` is `FunCall` checked in previous assert
        axes = ", ".join(label for label, _, _ in domain_ranges)
        origin = "{" + ", ".join(f"{label}: -{start}" for label, start, _ in domain_ranges) + "}"
        shape = "(" + ", ".join(f"{stop}-{start}" for _, start, stop in domain_ranges) + ")"
        assert node.dtype
        return f"{node.id} = {_create_tmp(axes, origin, shape, node.dtype)}"


_FENCIL_CACHE: dict[int, Callable] = {}


def fencil_generator(
    ir: itir.Program | itir.FencilDefinition,
    debug: bool,
    use_embedded: bool,
    offset_provider: common.OffsetProvider,
    transforms: itir_transforms.ITIRTransform,
) -> stages.CompiledProgram:
    """
    Generate a directly executable fencil from an ITIR node.

    Arguments:
        ir: The iterator IR (ITIR) node.
        debug: Keep module source containing fencil implementation.
        extract_temporaries: Extract intermediate field values into temporaries.
        use_embedded: Directly use builtins from embedded backend instead of
                      generic dispatcher. Gives faster performance and is easier
                      to debug.
        offset_provider: A mapping from offset names to offset providers.
    """
    # TODO(tehrengruber): just a temporary solution until we have a proper generic
    #  caching mechanism
    cache_key = hash(
        (
            ir,
            transforms,
            debug,
            use_embedded,
            tuple(common.offset_provider_to_type(offset_provider).items()),
        )
    )
    if cache_key in _FENCIL_CACHE:
        if debug:
            print(f"Using cached fencil for key {cache_key}")
        return typing.cast(stages.CompiledProgram, _FENCIL_CACHE[cache_key])

    ir = transforms(ir, offset_provider=offset_provider)

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
    axis_literals_set: Iterable[itir.AxisLiteral] = (
        ir.pre_walk_values().if_isinstance(itir.AxisLiteral).to_set()
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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as source_file:
        source_file_name = source_file.name
        if debug:
            print(source_file_name)
        offset_literals = [f'{o} = offset("{o}")' for o in offset_literals]
        axis_literals = [
            f'{o.value} = gtx.Dimension("{o.value}", kind=gtx.DimensionKind("{o.kind}"))'
            for o in axis_literals_set
        ]
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

    assert isinstance(ir, itir.Program)
    fencil_name = ir.id
    fencil = getattr(mod, fencil_name)

    _FENCIL_CACHE[cache_key] = fencil

    return typing.cast(stages.CompiledProgram, fencil)


@dataclasses.dataclass(frozen=True)
class Roundtrip(workflow.Workflow[stages.CompilableProgram, stages.CompiledProgram]):
    debug: Optional[bool] = None
    use_embedded: bool = True
    dispatch_backend: Optional[next_backend.Backend] = None
    transforms: itir_transforms.ITIRTransform = itir_transforms.apply_common_transforms  # type: ignore[assignment] # TODO(havogt): cleanup interface of `apply_common_transforms`

    def __call__(self, inp: stages.CompilableProgram) -> stages.CompiledProgram:
        debug = config.DEBUG if self.debug is None else self.debug

        fencil = fencil_generator(
            inp.data,
            offset_provider=inp.args.offset_provider,
            debug=debug,
            use_embedded=self.use_embedded,
            transforms=self.transforms,
        )

        def decorated_fencil(
            *args: Any,
            offset_provider: dict[str, common.Connectivity | common.Dimension],
            out: Any = None,
            column_axis: Optional[common.Dimension] = None,
            **kwargs: Any,
        ) -> None:
            if out is not None:
                args = (*args, out)
            if not column_axis:  # TODO(tehrengruber): This variable is never used. Bug?
                column_axis = inp.args.column_axis
            fencil(
                *args,
                offset_provider=offset_provider,
                backend=self.dispatch_backend,
                column_axis=inp.args.column_axis,
                **kwargs,
            )

        return decorated_fencil


# TODO(tehrengruber): introduce factory
default = next_backend.Backend(
    name="roundtrip",
    executor=Roundtrip(
        transforms=functools.partial(
            itir_transforms.apply_common_transforms,
            extract_temporaries=False,
        )
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)
with_temporaries = next_backend.Backend(
    name="roundtrip_with_temporaries",
    executor=Roundtrip(
        transforms=functools.partial(
            itir_transforms.apply_common_transforms,
            extract_temporaries=True,
        )
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)
no_transforms = next_backend.Backend(
    name="roundtrip",
    executor=Roundtrip(transforms=lambda o, *, offset_provider: o),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)


gtir = next_backend.Backend(
    name="roundtrip_gtir",
    executor=Roundtrip(transforms=itir_transforms.apply_fieldview_transforms),  # type: ignore[arg-type] # on purpose doesn't support `FencilDefintion` will resolve itself later...
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.Transforms(
        past_to_itir=past_to_itir.past_to_itir_factory(to_gtir=True),
        foast_to_itir=foast_to_gtir.adapted_foast_to_gtir_factory(cached=True),
        field_view_op_to_prog=foast_to_past.operator_to_program_factory(
            foast_to_itir_step=foast_to_gtir.adapted_foast_to_gtir_factory()
        ),
    ),
)
foast_to_gtir_step = foast_to_gtir.adapted_foast_to_gtir_factory(cached=True)
